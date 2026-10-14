"""Dependency-invalidated prefix scores and stable FCFS tie breaking."""

import heapq
import time
from collections import Counter, defaultdict

from vllm.v1.core.demand_policy import Counters


class HitQueue:
    def __init__(self, runtime):
        self.runtime = runtime
        self.requests = {}
        self.keys = {}
        self.dependencies = {}
        self.watchers = defaultdict(set)
        self.dirty = set()
        self.heap = []
        self.age_heap = []
        self.versions = Counter()
        self.serial = 0
        self.stats = Counters(getattr(runtime, "telemetry", True))

    def add_request(self, request):
        rid = request.request_id
        if not hasattr(request, "_demand_sequence"):
            request._demand_sequence = self.runtime.sequence
            self.runtime.sequence += 1
            request._demand_enqueued = time.monotonic()
        self.requests[rid] = request
        self.dirty.add(rid)
        heapq.heappush(self.age_heap, (request._demand_enqueued, rid))

    prepend_request = add_request

    def prepend_requests(self, requests):
        for request in requests:
            self.add_request(request)

    def invalidate(self, key):
        affected = self.watchers.get(key, ())
        self.stats["invalidations"] += len(affected)
        self.dirty.update(affected)

    def _forget_dependencies(self, rid):
        for key in self.dependencies.pop(rid, ()):
            owners = self.watchers[key]
            owners.discard(rid)
            if not owners:
                del self.watchers[key]

    def refresh(self):
        started = time.perf_counter_ns()
        for rid in sorted(self.dirty):
            request = self.requests.get(rid)
            if request is None:
                continue
            self._forget_dependencies(rid)
            hit, deps = self.runtime.score(request)
            self.dependencies[rid] = deps
            for key in deps:
                self.watchers[key].add(rid)
            self.serial += 1
            self.versions[rid] = self.serial
            key = (hit, request._demand_sequence)
            self.keys[rid] = key
            heapq.heappush(self.heap, (*key, self.versions[rid], rid))
            self.stats["lookups"] += 1
        self.dirty.clear()
        if len(self.heap) > 2 * len(self.requests) + 64:
            self.stats["cleanup_items"] += len(self.heap)
            self.heap = [(*self.keys[r], self.versions[r], r) for r in self.requests]
            heapq.heapify(self.heap)
        while self.heap:
            *_, version, rid = self.heap[0]
            if rid in self.requests and self.versions[rid] == version:
                break
            heapq.heappop(self.heap)
        if len(self.age_heap) > 2 * len(self.requests) + 64:
            self.age_heap = [
                (r._demand_enqueued, rid) for rid, r in self.requests.items()
            ]
            heapq.heapify(self.age_heap)
        while self.age_heap:
            age, rid = self.age_heap[0]
            if rid in self.requests and self.requests[rid]._demand_enqueued == age:
                break
            heapq.heappop(self.age_heap)
        if self.age_heap:
            age = time.monotonic() - self.age_heap[0][0]
            self.stats["oldest_pending_us"] = int(age * 1e6)
            self.stats["pending_over_1s_observations"] += int(age >= 1)
        self.stats["refresh_ns"] += time.perf_counter_ns() - started
        self.stats["max_size"] = max(self.stats["max_size"], len(self.requests))

    def peek_request(self):
        if len(self.requests) == 1:
            request = next(iter(self.requests.values()))
            age = time.monotonic() - request._demand_enqueued
            self.stats["oldest_pending_us"] = int(age * 1e6)
            self.stats["pending_over_1s_observations"] += int(age >= 1)
            self.stats["max_size"] = max(self.stats["max_size"], 1)
            return request
        self.refresh()
        return self.requests[self.heap[0][-1]]

    def peek_key(self):
        self.refresh()
        return self.keys[self.heap[0][-1]]

    def pop_request(self):
        request = self.peek_request()
        return self.pop_selected(request)

    def pop_selected(self, request):
        """Remove the admitted request even if its allocation changed scores."""
        assert self.requests[request.request_id] is request
        self.remove_request(request)
        waited = time.monotonic() - request._demand_enqueued
        self.stats["max_wait_us"] = max(self.stats["max_wait_us"], int(waited * 1e6))
        self.stats["wait_over_1s"] += int(waited >= 1)
        return request

    def remove_request(self, request):
        rid = request.request_id
        self.requests.pop(rid, None)
        self.keys.pop(rid, None)
        self.versions.pop(rid, None)
        self.dirty.discard(rid)
        self._forget_dependencies(rid)
        if hasattr(self.runtime, "probes"):
            self.runtime.probes.pop(rid, None)
        # Queue-global versions prevent ABA when a preempted request returns.

    def remove_requests(self, requests):
        for request in requests:
            self.remove_request(request)

    def __bool__(self):
        return bool(self.requests)

    def __len__(self):
        return len(self.requests)

    def __iter__(self):
        self.refresh()
        return iter(
            sorted(self.requests.values(), key=lambda r: self.keys[r.request_id])
        )
