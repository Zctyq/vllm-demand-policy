"""Single-threaded FullAttention adapter for demand policy and waiting order."""

from __future__ import annotations

import itertools
import os
import sys
import time

from vllm.v1.core.demand_policy import Counters, DemandPolicy, Plan
from vllm.v1.core.demand_queue import HitQueue


class DemandRuntime:
    def __init__(self, pool, *, eviction, waiting=False, clock=time.monotonic):
        self.pool = pool
        self.eviction = eviction
        self.waiting = waiting
        self.clock = clock
        self.telemetry = os.getenv("VLLM_DEMAND_TELEMETRY", "1") != "0"
        self.policy = (
            DemandPolicy(os.getenv("VLLM_DEMAND_TELEMETRY", "1") != "0")
            if eviction
            else None
        )
        self.clean = (
            dict.fromkeys(
                b.block_id
                for b in pool.free_block_queue.get_all_free_blocks()
                if b.block_hash is None
            )
            if eviction
            else None
        )
        self.reserved = None
        self.queues = []
        self.sequence = 0
        self.probes = {}
        self.stats = Counters(self.telemetry)
        self.mutating = False
        self.fail_after = None  # CPU fault injection; not an environment switch.

    def changed(self, hashes):
        for key in hashes:
            for queue in self.queues:
                for rid in queue.watchers.get(key, ()):
                    self.probes.pop(rid, None)
                queue.invalidate(key)

    def inserted(self, block, key):
        self.changed([key])
        if self.policy is not None:
            self.clean.pop(block.block_id, None)
            if block.block_id in self.policy.blocks:
                # Unexpected alias/promotion in a supported configuration:
                # preserve relationships but make the block non-recyclable.
                existing = self.policy.blocks[block.block_id]
                existing.known = existing.full = False
                self.policy.refresh(existing.owners)
                return
            self.policy.cache(block.block_id, self.clock(), active=block.ref_cnt)
            self.policy.blocks[block.block_id].key = key

    def removed(self, block, keys):
        self.changed(keys)
        if self.policy is not None and not self.mutating:
            self.policy.remove(block.block_id)
            if not block.ref_cnt and not block.is_null:
                self.clean[block.block_id] = None

    def reference_changed(self, block):
        if self.policy is None:
            return
        self.policy.set_active(block.block_id, block.ref_cnt)
        if block.ref_cnt or block.is_null or block.block_hash is not None:
            self.clean.pop(block.block_id, None)
        else:
            self.clean[block.block_id] = None

    def attach(self, request, blocks, *, reused=False):
        if self.policy is None:
            return
        from vllm.v1.core.demand_metadata import parse_request_id

        session = getattr(request, "session_id", None)
        if session is None:
            marker = parse_request_id(request.request_id)
            session = marker["session_id"] if marker else None
        bids = [b.block_id for b in blocks if b.block_id in self.policy.blocks]
        if session is None:
            for bid in bids:
                block = self.policy.blocks[bid]
                block.known = False
                self.policy.refresh(block.owners)
            self.policy.count("unknown_blocks_observed", len(bids))
        else:
            self.policy.attach(str(session), bids)
        if reused:
            now = self.clock()
            for bid in bids:
                self.policy.reuse(bid, now)

    def reserve(self, num_blocks, excluded=()):
        """Allocate atomically before native admission mutates hit references."""
        assert self.reserved is None
        if num_blocks < 0:
            raise ValueError("negative physical demand")
        self.stats["N"] += num_blocks
        if not num_blocks:
            self.reserved = []
            return True
        if num_blocks > self.pool.get_num_free_blocks():
            return False
        clean_ids = list(itertools.islice(self.clean, num_blocks))
        demand = num_blocks - len(clean_ids)
        self.stats["U_used"] += len(clean_ids)
        self.stats["D"] += demand
        if demand:
            plan = self.policy.plan(demand, self.clock(), excluded)
            if plan is None:
                self.stats["insufficient_safe_supply"] += 1
                return False
        else:
            self.stats["clean_fast_path"] += 1
            plan = Plan([], [], dict(self.policy.cursor))
        ids = clean_ids + plan.ids
        blocks = [self.pool.blocks[i] for i in ids]
        if len(set(ids)) != num_blocks or any(
            b.ref_cnt
            or b.is_null
            or b.prev_free_block is None
            or b.next_free_block is None
            or self.pool.cached_block_hashes_by_block.get(b.block_id)
            for b in blocks
        ):
            self.stats["validation_failure"] += 1
            return False
        if any(self.pool.blocks[i].block_hash is not None for i in clean_ids):
            self.stats["validation_failure"] += 1
            return False
        if any(
            i not in self.policy.blocks
            or self.pool.blocks[i].block_hash is None
            or self.policy.blocks[i].key != self.pool.blocks[i].block_hash
            or not self.pool.cached_block_hash_to_block.contain(
                self.pool.blocks[i].block_hash, i
            )
            for i in plan.ids
        ):
            self.stats["validation_failure"] += 1
            return False
        if any(
            (
                self.policy.blocks[i].version,
                tuple(sorted(self.policy.blocks[i].owners)),
                self.policy.blocks[i].known,
                self.policy.blocks[i].full,
            )
            != stamp
            for i, stamp in plan.stamps.items()
        ):
            self.stats["validation_failure"] += 1
            return False
        self.reserved = self._commit(blocks, plan)
        return self.reserved is not None

    def _commit(self, blocks, plan):
        pool = self.pool
        queue = pool.free_block_queue
        trail, removed = [], []
        hash_backup = {}
        for block in blocks:
            key = block.block_hash
            if key is not None and key not in hash_backup:
                value = pool.cached_block_hash_to_block._cache[key]
                hash_backup[key] = value.copy() if isinstance(value, dict) else value
        self.mutating = True
        try:
            for block in blocks:
                saved = (
                    block,
                    block.prev_free_block,
                    block.next_free_block,
                    block.block_hash,
                    block.block_hash_num_tokens,
                )
                queue.remove(block)
                trail.append(saved)
                key = block.block_hash
                if key is not None:
                    pool.cached_block_hash_to_block.pop(key, block.block_id)
                    block.reset_hash()
                    removed.append(key)
                block.ref_cnt = 1
                if self.fail_after == len(trail):
                    raise RuntimeError("injected commit failure")
        except Exception:
            for block, prev, following, key, tokens in reversed(trail):
                block.ref_cnt = 0
                block.prev_free_block, block.next_free_block = prev, following
                prev.next_free_block = following.prev_free_block = block
                queue.num_free_blocks += 1
                if key is not None:
                    block.reset_hash()
                    block.set_block_hash(key, num_tokens=tokens)
            for key, value in hash_backup.items():
                pool.cached_block_hash_to_block._cache[key] = value
            self.stats["rollback"] += 1
            return None
        finally:
            self.mutating = False
        self.policy.commit(plan)
        for block in blocks:
            self.clean.pop(block.block_id, None)
        self.changed(removed)
        try:
            pool._emit_block_removed_events(removed)
            if pool.metrics_collector:
                for block in blocks:
                    pool.metrics_collector.on_block_evicted(block)
                    pool.metrics_collector.on_block_allocated(block)
        except Exception:
            # Allocation has committed. A telemetry failure must not make the
            # caller retry and lose another batch of cached blocks.
            self.stats["notification_errors"] += 1
        self.stats["planned_victims"] += len(plan.ids)
        self.stats["actual_victims"] += len(removed)
        assert len(plan.ids) == len(removed)
        return blocks

    def allocate(self, count):
        if self.reserved is None and not self.reserve(count):
            raise ValueError("Insufficient safe demand-policy blocks")
        if len(self.reserved) != count:
            raise RuntimeError("Native physical demand differs from reservation")
        blocks, self.reserved = self.reserved, None
        return blocks

    def reset(self):
        self.probes.clear()
        for queue in self.queues:
            queue.dirty.update(queue.requests)
        if self.policy is not None:
            self.policy = DemandPolicy(self.policy.telemetry)
            self.clean = dict.fromkeys(
                b.block_id
                for b in self.pool.free_block_queue.get_all_free_blocks()
                if b.block_hash is None
            )
        self.reserved = None

    def diagnostics(self):
        """Explicit O(B+S) snapshot, never called by allocation/scheduling."""
        result = dict(self.stats)
        if self.policy is not None:
            p = self.policy
            result.update(
                policy=dict(p.stats),
                cached_blocks=len(p.blocks),
                sessions=len(p.names),
                expiry_heap=len(p.deadlines),
                unknown_cached_blocks=sum(
                    not b.known or not b.owners for b in p.blocks.values()
                ),
                active_cached_blocks=sum(b.active > 0 for b in p.blocks.values()),
                shared_cached_blocks=sum(len(b.owners) > 1 for b in p.blocks.values()),
            )
        result["queues"] = [dict(q.stats) for q in self.queues]
        return result

    def score(self, request):
        manager = self.manager
        if request.num_computed_tokens or not manager.prefix_cache_lookup_enabled(
            request
        ):
            return 0, set()
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        probe = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        self.stats["prefix_queries"] += 1
        self.probes[request.request_id] = probe
        _, hits, _ = probe
        limit = min(len(request.block_hashes), hits // self.pool.hash_block_size + 1)
        deps = {
            make_block_hash_with_group_id(h, 0) for h in request.block_hashes[:limit]
        }
        return hits, deps

    def metadata_bytes(self):
        """Approximate reachable policy storage; explicit offline probe."""
        seen = set()

        def size(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            result = sys.getsizeof(obj)
            if isinstance(obj, dict):
                return result + sum(size(k) + size(v) for k, v in obj.items())
            if isinstance(obj, (list, tuple, set)):
                return result + sum(size(v) for v in obj)
            if hasattr(obj, "__dict__"):
                result += size(vars(obj))
            return result

        return size(self.policy) + size(self.clean)

    def select_queue(self):
        available = [q for q in self.queues if q]
        if len(available) == 1:
            return available[0]
        return min(available, key=lambda q: q.peek_key()) if available else None


def create_runtime(scheduler):
    waiting = os.getenv("VLLM_WAITING_HIT_ASC", "0") == "1"
    eviction = os.getenv("VLLM_DEMAND_EVICTION", "0") == "1"
    if not waiting and not eviction:
        return None
    config = scheduler.vllm_config
    groups = scheduler.kv_cache_config.kv_cache_groups
    reasons = []
    if config.scheduler_config.async_scheduling or config.max_concurrent_batches != 1:
        reasons.append("synchronous scheduling and one in-flight batch required")
    if config.kv_transfer_config or config.ec_transfer_config:
        reasons.append("connectors unsupported")
    if config.speculative_config or getattr(
        config.cache_config, "kv_offloading_size", 0
    ):
        reasons.append("speculation/offload unsupported")
    if not config.cache_config.enable_prefix_caching:
        reasons.append("prefix caching required")
    if (
        len(groups) != 1
        or type(groups[0].kv_cache_spec).__name__ != "FullAttentionSpec"
    ):
        reasons.append("one FullAttention group required")
    elif groups[0].kv_cache_spec.block_size != scheduler.hash_block_size:
        reasons.append("partial-cache aliases unsupported")
    elif any(
        getattr(groups[0].kv_cache_spec, name, None)
        for name in ("sliding_window", "attention_chunk_size", "non_causal")
    ):
        reasons.append("causal full attention without sparse windows required")
    if scheduler.is_encoder_decoder or scheduler.is_mm_encoder_only:
        reasons.append("decoder-only required")
    if scheduler.dcp_world_size != 1 or scheduler.pcp_world_size != 1:
        reasons.append("context parallelism unsupported")
    if reasons:
        raise ValueError("Demand policy: " + "; ".join(reasons))
    runtime = DemandRuntime(
        scheduler.kv_cache_manager.block_pool, eviction=eviction, waiting=waiting
    )
    runtime.manager = scheduler.kv_cache_manager
    if waiting:
        scheduler.waiting = HitQueue(runtime)
        scheduler.skipped_waiting = HitQueue(runtime)
        runtime.queues = [scheduler.waiting, scheduler.skipped_waiting]
    return runtime
