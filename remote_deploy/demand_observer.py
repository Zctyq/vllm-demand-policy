"""Identical observation hooks for native and demand experiment engines."""

import hashlib
import json
import os
import time
from pathlib import Path

from demand_metadata import parse_request_id
from vllm.v1.core.sched.scheduler import Scheduler


class ObservedScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_log = Path(os.environ["DEMAND_EXPERIMENT_LOG"])
        self.dp_log.mkdir(parents=True, exist_ok=True)
        self.dp_history, self.dp_admission = {}, {}
        self.dp_steps = self.dp_schedule_ns = self.dp_preemptions = 0
        self.dp_scheduled_tokens = self.dp_native_victims = 0
        manager = self.kv_cache_manager
        original = manager.record_prefix_cache_stats

        def record(request, hits):
            result = original(request, hits)
            if request.request_id not in self.dp_admission:
                marker = parse_request_id(request.request_id)
                if marker:
                    hashes = request.block_hashes
                    history = self.dp_history.get(marker["session_id"], set())
                    k = 0
                    for h in hashes[: (request.num_tokens - 1) // self.hash_block_size]:
                        if h not in history:
                            break
                        k += self.hash_block_size
                    self.dp_admission[request.request_id] = dict(
                        **marker,
                        prompt_tokens=request.num_prompt_tokens,
                        H=hits,
                        K=k,
                        historical_hits=min(hits, k),
                    )
            return result

        manager.record_prefix_cache_stats = record
        pool = manager.block_pool
        original_evict = pool._maybe_evict_cached_block

        def evict(block):
            result = original_evict(block)
            self.dp_native_victims += int(result)
            return result

        pool._maybe_evict_cached_block = evict
        root = Path(__import__("vllm").__file__).parent
        paths = [
            "v1/core/sched/scheduler.py",
            "v1/core/block_pool.py",
            "v1/core/kv_cache_manager.py",
        ]
        runtime = getattr(self, "demand_runtime", None)
        if runtime:
            paths += [
                "v1/core/demand_policy.py",
                "v1/core/demand_runtime.py",
                "v1/core/demand_queue.py",
                "v1/core/demand_metadata.py",
            ]
        self.emit(
            "engine.json",
            {
                "version": __import__("vllm").__version__,
                "root": str(root),
                "hashes": {
                    p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                    for p in paths
                },
                "runtime": runtime is not None,
                "waiting": bool(runtime and runtime.waiting),
                "eviction": bool(runtime and runtime.eviction),
                "num_blocks": pool.num_gpu_blocks,
                "hash_block_size": self.hash_block_size,
                "groups": [
                    type(g.kv_cache_spec).__name__
                    for g in self.kv_cache_config.kv_cache_groups
                ],
                "async_scheduling": self.vllm_config.scheduler_config.async_scheduling,
            },
        )

    def emit(self, filename, value, append=False):
        with (self.dp_log / filename).open("a" if append else "w") as f:
            f.write(json.dumps(value) + "\n")

    def schedule(self, *args, **kwargs):
        begin = time.perf_counter_ns()
        output = super().schedule(*args, **kwargs)
        self.dp_schedule_ns += time.perf_counter_ns() - begin
        self.dp_steps += 1
        self.dp_scheduled_tokens += sum(output.num_scheduled_tokens.values())
        return output

    def _free_request(self, request, delay_free_blocks=False):
        marker = parse_request_id(request.request_id)
        if marker:
            history = self.dp_history.setdefault(marker["session_id"], set())
            history.update(
                request.block_hashes[
                    : request.num_computed_tokens // self.hash_block_size
                ]
            )
            row = self.dp_admission.pop(request.request_id, dict(**marker))
            row.update(
                output_tokens=request.num_output_tokens,
                preemptions=request.num_preemptions,
                finished_status=str(request.status),
            )
            self.dp_preemptions += request.num_preemptions
            self.emit("requests.jsonl", row, True)
        result = super()._free_request(request, delay_free_blocks)
        runtime = getattr(self, "demand_runtime", None)
        self.emit(
            "latest.json",
            {
                "schedule_steps": self.dp_steps,
                "schedule_wall_ns": self.dp_schedule_ns,
                "scheduled_tokens": self.dp_scheduled_tokens,
                "native_victims": self.dp_native_victims,
                "preemptions": self.dp_preemptions,
                "demand": dict(runtime.stats) if runtime else None,
                "policy": dict(runtime.policy.stats)
                if runtime and runtime.policy
                else None,
            },
        )
        return result

    def reset_prefix_cache(self, *args, **kwargs):
        result = super().reset_prefix_cache(*args, **kwargs)
        if result:
            runtime = getattr(self, "demand_runtime", None)
            if runtime:
                runtime.stats.clear()
                for queue in runtime.queues:
                    queue.stats.clear()
            self.dp_history.clear()
            self.dp_admission.clear()
            self.dp_steps = self.dp_schedule_ns = self.dp_preemptions = 0
            self.dp_scheduled_tokens = self.dp_native_victims = 0
        return result

    def shutdown(self):
        runtime = getattr(self, "demand_runtime", None)
        self.emit(
            "summary.json",
            {
                "schedule_steps": self.dp_steps,
                "schedule_wall_ns": self.dp_schedule_ns,
                "scheduled_tokens": self.dp_scheduled_tokens,
                "native_victims": self.dp_native_victims,
                "preemptions": self.dp_preemptions,
                "demand": runtime.diagnostics() if runtime else None,
            },
        )
        return super().shutdown()
