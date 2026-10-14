# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in bridge between the experimental policy and synchronous V1 scheduling."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from vllm.v1.core.progressive_policy import ProgressivePolicy


def parse_request_id(request_id: str) -> dict[str, Any] | None:
    """Decode the replay marker while allowing vLLM's ID prefix and suffix."""
    marker = re.search(r"pe1\.([A-Za-z0-9_-]+)", request_id)
    if marker is None:
        return None
    encoded = marker.group(1)
    # Completions add -<prompt-index>; InputProcessor then adds -<8 hex>.
    for length in range(len(encoded), max(0, len(encoded) - 128), -1):
        candidate = encoded[:length]
        try:
            raw = base64.urlsafe_b64decode(candidate + "=" * (-len(candidate) % 4))
            value = json.loads(raw)
            if (
                isinstance(value, dict)
                and set(value) == {"r", "s", "c"}
                and base64.urlsafe_b64encode(raw).decode().rstrip("=") == candidate
                and isinstance(value.get("r"), str)
                and isinstance(value.get("s"), str)
                and isinstance(value.get("c"), int)
                and not isinstance(value["c"], bool)
                and value["c"] >= 0
            ):
                return {
                    "run_id": value["r"],
                    "session_id": value["s"],
                    "call_index": value["c"],
                    "request_id": "pe1." + candidate,
                }
        except (ValueError, UnicodeError, TypeError):
            continue
    return None


class BlockPoolAdapter:
    """Validate all physical copies before using native eviction and queue APIs."""

    def __init__(self, pool, block_size: int, generations=None, runtime=None):
        self.pool = pool
        self.block_size = block_size
        self.generations = (
            generations if generations is not None else [0] * len(pool.blocks)
        )
        self.runtime = runtime

    def copies(self, key: bytes) -> list[tuple[int, int]]:
        value = self.pool.cached_block_hash_to_block._cache.get(key)
        if value is None:
            return []
        blocks = value.values() if isinstance(value, dict) else (value,)
        return [(block.block_id, self.generations[block.block_id]) for block in blocks]

    def check(self, key: bytes, expected_copies: list[tuple[int, int]]) -> str | None:
        actual = self.copies(key)
        if not actual:
            return "not_resident"
        for block_id, _ in actual:
            block = self.pool.blocks[block_id]
            if block.ref_cnt != 0:
                return "active"
        if sorted(actual) != sorted(expected_copies):
            return "identity"
        for block_id, generation in actual:
            block = self.pool.blocks[block_id]
            if block.is_null:
                return "null"
            if generation != self.generations[block_id] or block.block_hash != key:
                return "identity"
            if self.pool.cached_block_hashes_by_block.get(block_id):
                return "alias"
            length = block.block_hash_num_tokens
            if not length or length % self.block_size:
                return "partial"
            previous = block.prev_free_block
            following = block.next_free_block
            if (
                previous is None
                or following is None
                or previous.next_free_block is not block
                or following.prev_free_block is not block
            ):
                return "not_free"
        return None

    def evict(self, key: bytes, expected_copies: list[tuple[int, int]]) -> int:
        if self.check(key, expected_copies) is not None:
            return 0
        blocks = [self.pool.blocks[block_id] for block_id, _ in expected_copies]
        free_before = self.free_count()
        if self.runtime is not None:
            self.runtime._remove_reason = "policy"
        try:
            self.pool.evict_blocks({block.block_id for block in blocks})
        finally:
            if self.runtime is not None:
                self.runtime._remove_reason = "reset"
        if self.copies(key):
            raise RuntimeError("Progressive eviction left a physical hash copy live")
        for block in blocks:
            self.pool.free_block_queue.remove(block)
        self.pool.free_block_queue.prepend_n(blocks)
        if self.free_count() != free_before:
            raise RuntimeError("Progressive eviction changed free block capacity")
        return len(blocks)

    def free_count(self) -> int:
        return self.pool.get_num_free_blocks()


def _keys(request, count: int | None = None) -> list[bytes]:
    hashes = request.block_hashes if count is None else request.block_hashes[:count]
    return [bytes(block_hash) + b"\0\0\0\0" for block_hash in hashes]


class ProgressiveRuntime:
    """Buffer request/checkpoint logs and keep GPU execution evidence separate."""

    def __init__(self, scheduler, config: dict[str, Any]):
        self.scheduler = scheduler
        self.pool = scheduler.kv_cache_manager.block_pool
        self.config = config
        self.block_size = config["block_size"]
        self.generations = [0] * len(self.pool.blocks)
        self.adapter = BlockPoolAdapter(
            self.pool, self.block_size, self.generations, self
        )
        self.policy = ProgressivePolicy(
            block_size=self.block_size,
            mode=config["mode"],
            check_interval=config["check_interval"],
            first_ratio=config["first_cold_ratio"],
            later_ratio=config["later_cold_ratio"],
        )
        self.requests: dict[str, dict[str, Any]] = {}
        self._remove_reason = "reset"
        self.log_dir = Path(config["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.log_dir / "run_config.json"
        if config_path.exists():
            raise ValueError(
                "VLLM_PE_LOG_DIR already contains a run; choose a new path"
            )
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        self.files = {
            name: (self.log_dir / name).open("a", buffering=65536)
            for name in ("requests.jsonl", "policy_checks.jsonl", "reset.jsonl")
        }
        print(
            "[progressive-eviction] " + json.dumps(config, ensure_ascii=False),
            flush=True,
        )

    def _write(self, name: str, record: dict[str, Any]) -> None:
        stream = self.files[name]
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        stream.flush()

    def _state(self, request) -> dict[str, Any]:
        state = self.requests.get(request.request_id)
        if state is not None:
            return state
        marker = parse_request_id(request.request_id)
        if marker is not None and marker["run_id"] != self.config["run_id"]:
            raise ValueError("Replay request run_id does not match VLLM_PE_RUN_ID")
        if getattr(request, "resumable", False):
            raise ValueError("Progressive prototype does not support streaming input")
        params = request.sampling_params
        if params is not None and getattr(params, "n", 1) != 1:
            raise ValueError("Progressive replay requires one output per request")
        state = {
            "request_id": marker["request_id"] if marker else request.request_id,
            "engine_request_id": request.request_id,
            "session_id": marker["session_id"] if marker else None,
            "call_index": marker["call_index"] if marker else None,
            "run_id": self.config["run_id"],
            "input_tokens": request.num_prompt_tokens,
            "historical_prefix_tokens": 0,
            "first_lookup_hit_tokens": None,
            "adopted_prefix_hit_tokens": None,
            "actual_prefix_hit_tokens": None,
            "historical_cache_hit_tokens": None,
            "committed_blocks": 0,
            "admissions": 0,
            "executed": False,
            "finished": False,
            "lookup_observed": False,
        }
        self.requests[request.request_id] = state
        return state

    def on_lookup(self, request, hit_tokens: int) -> None:
        state = self._state(request)
        if state["lookup_observed"]:
            return
        state["lookup_observed"] = True
        state["first_lookup_hit_tokens"] = hit_tokens
        state["historical_prefix_tokens"] = self.policy.observe_access(
            request.request_id,
            state["session_id"],
            _keys(request),
            max(0, request.num_prompt_tokens - 1),
        )

    def on_admit(self, request, hit_tokens: int) -> None:
        state = self._state(request)
        state["admissions"] += 1
        if state["admissions"] > 1:
            state["committed_blocks"] = 0
        if state["adopted_prefix_hit_tokens"] is None:
            state["adopted_prefix_hit_tokens"] = hit_tokens

    def on_computed(self, request, blocks, computed_tokens: int) -> None:
        state = self._state(request)
        if not state["executed"]:
            state["executed"] = True
            adopted = state["adopted_prefix_hit_tokens"] or 0
            state["actual_prefix_hit_tokens"] = adopted
            state["historical_cache_hit_tokens"] = min(
                adopted, state["historical_prefix_tokens"]
            )
        count = max(0, computed_tokens) // self.block_size
        if count <= state["committed_blocks"]:
            return
        keys = _keys(request, count)
        if len(keys) != count or len(blocks) != 1 or len(blocks[0]) < count:
            raise RuntimeError(
                "Executed prefix does not match the single-group block table"
            )
        copies = {}
        for key, block in zip(keys, blocks[0][:count]):
            if block.is_null or block.block_hash != key:
                raise RuntimeError("Executed full block has unexpected hash identity")
            copies[key] = self.adapter.copies(key)
        self.policy.commit(request.request_id, state["session_id"], keys, copies)
        state["committed_blocks"] = count

    def on_finish(self, request) -> None:
        state = self._state(request)
        if state["finished"]:
            return
        state["finished"] = True
        status = request.status.name
        success = status in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
        self.policy.finish(request.request_id, success)
        self._write(
            "requests.jsonl",
            {
                **{
                    key: value
                    for key, value in state.items()
                    if key != "lookup_observed"
                },
                "output_tokens": len(request.output_token_ids),
                "history_reusable_tokens": state["historical_prefix_tokens"],
                "adopted_hit_tokens": state["historical_cache_hit_tokens"],
                "success": success,
                "status": status,
                "preemptions": request.num_preemptions,
                "timestamp": time.monotonic(),
            },
        )
        self._save_totals()

    def maybe_check(self) -> None:
        if self.scheduler.deferred_frees or any(
            request.num_in_flight_tokens for request in self.scheduler.requests.values()
        ):
            raise RuntimeError("Progressive check reached a boundary with in-flight KV")
        event = self.policy.maybe_check(self.adapter)
        if event is not None:
            self._write(
                "policy_checks.jsonl", {"run_id": self.config["run_id"], **event}
            )
            self._save_totals()

    def on_remove(self, block, removed_hashes) -> None:
        if removed_hashes:
            self.policy.on_remove(
                block.block_id,
                self.generations[block.block_id],
                list(removed_hashes),
                self._remove_reason,
            )

    def on_native_recycle(self, block) -> None:
        self._remove_reason = "native"

    def on_allocate(self, block) -> None:
        self.generations[block.block_id] += 1
        self._remove_reason = "reset"

    def _save_totals(self) -> None:
        destination = self.log_dir / "eviction_summary.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.policy.totals, indent=2) + "\n")
        temporary.replace(destination)

    def reset(self) -> None:
        self.policy.reset()
        self.requests.clear()
        self._write(
            "reset.jsonl",
            {
                "run_id": self.config["run_id"],
                "success": True,
                "completed": 0,
                "free_blocks": self.adapter.free_count(),
                "timestamp": time.monotonic(),
            },
        )
        self._save_totals()

    def close(self) -> None:
        self._save_totals()
        for stream in self.files.values():
            stream.close()


def _configuration(scheduler, mode: str, log_dir: str) -> dict[str, Any]:
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise ValueError(
            "Set VLLM_USE_FLASHINFER_SAMPLER=0 explicitly for both experiment modes"
        )
    from vllm import envs

    use_flashinfer_sampler = envs.VLLM_USE_FLASHINFER_SAMPLER
    if use_flashinfer_sampler is not False:
        raise ValueError(
            "VLLM_USE_FLASHINFER_SAMPLER resolved enabled; restart with it set to 0"
        )
    config = scheduler.vllm_config
    parallel = config.parallel_config
    cache = config.cache_config
    groups = scheduler.kv_cache_config.kv_cache_groups
    reasons = []
    if config.scheduler_config.async_scheduling or config.max_concurrent_batches != 1:
        reasons.append(
            "synchronous scheduling with max_concurrent_batches=1 is required"
        )
    if config.kv_transfer_config is not None or config.ec_transfer_config is not None:
        reasons.append("KV/EC transfer connectors are unsupported")
    if getattr(cache, "kv_offloading_size", None):
        reasons.append("KV offloading is unsupported")
    if config.speculative_config is not None:
        reasons.append("speculative decoding is unsupported")
    if not cache.enable_prefix_caching:
        reasons.append("prefix caching must remain enabled")
    if any(
        getattr(parallel, key, 1) != 1
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        reasons.append("the prototype requires one GPU and one engine")
    if len(groups) != 1:
        reasons.append("exactly one KV cache group is supported")
    else:
        spec = groups[0].kv_cache_spec
        specs = getattr(spec, "kv_cache_specs", {"single": spec}).values()
        if any(
            type(item).__name__ != "FullAttentionSpec"
            or getattr(item, "sliding_window", None) is not None
            or getattr(item, "attention_chunk_size", None) is not None
            or getattr(item, "non_causal", False)
            for item in specs
        ):
            reasons.append(
                "only causal full attention with no sliding window is supported"
            )
        if spec.block_size != scheduler.hash_block_size:
            reasons.append("hash_block_size must equal physical block_size")
    if scheduler.is_encoder_decoder or scheduler.is_mm_encoder_only:
        reasons.append("text decoder-only requests are required")
    if not config.model_config.enforce_eager:
        reasons.append("CUDA Graph must be disabled with enforce_eager")
    backend = str(getattr(config.attention_config, "backend", ""))
    if backend in ("", "None"):
        reasons.append(
            "an explicit verified non-FlashInfer attention backend is required"
        )
    if "FLASHINFER" in backend.upper():
        reasons.append("FlashInfer is excluded from this experiment")
    if reasons:
        raise ValueError(
            "Unsafe progressive experiment configuration: " + "; ".join(reasons)
        )

    import vllm

    paths = {}
    hashes = {}
    for name in (
        "sched.scheduler",
        "kv_cache_manager",
        "block_pool",
        "progressive_policy",
        "progressive_runtime",
    ):
        module = sys.modules.get("vllm.v1.core." + name)
        filename = getattr(module, "__file__", None)
        paths[name] = filename
        if filename:
            hashes[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    gpu_name = None
    try:
        from vllm.platforms import current_platform

        gpu_name = current_platform.get_device_name()
    except Exception:
        pass
    return {
        "run_id": os.environ.get("VLLM_PE_RUN_ID", "unspecified"),
        "mode": mode,
        "policy": mode,
        "log_dir": str(Path(log_dir).resolve()),
        "check_interval": int(os.environ.get("VLLM_PE_CHECK_INTERVAL", "10")),
        "first_cold_ratio": float(os.environ.get("VLLM_PE_FIRST_COLD_RATIO", ".2")),
        "later_cold_ratio": float(os.environ.get("VLLM_PE_LATER_COLD_RATIO", ".4")),
        "private_suffix_only": True,
        "python": sys.executable,
        "pid": os.getpid(),
        "version": vllm.__version__,
        "core_paths": paths,
        "core_sha256": hashes,
        "num_gpu_blocks": scheduler.kv_cache_config.num_blocks,
        "block_size": groups[0].kv_cache_spec.block_size,
        "hash_block_size": scheduler.hash_block_size,
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": config.model_config.model,
        "max_model_len": config.model_config.max_model_len,
        "dtype": str(config.model_config.dtype),
        "kv_cache_dtype": str(cache.cache_dtype),
        "gpu_memory_utilization": cache.gpu_memory_utilization,
        "prefix_caching": cache.enable_prefix_caching,
        "attention_backend": backend,
        "use_flashinfer_sampler": use_flashinfer_sampler,
        "enforce_eager": config.model_config.enforce_eager,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "max_concurrent_batches": config.max_concurrent_batches,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_num_scheduled_tokens": config.scheduler_config.max_num_scheduled_tokens,
        "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "seed": config.model_config.seed,
        "timestamp": time.monotonic(),
    }


def create_runtime(scheduler) -> ProgressiveRuntime | None:
    mode = os.environ.get("VLLM_PE_POLICY", "native")
    if mode not in ("native", "progressive"):
        raise ValueError("VLLM_PE_POLICY must be native or progressive")
    log_dir = os.environ.get("VLLM_PE_LOG_DIR")
    if not log_dir and mode == "native":
        return None
    if not log_dir:
        raise ValueError("VLLM_PE_LOG_DIR is required for progressive mode")
    return ProgressiveRuntime(scheduler, _configuration(scheduler, mode, log_dir))
