#!/usr/bin/env python3
"""Manual GPU mechanism check; --self-test never loads a model or contacts HTTP."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path


def load_replay():
    source = Path(__file__).resolve().parents[1] / "experiment" / "replay.py"
    spec = importlib.util.spec_from_file_location("pe_replay_helpers", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_plan(block_size: int, stage: int, mode: str) -> dict:
    """A request after threshold 20/30 triggers the pending safe-boundary check."""
    if block_size < 2 or stage not in (20, 30) or mode not in ("native", "progressive"):
        raise ValueError("unsupported fixture parameters")
    first = 100 - math.ceil(0.2 * 100)
    second = first - math.ceil(0.4 * first)
    remaining = 100 if mode == "native" else first if stage == 20 else second
    return {
        "prompt_tokens": 100 * block_size + 1,
        "target_full_blocks": 100,
        "filler_requests": stage,
        "completed_before_probe": stage + 1,
        "expected_checkpoints": stage // 10,
        "expected_remaining_blocks": remaining,
        "expected_probe_hit_tokens": remaining * block_size,
        "expected_history_tokens": 100 * block_size,
    }


def self_test() -> dict:
    replay = load_replay()
    for size in (2, 16, 32, 64):
        for mode in ("native", "progressive"):
            for stage in (20, 30):
                plan = fixture_plan(size, stage, mode)
                assert (plan["prompt_tokens"] - 1) // size == 100
                assert plan["filler_requests"] + 1 == plan["completed_before_probe"]
                assert (plan["completed_before_probe"] - 1) // 10 == stage // 10
                expected = 100 if mode == "native" else 80 if stage == 20 else 48
                assert plan["expected_probe_hit_tokens"] == expected * size
    marker = replay.request_marker("smoke", "fixture", 1)
    assert replay.metadata("cmpl-" + marker + "-deadbeef") == {
        "r": "smoke",
        "s": "fixture",
        "c": 1,
    }
    return {
        "status": "passed",
        "scope": "CPU fixture arithmetic and request-marker self-test only",
        "gpu_executed": False,
        "http_requests_sent": 0,
        "fixtures_checked": 16,
        "progressive_remaining_blocks": [80, 48],
    }


def check_config(config: dict) -> None:
    if config.get("policy", config.get("mode")) not in ("native", "progressive"):
        raise ValueError("missing instrumented native/progressive run_config")
    if not config.get("run_id") or not config.get("core_sha256"):
        raise ValueError(
            "run ID and actual EngineCore source fingerprints are required"
        )
    if config.get("check_interval") != 10:
        raise ValueError("fixture requires check_interval=10")
    if config.get("first_cold_ratio") != 0.2 or config.get("later_cold_ratio") != 0.4:
        raise ValueError("fixture requires first/later ratios 0.20/0.40")
    if config.get("num_gpu_blocks", 0) < 104:
        raise ValueError("fixture requires at least 104 total KV blocks")
    block_size = config["block_size"]
    if block_size < 2 or config["max_model_len"] < 100 * block_size + 2:
        raise ValueError(
            "model length or block size cannot accommodate the 100-block fixture"
        )
    if not config.get("prefix_caching"):
        raise ValueError("prefix caching must be enabled")
    if config.get("max_concurrent_batches") != 1 or config.get("async_scheduling"):
        raise ValueError("fixture requires the supported synchronous runtime")
    if config.get("use_flashinfer_sampler") is not False:
        raise ValueError("EngineCore must record use_flashinfer_sampler=false")


def sampler_log_evidence(path: Path) -> dict:
    """Require the native CUDA sampler initialization message from this launch."""
    if not path.is_file():
        raise ValueError(f"Missing worker initialization log: {path}")
    snapshot = path.read_bytes()
    lines = snapshot.decode("utf-8", errors="replace").splitlines()
    enabled = "Using FlashInfer for top-p & top-k sampling."
    disabled = (
        "FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0."
    )
    if any(enabled in line for line in lines):
        raise ValueError("Server log reports FlashInfer sampler initialization")
    matches = [line for line in lines if disabled in line]
    if not matches:
        raise ValueError(
            "Worker sampler-disabled initialization message not found; "
            "preserve INFO worker logs from this launch. "
            "An EngineCore environment record alone is insufficient."
        )
    return {
        "server_log": str(path.resolve()),
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "snapshot_bytes": len(snapshot),
        "disabled_initialization_lines": matches,
        "scope": "CUDA sampler initialization log; no stochastic sampling claim",
    }


def read_logs(replay, directory: Path, name: str, offset: int):
    return replay.jsonl(directory / name)[offset:]


def reset(replay, args, config) -> dict:
    directory = Path(args.engine_log_dir)
    prior = len(replay.jsonl(directory / "reset.jsonl"))
    result = json.loads(
        replay.http(
            args.base_url,
            "/reset_prefix_cache?reset_running_requests=false&reset_external=false",
            method="POST",
            timeout=args.timeout,
        )
    )
    if result != {"success": True}:
        raise RuntimeError(
            "safe reset failed; use an idle dedicated experiment instance"
        )
    evidence = read_logs(replay, directory, "reset.jsonl", prior)
    if (
        len(evidence) != 1
        or not evidence[0].get("success")
        or evidence[0].get("completed") != 0
        or evidence[0].get("run_id") != config["run_id"]
    ):
        raise RuntimeError("HTTP reset lacks matching EngineCore policy-reset evidence")
    if evidence[0].get("free_blocks") != config["num_gpu_blocks"] - 1:
        raise RuntimeError("reset did not leave exactly the non-null pool blocks free")
    return evidence[0]


def execute_fixture(replay, args, config, vocab: list[int], stage: int) -> dict:
    reset_event = reset(replay, args, config)
    directory = Path(args.engine_log_dir)
    request_offset = len(replay.jsonl(directory / "requests.jsonl"))
    check_offset = len(replay.jsonl(directory / "policy_checks.jsonl"))
    mode = config.get("policy", config["mode"])
    size = config["block_size"]
    plan = fixture_plan(size, stage, mode)
    rng = random.Random(args.seed + stage)
    prompt = [rng.choice(vocab) for _ in range(plan["prompt_tokens"])]
    session = f"__pe_smoke_{stage}__"
    markers = []
    response_records = []

    def send(tokens, owner, call_index):
        marker = replay.request_marker(config["run_id"], owner, call_index)
        response = json.loads(
            replay.http(
                args.base_url,
                "/v1/completions",
                {
                    "model": args.model or config["model"],
                    "prompt": tokens,
                    "temperature": 0,
                    "seed": args.seed,
                    "max_tokens": 1,
                    "min_tokens": 1,
                    "ignore_eos": True,
                    "stop": [],
                    "stop_token_ids": [],
                    "n": 1,
                    "stream": False,
                    "return_token_ids": True,
                    "add_special_tokens": False,
                },
                marker=marker,
                timeout=args.timeout,
            )
        )
        if replay.metadata(response.get("id")) != replay.metadata(marker):
            raise RuntimeError("completion request ID did not preserve replay metadata")
        usage = response.get("usage") or {}
        if (
            usage.get("prompt_tokens") != len(tokens)
            or usage.get("completion_tokens") != 1
        ):
            raise RuntimeError(
                "fixture actual token lengths differ from requested lengths"
            )
        choices = response.get("choices") or []
        if len(choices) != 1 or choices[0].get("finish_reason") != "length":
            raise RuntimeError(
                "fixture generation did not complete normally at one token"
            )
        if choices[0].get("prompt_token_ids") != tokens:
            raise RuntimeError("server changed the explicit token-ID prompt")
        markers.append(marker)
        response_records.append(
            {
                "request_id": marker,
                "response_id": response["id"],
                "input_tokens": len(tokens),
                "output_tokens": 1,
            }
        )
        return marker

    initial = send(prompt, session, 0)
    # A one-token filler never commits a full block; every owner is distinct.
    for index in range(plan["filler_requests"]):
        send([rng.choice(vocab)], f"__pe_smoke_filler_{stage}_{index}__", 0)
    checks_before_probe = read_logs(
        replay, directory, "policy_checks.jsonl", check_offset
    )
    if len(checks_before_probe) != plan["expected_checkpoints"]:
        raise RuntimeError(
            "wrong number of safe-boundary checks; foreign traffic or missed hooks"
        )
    expected_completed = list(range(10, stage + 1, 10))
    if [row["completed_requests"] for row in checks_before_probe] != expected_completed:
        raise RuntimeError("checkpoints did not occur at the expected completed counts")
    if any(row["free_before"] != row["free_after"] for row in checks_before_probe):
        raise RuntimeError("pure eviction changed the free block count")
    expected_evictions = (
        [0] * (stage // 10)
        if mode == "native"
        else [0, 20]
        if stage == 20
        else [0, 20, 32]
    )
    if [
        row["evicted_logical_blocks"] for row in checks_before_probe
    ] != expected_evictions:
        raise RuntimeError(
            "actual hard evictions differ from birth/20%/40% expectations"
        )
    if [
        row["evicted_physical_blocks"] for row in checks_before_probe
    ] != expected_evictions:
        raise RuntimeError(
            "fixture unexpectedly contains duplicate copies or failed eviction"
        )
    summary_before_probe = json.loads((directory / "eviction_summary.json").read_text())
    if summary_before_probe.get("native_physical") != 0:
        raise RuntimeError(
            "native pressure contaminated the no-pressure mechanism fixture"
        )

    probe = send(prompt, session, 1)
    records = read_logs(replay, directory, "requests.jsonl", request_offset)
    if len(records) != len(markers) or {row["request_id"] for row in records} != set(
        markers
    ):
        raise RuntimeError(
            "engine request logs are missing records or contain foreign traffic"
        )
    by_id = {row["request_id"]: row for row in records}
    for row in records:
        if (
            not row.get("success")
            or not row.get("executed")
            or row["output_tokens"] != 1
        ):
            raise RuntimeError(
                "fixture contains a failed, unexecuted, or wrong-length request"
            )
    if by_id[initial]["history_reusable_tokens"] != 0:
        raise RuntimeError(
            "first content generation was incorrectly counted as historical reuse"
        )
    observed = by_id[probe]
    if observed["history_reusable_tokens"] != plan["expected_history_tokens"]:
        raise RuntimeError(
            "ghost history did not preserve all 100 previously computed blocks"
        )
    if observed["adopted_hit_tokens"] != plan["expected_probe_hit_tokens"]:
        raise RuntimeError(
            "actual adopted probe hit does not match the remaining prefix"
        )
    if observed["actual_prefix_hit_tokens"] != plan["expected_probe_hit_tokens"]:
        raise RuntimeError(
            "actual physical prefix hit differs from expected fixture residency"
        )
    return {
        "stage": stage,
        "plan": plan,
        "reset_evidence": reset_event,
        "checks_before_probe": checks_before_probe,
        "eviction_summary_before_probe": summary_before_probe,
        "first_request": by_id[initial],
        "probe_request": observed,
        "responses": response_records,
        "gpu_executed": True,
        "status": "passed",
    }


def run(args) -> dict:
    if not all((args.engine_log_dir, args.tokenizer, args.output)):
        raise ValueError(
            "--engine-log-dir, --tokenizer and --output are required for --run"
        )
    replay = load_replay()
    output = Path(args.output)
    if output.exists():
        raise ValueError("smoke output already exists; choose a new filename")
    config = json.loads((Path(args.engine_log_dir) / "run_config.json").read_text())
    check_config(config)
    server_log = (
        Path(args.server_log)
        if args.server_log
        else Path(args.engine_log_dir).parent / "server.log"
    )
    sampler_evidence = sampler_log_evidence(server_log)
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_dir():
        raise ValueError(
            "--tokenizer must be a local directory; downloads are not allowed"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    special = set(tokenizer.all_special_ids)
    vocab = sorted(set(tokenizer.get_vocab().values()) - special)
    if len(vocab) < 100:
        raise ValueError("local tokenizer has too few ordinary valid IDs")
    result = {
        "schema": "progressive-gpu-smoke-1",
        "run_id": config["run_id"],
        "mode": config.get("policy", config.get("mode")),
        "engine_config": config,
        "sampler_initialization_evidence": sampler_evidence,
        "started_monotonic": time.monotonic(),
        "fixtures": [],
        "scope": "real HTTP -> EngineCore -> BlockPool -> GPU prefix hit mechanism",
        "performance_benchmark": False,
        "status": "running",
    }
    replay.dump(output, result)
    try:
        for stage in (20, 30):
            result["fixtures"].append(
                execute_fixture(replay, args, config, vocab, stage)
            )
            replay.dump(output, result)
        result["status"] = "passed"
        result["gpu_executed"] = True
        # Leave a clean idle cache and a zeroed policy state for any next run.
        result["final_reset"] = reset(replay, args, config)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        replay.dump(output, result)
        raise
    result["finished_monotonic"] = time.monotonic()
    replay.dump(output, result)
    return {
        "status": "passed",
        "gpu_executed": True,
        "output": str(output),
        "mode": result["mode"],
        "fixtures": len(result["fixtures"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--run", action="store_true", help="send real HTTP requests")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--engine-log-dir")
    parser.add_argument(
        "--server-log", help="this launch's worker log; defaults to ../server.log"
    )
    parser.add_argument("--tokenizer", help="local deployed model/tokenizer directory")
    parser.add_argument(
        "--model", help="served model name, if different from model path"
    )
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    result = self_test() if args.self_test else run(args)
    if args.self_test and args.output:
        load_replay().dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
