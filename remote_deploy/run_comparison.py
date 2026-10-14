"""Causal fixed-manifest GPU replay, paired cold starts and raw measurements."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

import aiohttp
import replay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = Path("/data/zhaohairui/vllm-progressive-eviction")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def percentile(values, p):
    import math

    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def distribution(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


async def http(session, base, route, payload=None):
    async with session.request(
        "GET" if payload is None else "POST", base + route, json=payload
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"{route}: HTTP {response.status}")
        return await response.text()


async def send(session, base, model, call, marker, extra):
    begin, first = time.monotonic(), None
    output_ids, prompt_ids, usage, rid, finish = [], None, None, None, None
    payload = {
        "model": model,
        "prompt": call["prompt_token_ids"],
        "temperature": 0,
        "seed": 17,
        "max_tokens": 128,
        "min_tokens": 128,
        "ignore_eos": True,
        "stop": [],
        "stop_token_ids": [],
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "request_id": marker,
    }
    async with session.post(
        base + "/v1/completions", json=payload, headers={"X-Request-Id": marker}
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"completion HTTP {response.status}")
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError("server stream error")
            rid = chunk.get("id", rid)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("prompt_token_ids") is not None:
                    prompt_ids = choice["prompt_token_ids"]
                ids = choice.get("token_ids") or []
                if ids and first is None:
                    first = time.monotonic()
                output_ids.extend(ids)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    end = time.monotonic()
    if first is None or len(output_ids) != 128:
        raise RuntimeError("missing first output token or incorrect output count")
    extra[marker] = {
        "ttft_s": first - begin,
        "latency_s": end - begin,
        "tpot_s": (end - first) / 127,
        "output_sha256": replay.hash_ids(output_ids),
    }
    return (
        {
            "id": rid,
            "prompt_token_ids": prompt_ids,
            "usage": usage,
            "choices": [{"finish_reason": finish}],
        },
        begin,
        end,
    )


async def monitor(session, base, out, stop):
    with (out / "metrics.jsonl").open("w") as log:
        while not stop.is_set():
            try:
                text = await http(session, base, "/metrics")
                log.write(
                    json.dumps({"monotonic": time.monotonic(), "text": text}) + "\n"
                )
                log.flush()
            except Exception as exc:
                log.write(json.dumps({"error": str(exc)}) + "\n")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass


async def measure(proc, base, config, manifest, out, run_id, concurrency):
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}; see server.log")
            try:
                await http(session, base, "/health")
                break
            except Exception:
                await asyncio.sleep(1)
        else:
            raise RuntimeError("server readiness timeout")
        evidence = json.loads((out / "engine/engine.json").read_text())
        assert evidence["num_blocks"] == config["num_gpu_blocks"], evidence
        assert evidence["hash_block_size"] == config["block_size"], evidence
        assert evidence["groups"] == ["FullAttentionSpec"], evidence
        assert evidence["async_scheduling"] is False, evidence
        if out.name.startswith("native"):
            assert evidence["runtime"] is False
            expected = json.loads((HERE / "source-versions.json").read_text())["native"]
            for rel, digest in evidence["hashes"].items():
                assert expected["vllm/" + rel] == digest
        else:
            assert evidence["waiting"] and evidence["eviction"], evidence
            for rel, digest in evidence["hashes"].items():
                assert (
                    digest
                    == hashlib.sha256(
                        (HERE / "worktree/vllm" / rel).read_bytes()
                    ).hexdigest()
                )
        # Identical warmup, then explicit successful cold-cache reset for each run.
        extra = {}
        call = manifest["sessions"][0]["calls"][0]
        for index in range(2):
            marker = replay.request_marker("warmup", "warmup", index)
            response, _, _ = await send(
                session, base, config["served_model_name"], call, marker, extra
            )
            replay.validate_response(response, call, marker)
        warmup_rows = [
            json.loads(line)
            for line in (out / "engine/requests.jsonl").read_text().splitlines()
        ]
        assert warmup_rows[-1]["H"] > 0, "warmup cache hit missing"
        write(out / "warmup_outputs.json", extra)
        reset = json.loads(await http(session, base, "/reset_prefix_cache", {}))
        assert reset.get("success") is True, reset
        write(out / "cache_reset.json", reset)
        await asyncio.sleep(2)
        before = await http(session, base, "/metrics")
        replay.assert_idle(before)
        (out / "metrics_before.txt").write_text(before)
        stop = asyncio.Event()
        sampling = asyncio.create_task(monitor(session, base, out, stop))
        extra.clear()

        async def request(call, marker):
            return await send(
                session, base, config["served_model_name"], call, marker, extra
            )

        def emit(kind, row):
            with (out / (kind + ".jsonl")).open("a") as log:
                log.write(json.dumps(row) + "\n")
            if kind == "request":
                print(
                    run_id,
                    row["session_id"],
                    row["call_index"],
                    row["success"],
                    flush=True,
                )

        try:
            result = await replay.replay_sessions(
                manifest["sessions"],
                concurrency=concurrency,
                seed=17,
                run_id=run_id,
                send=request,
                emit=emit,
            )
        finally:
            stop.set()
            await sampling
        for row in result["requests"]:
            row.update(extra.get(row["request_id"], {}))
        await asyncio.sleep(2)
        after = await http(session, base, "/metrics")
        (out / "metrics_after.txt").write_text(after)
        write(out / "result.json", result)
        summary = replay.client_summary(result)
        summary.update(
            ttft_s=distribution(
                [r["ttft_s"] for r in result["requests"] if r["success"]]
            ),
            request_latency_s=distribution(
                [r["latency_s"] for r in result["requests"] if r["success"]]
            ),
            tpot_s=distribution(
                [r["tpot_s"] for r in result["requests"] if r["success"]]
            ),
        )
        write(out / "summary.json", summary)
        return summary


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)


def run(variant, repeat, args, config, manifest, deployment):
    out = ROOT / f"{variant}-r{repeat}"
    out.mkdir(exist_ok=False)
    run_id = out.name
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VLLM_PE_"):
            env.pop(key)
    overlay = deployment["native_overlay" if variant == "native" else "demand_overlay"]
    env.update(
        PYTHONPATH=f"{overlay}:{HERE}",
        CUDA_VISIBLE_DEVICES=str(config["gpu"]),
        VLLM_WAITING_HIT_ASC="0" if variant == "native" else "1",
        VLLM_DEMAND_EVICTION="0" if variant == "native" else "1",
        VLLM_DEMAND_TELEMETRY="1",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        DEMAND_EXPERIMENT_LOG=str(out / "engine"),
    )
    spec = importlib.util.spec_from_file_location(
        "old_manage", PROJECT / "tools/manage.py"
    )
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    config = dict(config, overlay_site=overlay, port=args.port)
    cmd = manager.server_command(config)
    cmd += ["--scheduler-cls", "demand_observer.ObservedScheduler"]
    write(
        out / "launch.json",
        {
            "command": cmd,
            "flags": {
                k: env[k]
                for k in [
                    "PYTHONPATH",
                    "CUDA_VISIBLE_DEVICES",
                    "VLLM_WAITING_HIT_ASC",
                    "VLLM_DEMAND_EVICTION",
                    "VLLM_DEMAND_TELEMETRY",
                ]
            },
            "config": config,
            "manifest_sha256": replay.file_sha256(args.manifest),
            "concurrency": args.concurrency,
        },
    )
    with (out / "server.log").open("wb") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    write(out / "process.json", {"pid": proc.pid, "command": cmd})
    try:
        summary = asyncio.run(
            measure(
                proc,
                f"http://127.0.0.1:{args.port}",
                config,
                manifest,
                out,
                run_id,
                args.concurrency,
            )
        )
        print("COMPLETE", run_id, json.dumps(summary), flush=True)
        if (
            not summary["valid_complete_run"]
            or summary["requests_successful"] != manifest["selected_calls"]
        ):
            raise RuntimeError("incomplete workload; comparison invalid")
    finally:
        stop_process(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(PROJECT / "results/stratified10-seed17-i28ico0v/manifest_10.json"),
    )
    parser.add_argument(
        "--profile", default=str(PROJECT / "configs/server.gpu1.calibration.json")
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    config = json.loads(Path(args.profile).read_text())
    manifest = replay.validate_manifest(
        args.manifest, expected_sessions=10, concurrency=args.concurrency
    )
    deployment = json.loads((ROOT / "deployment.json").read_text())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(config["gpu"]),
            "--query-compute-apps=process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if any(
        Path(name.strip()).name != "nvidia-cuda-mps-server"
        for name in processes.splitlines()
    ):
        raise RuntimeError("selected GPU has another compute application")
    idle = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(config["gpu"]),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization = [int(x.strip()) for x in idle.split(",")]
    if memory > 128 or utilization != 0:
        raise RuntimeError("selected GPU is not idle")
    write(
        ROOT / "gpu-preflight.json",
        {
            "processes": processes,
            "memory_mib": memory,
            "utilization_percent": utilization,
            "mps_clients_verified": False,
            "existing_mps_left_running": True,
        },
    )
    write(
        ROOT / "experiment.json",
        {
            "manifest_sha256": replay.file_sha256(args.manifest),
            "selected_sessions": 10,
            "selected_calls": manifest["selected_calls"],
            "seed": 17,
            "repeats": args.repeats,
            "concurrency": args.concurrency,
            "output_tokens": 128,
            "preserve_pre_gap": True,
            "transport": "pretokenized completions SSE",
        },
    )
    for repeat in range(1, args.repeats + 1):
        for variant in ["native", "demand"] if repeat % 2 else ["demand", "native"]:
            run(variant, repeat, args, config, manifest, deployment)
    print("ALL_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
