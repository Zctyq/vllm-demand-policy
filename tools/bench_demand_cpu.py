"""Reproducible CPU timings; no engine import, network, model or GPU use."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from native_loader import (
    REFERENCE,
    WORKTREE,
    load_cache_manager_core,
    load_runtime,
    load_source_module,
)
from scheduler_cpu import enqueue, scheduler
from test_demand import request
from test_native_core import chain_hashes

SRC = WORKTREE / "vllm/v1/core"
policy = load_source_module(SRC / "demand_policy.py", "vllm.v1.core.demand_policy")
queue = load_source_module(SRC / "demand_queue.py", "vllm.v1.core.demand_queue")
sys.modules["vllm.v1.core.progressive_runtime"] = load_runtime()
runtime = load_source_module(SRC / "demand_runtime.py", "vllm.v1.core.demand_runtime")
core = load_cache_manager_core()
native = load_cache_manager_core(REFERENCE)
sys.modules["vllm.v1.core.kv_cache_utils"] = core


def quantiles(values):
    values = sorted(values)
    return {
        "samples": len(values),
        "p50_us": statistics.median(values) / 1000,
        "p95_us": values[max(0, math.ceil(0.95 * len(values)) - 1)] / 1000,
        "p99_us": values[max(0, math.ceil(0.99 * len(values)) - 1)] / 1000,
    }


def measure(factory, repeats, warmup):
    wall, cpu, warm_wall, warm_cpu = [], [], [], []
    last = None
    for i in range(warmup + repeats):
        call, info = factory()
        c0, w0 = time.process_time_ns(), time.perf_counter_ns()
        call()
        w, c = time.perf_counter_ns() - w0, time.process_time_ns() - c0
        (warm_wall if i < warmup else wall).append(w)
        (warm_cpu if i < warmup else cpu).append(c)
        last = info()
    return {
        "steady_wall": quantiles(wall),
        "steady_cpu": quantiles(cpu),
        "warmup_wall": quantiles(warm_wall),
        "warmup_cpu": quantiles(warm_cpu),
        "raw_wall_ns": wall,
        "raw_cpu_ns": cpu,
        "last_counters": last,
    }


def build_pool(sessions, per_session, spare, enabled, telemetry=True):
    c = core if enabled else native
    p = c.BlockPool(sessions * per_session + spare + 1, True, 16)
    rt = None
    if enabled:
        rt = runtime.DemandRuntime(p, eviction=True, clock=lambda: 0)
        rt.telemetry = telemetry
        rt.stats.enabled = telemetry
        rt.policy.telemetry = telemetry
        p.demand_runtime = rt
    for i in range(sessions):
        blocks = p.get_new_blocks(per_session)
        r = request(c, str(i), per_session * 16, chain_hashes(per_session, str(i)))
        p.cache_full_blocks(r, blocks, 0, per_session, 16, 0)
        p.free_blocks(reversed(blocks))
    if rt:
        rt.policy.advance(1)
        rt.clock = lambda: 1
        rt.stats.clear()
        rt.policy.stats.clear()
    return p, rt


def allocation_factory(s, b, d, spare, enabled, telemetry):
    def factory():
        p, rt = build_pool(s, b, spare, enabled, telemetry)
        return lambda: p.get_new_blocks(d), lambda: rt.diagnostics() if rt else {}

    return factory


def selection_factory(s, b, d, reference=False, telemetry=True):
    _, rt = build_pool(s, b, 0, True, telemetry)
    m = rt.policy
    if reference:
        m.__class__ = policy.ReferencePolicy

    def factory():
        m.stats.clear()
        return lambda: m.plan(d, 1), lambda: dict(m.stats)

    return factory


def maintenance_factory(s, b, telemetry):
    _, rt = build_pool(s, b, 0, True, telemetry)
    m = rt.policy
    ticks = [1.0]

    def factory():
        m.stats.clear()
        ticks[0] += 0.001
        now = ticks[0]
        return lambda: m.reuse(b, now), lambda: dict(
            m.stats, heap_size=len(m.deadlines)
        )

    return factory


def cache_release_factory(s, b, enabled):
    def factory():
        pool, rt = build_pool(s, b, 16, enabled)
        blocks = pool.get_new_blocks(16)
        r = request(core, "new-cache", 256)
        if rt:
            rt.policy.stats.clear()

        def call():
            pool.cache_full_blocks(r, blocks, 0, 16, 16, 0)
            pool.free_blocks(reversed(blocks))

        return call, lambda: rt.diagnostics() if rt else {}

    return factory


def expiry_factory(s, b):
    def factory():
        _, rt = build_pool(s, b, 0, True)
        for bid in rt.policy.blocks:
            rt.policy.reuse(bid, 1)
        rt.policy.stats.clear()
        return lambda: rt.policy.advance(2), lambda: dict(rt.policy.stats)

    return factory


def scoring_factory(w, hit_blocks):
    p, rt = build_pool(1, hit_blocks, 1, True)
    # Actual native prefix query and HitQueue dependency registration.
    from native_loader import make_cache_manager

    rt.manager = make_cache_manager(core, p)

    def factory():
        q = queue.HitQueue(rt)
        for i in range(w):
            count = i % (hit_blocks + 1)
            hashes = chain_hashes(hit_blocks, "0")[:count]
            hashes += chain_hashes(hit_blocks - count, "miss" + str(i))
            q.add_request(request(core, str(i), hit_blocks * 16 + 1, hashes))
        return q.refresh, lambda: dict(q.stats)

    return factory


def heap_cleanup_factory(s, b):
    def factory():
        _, rt = build_pool(s, b, 0, True)
        m = rt.policy
        for bid in m.blocks:
            m.reuse(bid, 1)
        while len(m.deadlines) < 2 * len(m.blocks) + 64:
            m.reuse(1, 1)
        m.stats.clear()
        return lambda: m.reuse(1, 1), lambda: dict(m.stats, heap_size=len(m.deadlines))

    return factory


def scheduler_factory(w, enabled, eviction):
    def factory():
        c = core if (enabled or eviction) else native
        p = c.BlockPool(1025, True, 16)
        s = scheduler(
            c,
            p,
            source=WORKTREE if (enabled or eviction) else REFERENCE,
            budget=128,
            max_running=16,
        )
        rt = None
        if enabled or eviction:
            rt = runtime.DemandRuntime(
                p, eviction=eviction, waiting=enabled, clock=lambda: 0
            )
            rt.manager = s.kv_cache_manager
            s.demand_runtime = s.kv_cache_manager.demand_runtime = p.demand_runtime = rt
        if enabled:
            s.waiting, s.skipped_waiting = queue.HitQueue(rt), queue.HitQueue(rt)
            rt.queues = [s.waiting, s.skipped_waiting]
        cached = request(c, "cache", 256)
        s.kv_cache_manager.allocate_slots(cached, 256)
        s.kv_cache_manager.free(cached)
        for i in range(w):
            count = i % 17
            hashes = cached.block_hashes[:count] + chain_hashes(16 - count, str(i))
            enqueue(s, request(c, str(i), 257, hashes))
        if rt:
            rt.stats.clear()
        return s.schedule, lambda: rt.diagnostics() if rt else {}

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/demand/cpu-bench.json"
    )
    args = parser.parse_args()
    results = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "pid": os.getpid(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "units": "microseconds",
        "scope": "real extracted native CPU methods; config and worker output stubs; no EngineCore/GPU",
        "cases": [],
    }

    def run(component, params, factory):
        entry = {
            "component": component,
            **params,
            **measure(factory, args.repeats, args.warmup),
        }
        results["cases"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(component, params, round(entry["steady_wall"]["p50_us"], 2), flush=True)

    results["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            SRC / "demand_policy.py",
            SRC / "demand_runtime.py",
            SRC / "demand_queue.py",
            SRC / "block_pool.py",
            SRC / "kv_cache_manager.py",
            SRC / "sched/scheduler.py",
            Path(__file__).resolve(),
            ROOT / "tests/scheduler_cpu.py",
            ROOT / "tests/native_loader.py",
        ]
    }
    for s, b in [(8, 64), (32, 64), (128, 64)]:
        for d, spare, pressure in [(1, 8, "none"), (1, 0, "light"), (32, 0, "heavy")]:
            for enabled in (False, True):
                run(
                    "allocation",
                    {
                        "S": s,
                        "B": s * b,
                        "N": d,
                        "U": spare,
                        "D": max(0, d - spare),
                        "mode": "demand" if enabled else "native",
                        "pressure": pressure,
                    },
                    allocation_factory(s, b, d, spare, enabled, True),
                )
        for ref in (False, True):
            run(
                "selection",
                {
                    "S": s,
                    "B": s * b,
                    "D": 32,
                    "mode": "reference" if ref else "indexed",
                },
                selection_factory(s, b, 32, reference=ref),
            )
        for telemetry in (False, True):
            run(
                "maintenance_reuse",
                {"S": s, "B": s * b, "telemetry": telemetry},
                maintenance_factory(s, b, telemetry),
            )
    for telemetry in (False, True):
        run(
            "allocation_telemetry",
            {"S": 32, "B": 2048, "D": 32, "telemetry": telemetry},
            allocation_factory(32, 64, 32, 0, True, telemetry),
        )
    for per_session in (64, 256, 1024):
        run(
            "selection_fixed_S",
            {"S": 8, "B": 8 * per_session, "D": 32},
            selection_factory(8, per_session, 32),
        )
        run(
            "expiry_burst",
            {"S": 8, "B": 8 * per_session, "E": 8 * per_session},
            expiry_factory(8, per_session),
        )
    run(
        "large_D",
        {"S": 128, "B": 8192, "D": 256},
        allocation_factory(128, 64, 256, 0, True, True),
    )
    results["metadata_bytes"] = []
    for s in (8, 128):
        for enabled in (False, True):
            run(
                "cache_release_maintenance",
                {
                    "S": s,
                    "B": s * 64,
                    "blocks": 16,
                    "mode": "demand" if enabled else "native",
                },
                cache_release_factory(s, 64, enabled),
            )
    for s in (8, 128):
        run("expiry_heap_cleanup", {"S": s, "B": s * 64}, heap_cleanup_factory(s, 64))
    for s in (8, 32, 128):
        _, rt = build_pool(s, 64, 0, True)
        results["metadata_bytes"].append(
            {"S": s, "B": s * 64, "reachable_python_bytes": rt.metadata_bytes()}
        )
    for w in (8, 32, 128):
        run("score_sort", {"W": w, "prefix_blocks": 16}, scoring_factory(w, 16))
        for enabled, eviction in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            run(
                "schedule",
                {"W": w, "waiting": enabled, "eviction": eviction, "B": 1024},
                scheduler_factory(w, enabled, eviction),
            )


if __name__ == "__main__":
    main()
