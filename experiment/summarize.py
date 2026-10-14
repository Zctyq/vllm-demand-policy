#!/usr/bin/env python3
"""Produce raw and paired results; fail closed on unequal/failed runs.

No inference is drawn from synthetic tests. Only runs with exact EngineCore H/K
joins and matching workloads/capacity are eligible for paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from replay import dump

PAIR_FIELDS = (
    "model",
    "manifest_sha256",
    "num_gpu_blocks",
    "block_size",
    "max_model_len",
    "output_tokens",
    "sessions",
    "concurrency",
)
METRICS = (
    "historical_prefix_loss_rate",
    "task_throughput_per_second",
    "mean_task_completion_seconds",
    "output_token_throughput_per_second",
)


def paired_results(runs):
    grouped = {}
    for run in runs:
        seed, policy = run["config"]["seed"], run["config"]["policy"]
        if policy not in ("native", "progressive"):
            raise ValueError("unknown policy")
        if policy in grouped.setdefault(seed, {}):
            raise ValueError(f"duplicate {policy} run for seed {seed}")
        grouped[seed][policy] = run
    pairs = []
    for seed, modes in sorted(grouped.items()):
        if set(modes) != {"native", "progressive"}:
            pairs.append(
                {"seed": seed, "valid_pair": False, "reason": "missing one policy"}
            )
            continue
        a, b = modes["native"], modes["progressive"]
        reasons = []
        for run in (a, b):
            if run["config"].get("status") != "completed":
                reasons.append("run did not complete all runtime validation gates")
            if run["config"].get("sessions") != 50:
                reasons.append(
                    "formal paired experiment requires exactly 50 complete sessions"
                )
            if not run["summary"].get("valid_complete_run"):
                reasons.append("incomplete or failed workload")
            if not run["summary"].get("engine_request_metadata_verified"):
                reasons.append("EngineCore H/K metadata not verified")
            if run["config"].get("cache_start") != "warmup_then_confirmed_reset":
                reasons.append("empty cache start not confirmed")
            if not run["summary"].get("policy_free_count_invariant_verified"):
                reasons.append("engine eviction/checkpoint counters not verified")
        for field in PAIR_FIELDS:
            if field not in a["config"] or a["config"].get(field) != b["config"].get(
                field
            ):
                reasons.append(f"mismatched/missing {field}")
        if a.get("task_order") != b.get("task_order") or not a.get("task_order"):
            reasons.append("different/missing session order")
        if a["summary"].get("output_tokens") != b["summary"].get("output_tokens"):
            reasons.append("different actual output workload")
        # Engine startup proof is kept as a stable comparison signature by the
        # version-specific launcher. CLI flags alone are insufficient evidence.
        if not a["config"].get("engine_comparison_signature") or a["config"].get(
            "engine_comparison_signature"
        ) != b["config"].get("engine_comparison_signature"):
            reasons.append("different engine build/model/scheduler configuration")
        pair = {
            "seed": seed,
            "valid_pair": not reasons,
            "native_directory": a.get("directory"),
            "progressive_directory": b.get("directory"),
        }
        pair["cache_pressure_observed_in_pair"] = bool(
            a["summary"].get("cache_pressure_observed")
            or b["summary"].get("cache_pressure_observed")
        )
        pair["progressive_eviction_exercised"] = bool(
            b["summary"].get("proactive_eviction_exercised")
        )
        pair["competition_and_eviction_tested"] = (
            pair["cache_pressure_observed_in_pair"]
            and pair["progressive_eviction_exercised"]
        )
        notes = []
        if not pair["cache_pressure_observed_in_pair"]:
            notes.append(
                "Neither run observed native replacement of valid cached blocks "
                "or request preemption; cache competition was not demonstrated."
            )
        if not pair["progressive_eviction_exercised"]:
            notes.append(
                "The progressive run performed zero proactive hard evictions; "
                "it cannot demonstrate the effect of proactive eviction."
            )
        pair["qualification_notes"] = notes
        if reasons:
            pair["reasons"] = sorted(set(reasons))
        else:
            for name in METRICS:
                va, vb = a["summary"].get(name), b["summary"].get(name)
                pair[name] = {
                    "native": va,
                    "progressive": vb,
                    "progressive_minus_native": vb - va
                    if va is not None and vb is not None
                    else None,
                    "relative_change_percent": (vb / va - 1) * 100
                    if va not in (None, 0) and vb is not None
                    else None,
                }
        pairs.append(pair)
    return pairs


def load_run(path):
    path = Path(path)
    config = json.loads((path / "run_config.json").read_text())
    summary_path = path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    elif (path / "client_summary.json").exists():
        summary = json.loads((path / "client_summary.json").read_text())
        summary["valid_complete_run"] = False
    else:
        summary = {"valid_complete_run": False, "error": "missing run summary"}
    order = (
        json.loads((path / "task_order.json").read_text())
        if (path / "task_order.json").exists()
        else None
    )
    return {
        "directory": str(path.resolve()),
        "config": config,
        "summary": summary,
        "task_order": order,
    }


def plan(seeds):
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct order seeds required")
    return [
        {"run_index": 2 * i + j + 1, "seed": seed, "policy": policy}
        for i, seed in enumerate(seeds)
        for j, policy in enumerate(
            ("native", "progressive") if i % 2 == 0 else ("progressive", "native")
        )
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser(
        "plan", help="print six-run paired schedule; does not start servers"
    )
    a.add_argument("--seeds", nargs=3, type=int, default=[17, 29, 43])
    a.add_argument("--out", required=True)
    b = sub.add_parser("summarize", help="join six existing client run directories")
    b.add_argument("runs", nargs="+")
    b.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "plan":
        dump(args.out, plan(args.seeds))
        print(args.out)
        return
    runs = [load_run(path) for path in args.runs]
    pairs = paired_results(runs)
    valid = [p for p in pairs if p["valid_pair"]]
    deltas = {
        metric: [
            p[metric]["progressive_minus_native"]
            for p in valid
            if p[metric]["progressive_minus_native"] is not None
        ]
        for metric in METRICS
    }
    result = {
        "runs": runs,
        "pairs": pairs,
        "valid_pairs": len(valid),
        "complete_three_pair_experiment": len(pairs) == 3 and len(valid) == 3,
        "valid_pairs_exercising_eviction_with_pressure_evidence": sum(
            p["competition_and_eviction_tested"] for p in valid
        ),
        "mean_paired_absolute_changes": {
            k: statistics.mean(v) if v else None for k, v in deltas.items()
        },
        "interpretation": (
            "positive loss/completion-time changes are worse; "
            "positive throughput changes are better; "
            "no claimed speedup when a pair is invalid"
        ),
    }
    dest = Path(args.out)
    dump(dest, result)
    with dest.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "directory",
            "seed",
            "policy",
            "valid_complete_run",
            "sessions_completed",
            "sessions_failed",
            "configured_active_session_limit",
            "effective_worker_count",
            "observed_max_inflight_http_requests",
            "output_tokens",
            *METRICS,
            "active_evicted_logical_blocks",
            "active_evicted_physical_blocks",
            "native_evicted_observed_logical_blocks",
            "native_evicted_physical_blocks",
            "policy_check_count",
            "policy_check_total_ms",
            "policy_check_mean_ms",
            "proactive_eviction_exercised",
            "cache_pressure_observed",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run["summary"].get(key) for key in fields}
            row.update(
                directory=run["directory"],
                seed=run["config"]["seed"],
                policy=run["config"]["policy"],
            )
            writer.writerow(row)
    print(
        json.dumps(
            {"report": str(dest), "valid_pairs": len(valid), "expected_pairs": 3}
        )
    )
    if not result["complete_three_pair_experiment"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
