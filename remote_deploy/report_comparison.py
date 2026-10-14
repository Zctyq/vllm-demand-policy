"""Join real client/engine records and report valid fixed-workload GPU runs."""

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from demand_metadata import parse_request_id

ROOT = Path(__file__).resolve().parent.parent


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def prom(text, name):
    return sum(
        float(value)
        for value in re.findall(
            r"(?m)^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([^\s]+)", text
        )
    )


def analyze(run):
    summary = json.loads((run / "summary.json").read_text())
    if not summary["valid_complete_run"]:
        raise ValueError(f"invalid/incomplete run {run.name}")
    result = json.loads((run / "result.json").read_text())
    requests = result["requests"]
    assert len(requests) == 237 and all(r["success"] for r in requests)
    engine = [
        json.loads(line)
        for line in (run / "engine/requests.jsonl").read_text().splitlines()
    ]
    engine = [r for r in engine if r["run_id"] == run.name]
    indexed = {(r["session_id"], r["call_index"]): r for r in engine}
    assert len(engine) == len(indexed) == len(requests)
    for request in requests:
        identity = parse_request_id(request["request_id"])
        row = indexed[(identity["session_id"], identity["call_index"])]
        assert row["prompt_tokens"] == request["input_tokens"]
        assert row["output_tokens"] == request["output_tokens"] == 128
        assert 0 <= row["H"] <= request["input_tokens"]
    cumulative = run / "engine/summary.json"
    if not cumulative.exists():
        cumulative = run / "engine/latest.json"
    counters = json.loads(cumulative.read_text())
    h = sum(r["H"] for r in engine)
    k = sum(r["K"] for r in engine)
    tokens = sum(r["input_tokens"] for r in requests)
    preemptions = sum(r["preemptions"] for r in engine)
    expected_scheduled = tokens - h + 237 * 127
    before = (run / "metrics_before.txt").read_text()
    after = (run / "metrics_after.txt").read_text()
    policy = counters.get("demand") or {}
    jct = [r["completion_seconds"] - r["arrival_seconds"] for r in result["tasks"]]
    launch = json.loads((run / "launch.json").read_text())
    row = {
        "run": run.name,
        "requests": len(requests),
        "sessions": len(jct),
        "input_tokens": tokens,
        "elapsed_s": summary["elapsed_seconds"],
        "mean_jct_s": statistics.mean(jct),
        "p95_jct_s": percentile(jct, 0.95),
        "output_tokens_per_s": summary["output_token_throughput_per_second"],
        "ttft_mean_s": summary["ttft_s"]["mean"],
        "ttft_p95_s": summary["ttft_s"]["p95"],
        "latency_mean_s": summary["request_latency_s"]["mean"],
        "latency_p95_s": summary["request_latency_s"]["p95"],
        "tpot_mean_s": summary["tpot_s"]["mean"],
        "cached_tokens": h,
        "cache_hit_fraction": h / tokens,
        "historical_K": k,
        "historical_hit_fraction": sum(r["historical_hits"] for r in engine) / k
        if k
        else None,
        "scheduled_tokens": counters["scheduled_tokens"],
        "no_preemption_compute_identity_pass": counters["scheduled_tokens"]
        == expected_scheduled
        if not preemptions
        else None,
        "preemptions": preemptions,
        "cached_physical_blocks_evicted": policy.get(
            "actual_victims", counters["native_victims"]
        ),
        "planned_victims": policy.get("planned_victims"),
        "safe_supply_deferrals": policy.get("insufficient_safe_supply", 0),
        "rollbacks": policy.get("rollback", 0),
        "allocation_validation_failures": policy.get("validation_failure", 0),
        "schedule_wall_s": counters["schedule_wall_ns"] / 1e9,
        "metrics_preemptions_delta": prom(after, "vllm:num_preemptions_total")
        - prom(before, "vllm:num_preemptions_total"),
        "manifest_sha256": launch["manifest_sha256"],
    }
    row["hashes"] = {
        str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            run / "result.json",
            run / "summary.json",
            run / "engine/requests.jsonl",
            run / "launch.json",
        ]
    }
    outputs = {(r["session_id"], r["call_index"]): r["output_sha256"] for r in requests}
    return row, outputs


def main():
    rows, outputs = [], {}
    for run in sorted(ROOT.glob("*-r[0-9]")):
        if not (run / "summary.json").exists():
            continue
        row, hashes = analyze(run)
        rows.append(row)
        outputs[run.name] = hashes
    if not rows:
        raise SystemExit("No completed valid runs")
    assert len({r["manifest_sha256"] for r in rows}) == 1
    comparisons = []
    for repeat in range(1, 10):
        native = next((r for r in rows if r["run"] == f"native-r{repeat}"), None)
        demand = next((r for r in rows if r["run"] == f"demand-r{repeat}"), None)
        if not native or not demand:
            continue
        change = {
            key: 100 * (demand[key] / native[key] - 1)
            for key in [
                "elapsed_s",
                "mean_jct_s",
                "ttft_mean_s",
                "latency_mean_s",
                "output_tokens_per_s",
            ]
            if native[key]
        }
        match = sum(
            value == outputs[demand["run"]][key]
            for key, value in outputs[native["run"]].items()
        )
        comparisons.append(
            {
                "repeat": repeat,
                "relative_change_percent": change,
                "exact_output_sequences_equal": match,
                "exact_output_sequences_total": 237,
            }
        )
    data = {
        "runs": rows,
        "comparisons": comparisons,
        "mps_clients_verified": False,
        "notes": [
            "Same original 237 requests and gaps; 4 causal session workers.",
            "Each run has a new engine, identical warmup and confirmed successful prefix-cache reset.",
            "Raw prepared token IDs are sent without truncation; actual prompt token hashes and 128 output tokens are checked.",
            "All tasks arrive at t0; JCT includes worker admission waits and tool gaps.",
            "Existing MPS infrastructure is preserved; hidden clients could not be inspected, so GPU exclusivity is unverified.",
            "Exact output token equality is a consistency diagnostic, not task accuracy.",
        ],
    }
    (ROOT / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    fields = [k for k in rows[0] if k != "hashes"]
    with (ROOT / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    lines = [
        "# GPU 对照实测",
        "",
        "10 个任务，237 次请求，原始 pre_gap，4 个因果 session worker，每次输出 128 token。",
        "",
        "|运行|总耗时 s|平均 JCT s|平均 TTFT s|p95 TTFT s|平均请求延迟 s|缓存命中率|抢占|缓存块驱逐|调度 wall s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"|{r['run']}|{r['elapsed_s']:.3f}|{r['mean_jct_s']:.3f}|{r['ttft_mean_s']:.4f}|{r['ttft_p95_s']:.4f}|{r['latency_mean_s']:.4f}|{r['cache_hit_fraction']:.2%}|{r['preemptions']}|{r['cached_physical_blocks_evicted']}|{r['schedule_wall_s']:.3f}|"
        )
    lines.extend(
        [
            "",
            "MPS 客户端列表无法读取，因此独占 GPU 未得到核实。现有 MPS 进程与其他文件均保留。",
            "",
            "原始结果和逐请求记录保存在各运行目录；comparison.json 包含输入工作量、计算量校验、故障计数和文件 SHA。",
            "失败的预检及预热日志已保留，不纳入有效对照。",
        ]
    )
    (ROOT / "GPU_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "runs": [
                    {k: v for k, v in r.items() if k not in ["hashes"]} for r in rows
                ],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
