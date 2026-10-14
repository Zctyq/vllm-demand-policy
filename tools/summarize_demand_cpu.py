"""Render measured CPU results and verify their recorded source hashes."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/demand"
data = json.loads((OUT / "cpu-bench.json").read_text())
drift = [p for p, sha in data["source_sha256"].items()
         if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha]
if drift:
    raise SystemExit(f"Benchmark source changed: {drift}")
lines = ["# CPU 微基准实测", "", f"平台：{data['platform']}；Python {data['python'].split()[0]}。",
         "", f"每项 {data['warmup']} 次预热、{data['repeats']} 次稳态测量；总计 {len(data['cases'])} 个案例。",
         "时间单位为微秒；p95/p99 使用 nearest-rank；原始样本、预热分布、计数器和源码 SHA 位于 cpu-bench.json。",
         "", "仅执行实际源码提取的 CPU 方法。配置、worker 输出封装和编码器接口使用桩；不包含 EngineCore、GPU、推理输出序列化或网络。",
         "", "状态构造在计时外；每个 allocation/schedule 样本重新建立相同状态。选择测试恢复规划后的状态；过期处理和堆清理单独计时。",
         "cache_release_maintenance 计入新缓存建立、owner 注册和释放的事件开销。maintenance_reuse 计入真实复用时间更新。",
         "", "| 组件 / 参数 | wall p50 | wall p95 | wall p99 | CPU p50 | CPU p95 | CPU p99 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
csv_rows = []
for row in data["cases"]:
    params = {k: v for k, v in row.items() if k not in (
        "steady_wall", "steady_cpu", "warmup_wall", "warmup_cpu",
        "raw_wall_ns", "raw_cpu_ns", "last_counters")}
    label = ", ".join(f"{k}={v}" for k, v in params.items())
    values = [row[clock][q] for clock in ("steady_wall", "steady_cpu")
              for q in ("p50_us", "p95_us", "p99_us")]
    lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    csv_rows.append([label, data["repeats"], *values])
lines += ["", "## Python 元数据内存", "",
          "对 policy 和 clean-block 索引的可达对象进行 sys.getsizeof 去重估算，包含容器开销；不含 native KV pool、GPU 内存、解释器分配器碎片和 waiting 请求对象。此离线检查不在热路径执行。", ""]
for row in data["metadata_bytes"]:
    lines.append(f"- S={row['S']}，B={row['B']}：{row['reachable_python_bytes']:,} bytes。")
lines += ["", "## 解释范围", "",
          "新策略相对原生存在明显额外 CPU 成本；更快的索引版相对慢参考模型的改善，不表示相对原生更快。",
          "W 增大时首次前缀查询、依赖注册和建堆成为主要成本。稳定缓存下复用打分，hash 变化只标脏依赖请求；下一选择前统一重算。",
          "固定 S 的 B 扩展测试观察稳态选块；expiry_burst 显式展示 E 个新到期事件的集中成本，expiry_heap_cleanup 显式展示有界懒项清理成本。不能把稳态时间当成无过期事件的全程保证。",
          "列表式长度索引更新有 O(S) 移位最坏开销。数据规模或共享 fan-out 继续增大时，需要根据本表重新评估。",
          "没有给出吞吐/JCT/TTFT 收益，也没有设定性能通过阈值。本报告只证明可复现的 CPU 时间分布。"]
(OUT / "CPU_BENCHMARKS.md").write_text("\n".join(lines) + "\n")
with (OUT / "cpu-bench.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["case", "samples", "wall_p50_us", "wall_p95_us", "wall_p99_us",
                     "cpu_p50_us", "cpu_p95_us", "cpu_p99_us"])
    writer.writerows(csv_rows)
print(json.dumps({"cases": len(data["cases"]), "source_drift": drift,
                  "steady_samples": len(data["cases"]) * data["repeats"]}))
for row in data["cases"]:
    if ((row["component"] == "schedule" and row.get("W") == 128)
        or (row["component"] in ("selection", "allocation") and row.get("B") == 8192
            and row.get("D") == 32)):
        print({k: v for k, v in row.items() if k not in (
            "raw_wall_ns", "raw_cpu_ns", "last_counters", "warmup_cpu", "warmup_wall")})
