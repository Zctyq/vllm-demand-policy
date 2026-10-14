# Agent 回放与配对统计

此目录改编已有 `replay_concurrent_test_50tasks.py` 的数据清理、工具参数模板适配、token 哈希预检和标准库 HTTP 部分。现有脚本保留原样。

当前已收到本地真实导出 `lmcache_50_audit/`：50 个 session、1090 次 LLM 调用，与 `original_manifest.json` 的任务和原始行号一致。实际每个 session 为 5–50 次调用，仅 4 个满足 10–12 次；使用本批全部任务时必须显式选择 5–50，不截断任务。全部1090次请求的原/新 tokenizer token IDs 及原清单哈希已通过本地核对，输入共18,957,819 tokens，单次范围356–32152；详见 [实际导出审计报告](../results/export_audit/audit.json)。GPU 回放尚未执行，没有真实吞吐量、历史复用损失率或 GPU 端到端验收结果。

上述分词结果属于此前实施记录，本次审查修订没有重新分词。GitHub源码版不包含原始导出、tokenizer、manifest_50.json或calibration_8.json；用户本地两份大清单保留。复用本地清单和在执行机重建的两条路径见[根README第7节](../README.md)。

## 与原脚本的变化

- 用 `session_id` 分组，默认按每个 session 的 LLM 调用数筛选 10–12 轮。message 数不作为调用数。本批完整50任务的例外参数为 `--min-calls 5 --max-calls 50`。
- 可传 `--reference-manifest`，按 `(session_id, call, original_dataset_row)` 对齐原清单，核对 token 数、token SHA256 和原始间隔。任一不一致都会拒绝生成新清单；选满原任务集合时保留原清单的 session 顺序，再由正式回放的顺序 seed 洗牌。
- 全部 50 个任务在同一个客户端单调时钟的 t0 到达；最多 C 个 session 活跃。一个 session 在工具等待期间仍占一个活跃位置。
- C大于任务数直接报错。configured_active_session_limit与effective_worker_count描述客户端session名额；observed_max_inflight_http_requests统计客户端等待send返回的峰值，不等于GPU执行并发。
- 同一 session 严格串行：前次响应完成 → 等待本次 `pre_gap` → 发送本次完整 trace input。首轮立即发送；模型回答不拼入后续输入。
- `pre_gap` 单位必须显式传入，提供已核对来源的本地证据文件。程序检查数值合法性，不根据数值猜单位。
- tokenizer 按实际 chat template 无截断预检；任何一轮超过 `input+128 <= max_model_len`，整个 session 不进入固定任务集。服务返回的真实 `prompt_token_ids` 必须匹配预检。
- 设置 `max_tokens=min_tokens=128, ignore_eos=true, stop=[], stop_token_ids=[]`，还检查实际输出长度和结束原因。无 `cache_salt`。
- 元数据通过已核对 v0.29.0 的 `X-Request-Id` 通道：`pe1.BASE64URL({"r":run_id,"s":session_id,"c":call_index})`，c 从 0 开始。响应 ID 和 EngineCore 记录都必须可解码关联。
- 首次实际执行采用的 H/K 只取引擎观测日志。客户端 response cached_tokens 仅作诊断，不使用相邻输入 LCP 或累计全局命中率充当 H/K。
- 正式 t0 前先预热，再调用确认成功的 `/reset_prefix_cache`，并要求新追加 `reset.jsonl` 证明 policy/ghost/完成计数已重置。预热记录按文件行偏移排除。每轮仅执行一次这一流程。
- 引擎启动证据核对实际导入模块路径/源码哈希、KV block 数和大小；两组摘要比较实际模型、调度、后端和容量配置。
- EngineCore必须明确记录use_flashinfer_sampler=false，且该项进入配对配置签名；GPU初始化日志证据由tools/gpu_smoke.py检查，配置字段本身不能替代worker证据。

## 可执行命令

在项目根目录执行。`PE_PYTHON` 使用明确的解释器路径：本地 CPU 数据预检/测试可设为 `$PWD/.venv/bin/python`；GPU 回放时设为已核对部署环境的 Python 绝对路径。先按项目根 README 启动独立实验服务。此工具不会访问 SSH/VS Code、启动/停止服务或安装软件。

```bash
export PE_PYTHON="$PWD/.venv/bin/python"

"$PE_PYTHON" experiment/replay.py inspect-data \
  --data "$PE_DATA" --out "$PE_RESULT_ROOT/data_inspection.json"

"$PE_PYTHON" experiment/replay.py prepare \
  --data "$PE_DATA" --tokenizer "$PE_MODEL_PATH" \
  --manifest "$PE_RESULT_ROOT/manifest_50.json" \
  --max-model-len "$PE_MAX_MODEL_LEN" \
  --gap-unit "$PE_GAP_UNIT" --gap-evidence "$PE_GAP_EVIDENCE" \
  --sessions 50 --min-calls 10 --max-calls 12 --selection-seed 20261001
```

`PE_GAP_UNIT` 必须是核实后的 `s`、`ms` 或 `us`。证据文件记录实际数据字段的单位、该间隔是否为上一轮响应结束到本轮提交的等待，以及 session 内排序依据。如果数据有明确轮号，可加 `--order-field 实际字段名`。默认是已核实的 session 内数据行顺序。证据文件非空检查不能替代对来源的人工核实。

预检清单保存选中 session 的完整 messages 和 token IDs，保证后续六次输入固定；文件可能较大。若不足 50 个合格任务，命令失败并保存 `.preflight.json`，不会静默抽短任务或截断输入。

### 当前导出的完整50个任务

这批已固定的任务使用下面的导入命令，不重新抽取 10–12 轮子集。`PE_EXPORT` 指向本地 `lmcache_50_audit` 目录。原回放脚本将 `pre_gap` 作为“上一响应完成后等待的秒数”；导出缺少数据源时间戳/字典，证据文件应明确这是沿用既有回放语义，而不是通过数值推断原始数据源单位。所有50任务需完成 token/hash 核对及长度检查才会写出清单。

```bash
: "${PE_EXPORT:?请先设置当前执行机原始导出的实际目录}"

"$PE_PYTHON" experiment/replay.py prepare \
  --data "$PE_EXPORT/requests.jsonl" --tokenizer "$PE_EXPORT/tokenizer" \
  --reference-manifest "$PE_EXPORT/original_manifest.json" \
  --manifest "$PE_RESULT_ROOT/manifest_current50_fixed128.json" \
  --max-model-len 32768 --gap-unit s --gap-evidence "$PE_GAP_EVIDENCE" \
  --order-field call --sessions 50 --min-calls 5 --max-calls 50 \
  --selection-seed 20261001
```

提供参考清单时，输入必须是匹配该清单的完整导出；未知、重复或缺失的原始请求、token/hash/间隔差异都会报错。新清单记录参考文件路径与 SHA256、调用数范围、原清单 session 顺序；每次调用同时保留导出位置 `dataset_row`、原数据行号 `original_dataset_row` 和1起始原轮号 `source_call`。预检不会修改导出文件。可选参数未传时，原有预检方式继续可用。

### 校准子集与GPU启动前检查

以下操作不读tokenizer、不发HTTP。旧8任务优先复用现有selection记录（严格要求父SHA和seed/IDs一致）；输出存在时使用validate，不覆盖清单：

```bash
"$PE_PYTHON" experiment/replay.py subset \
  --manifest results/manifest_50.json --sessions 8 \
  --selection results/calibration_selection.json --out results/calibration_8.json
"$PE_PYTHON" experiment/replay.py validate \
  --manifest results/calibration_8.json --expected-sessions 8 --concurrency 4
```

在执行机重新prepare后，父清单字节可能因记录的路径不同而改变。此时旧selection不能冒充匹配；显式省略--selection并传--seed 20261001，生成新的旁置`.selection.json`。选择仍按完整任务进行，旁置记录保存父SHA、输出SHA、任务ID、seed和调用数。

8任务仅校准C=4或8。确需C=16，先生成至少16个任务：

```bash
"$PE_PYTHON" experiment/replay.py subset \
  --manifest results/manifest_50.json --sessions 16 --seed 20261001 \
  --out results/calibration_16.json
"$PE_PYTHON" experiment/replay.py validate \
  --manifest results/calibration_16.json --expected-sessions 16 --concurrency 16
```

validate检查schema、任务/调用数、token哈希、调用顺序、输入+128预算和并发；可用--out保存小型报告。run在读取引擎证据/发送HTTP前执行同一检查；manage.paired在启动GPU前验证50任务及profile预算。清单缺失会列出prepare/subset准备命令。

以下命令要求独立服务已启动、空闲，且 EngineCore 已在 `$PE_ENGINE_LOG` 写出 `run_config.json`。`PE_POLICY` 必须与引擎的 `VLLM_PE_POLICY` 一致；CLI `--run-id` 默认使用引擎的 `VLLM_PE_RUN_ID`。

```bash
"$PE_PYTHON" experiment/replay.py run \
  --manifest "$PE_RESULT_ROOT/manifest_50.json" \
  --base-url http://127.0.0.1:8001 --model "$PE_SERVED_MODEL" \
  --policy "$PE_POLICY" --seed "$PE_ORDER_SEED" --concurrency "$PE_CONCURRENCY" \
  --expected-block-size "$PE_BLOCK_SIZE" --expected-num-gpu-blocks "$PE_NUM_GPU_BLOCKS" \
  --engine-log-dir "$PE_ENGINE_LOG" --out "$PE_CLIENT_RUN"
```

每次使用全新的客户端目录、引擎日志目录和引擎 run_id。`expected-num-gpu-blocks` 是实际探测并固定的 KV 池块数，不是估算 token 容量，也不能只比较 gpu-memory-utilization。

管理工具为每轮服务生成独立模型别名，`--model` 使用该轮实际公布的别名。HTTP 请求仍发送这个别名；客户端配置另将它保存为 `request_model_alias`。配对比较的 `model` 取 EngineCore 记录的实际模型路径，因此不同启动别名不会误判成不同模型，真正的模型路径变化仍会拒绝配对。

先用8任务/C=4检查竞争和稳定性，必要时尝试C=8；C=16要求至少16任务。正式清单固定50个任务、一个C。每轮实际回放同时保存一秒一次的`/metrics`诊断；历史复用损失率不由该诊断计算。

```bash
"$PE_PYTHON" experiment/summarize.py plan --seeds 17 29 43 --out "$PE_RESULT_ROOT/run_plan.json"
```

计划顺序为 native17、progressive17、progressive29、native29、native43、progressive43。每次由根 README 的启动/停止命令切换专属进程，保持同一张卡及实际配置。完成后：

```bash
"$PE_PYTHON" experiment/summarize.py summarize \
  "$PE_RESULT_ROOT/native17/client" "$PE_RESULT_ROOT/progressive17/client" \
  "$PE_RESULT_ROOT/progressive29/client" "$PE_RESULT_ROOT/native29/client" \
  "$PE_RESULT_ROOT/native43/client" "$PE_RESULT_ROOT/progressive43/client" \
  --out "$PE_RESULT_ROOT/paired_results.json"
```

输出 JSON 保存逐次原始结果与配对差值，CSV 保存逐次主要指标。对照缺失、失败、任务数不等于 50、输入/输出/容量/引擎配置不一致、缺少 H/K 或 reset 证据时，该对不进入性能比较；不足三个有效配对时退出码为 2。允许损失增加、吞吐下降或没有变化。

另外保存 `proactive_eviction_exercised` 和 `cache_pressure_observed`。后者只把原生分配覆盖有效缓存或引擎实际抢占作为竞争证据；请求排队/占用率高本身不够。progressive 没有主动失效或整对没有竞争证据时，仍保留合法的零收益/负收益结果，并在 `qualification_notes` 说明本轮未验证的机制。不能据此声称已经检验了“缓存竞争下渐进淘汰”的效果。两组的检查间隔和两级冷却比例均参与引擎配置签名比较。

## 日志与统计

| 文件 | 内容 |
| --- | --- |
| `run_config.json` / `engine_startup.json` | 客户端固定参数、实际 EngineCore 配置快照、模块哈希、reset 证据 |
| `client_arrivals.jsonl` | 所有任务到达偏移均为 0 |
| `client_tasks.jsonl` | arrival/admission/completion 单调时钟偏移、成功状态、完成调用数 |
| `client_requests.jsonl` | 请求起止、实际 gap、token 数、预检哈希、响应 ID、失败原因 |
| `requests.jsonl` | 成功客户端请求与 EngineCore H/K 的严格关联结果 |
| `metrics.jsonl` | 运行/等待请求、占用、抢占和累计命中诊断；另有前后原始快照 |
| `policy_checks.jsonl` | 仅正式 reset 后实际执行的 checkpoint，含 free count、耗时和失效数 |
| `eviction_summary.json` | 仅正式 reset 后主动 logical/physical 与原生 logical/physical 计数 |
| `summary.json` | 时间、任务数、实际 token、H/K、失效和策略开销汇总 |

任务吞吐量为 `N / (最后完成 - t0)`；平均任务完成时间为 `mean(最终完成 - t0)`，包含 admission 前等待和原始工具间隔。失败批次不计算等工作量吞吐/平均完成时间。`observed_output_tokens_including_invalid_responses` 额外保留长度验证失败的可读响应 token 数；传输失败时未知的服务端输出不能推测。

历史复用损失率为 `sum(max(0,H-K))/sum(H)`，分母 0 为 null（N/A）。原生 `native_logical` 仅统计已观测逻辑节点的最后副本失效；`native_physical` 统计原生分配覆盖的所有仍带有效 hash 的物理副本。主动失效之后的槽位覆盖不重复算缓存损失。策略耗时只统计实际发生的检查；最后一轮完成后尚未消费的 pending checkpoint 不通过额外请求强行触发。

## 本地验证

```bash
"$PE_PYTHON" -m unittest discover -s experiment -p 'test_*.py' -v
```

测试覆盖因果顺序与 C 上限、t0 到达/排队时间、完整输入固定、gap 单位和无截断、参考清单顺序/原行号保留和错误哈希拒绝、ID 透传解析、实际 token 验证、H/K 严格关联、容量检查、reset 证据新鲜度、free count 与失效统计一致、失败/不匹配配对拒绝。使用人工构造消息和模拟响应，**不能作为 GPU 验收或真实 trace 预检通过的证据**。
