# vLLM 0.29.0 渐进式 KV 缓存淘汰预实验

## 当前交付状态

这是一份基于本地真实源码完成的原型，不是已通过 RTX 5090 验收的部署。

- 原始目录：/Users/zhangtianqi/Downloads/vllm-0.29.0，未修改。
- 独立目录：本目录下 worktree/，分支 progressive-eviction。
- 已核对 8 个缓存、调度和请求入口文件，与官方 v0.29.0 commit 98dff2a81d747d1dba01a47f939f48c3526d4206 字节一致。逐文件 SHA256 在 results/local_environment.json。
- 本机是 macOS arm64。本次测试用隔离 Python 3.12.7 环境，没有安装或升级现有 Torch/CUDA/vLLM。
- CPU 机制测试执行的是从实际源码加载的 BlockPool、hash map、free queue 方法，以及完整的新 policy/runtime；重依赖的导入用最小桩替代。它不是完整 EngineCore/GPU 测试。
- GPU 烟测、真实 Qwen 推理、50-session 六次性能对照尚未执行。真实数据/tokenizer 预检状态见 results/RESULTS.md。没有填造吞吐、延迟或收益。
- 已读取 lmcache_50_audit 导出：50个完整session、1090次LLM调用；1090/1090输入的原脚本、新脚本及原清单token数/哈希一致。每个session为5–50次调用，仅4个属于原先优先的10–12范围。当前准备保留原50个完整任务，明确记录这一差异。
- 本轮本地实施不通过 SSH、VS Code 或其他方式连接用户服务器。
- 本次按审查要求修订sampler开关、校准并发和清单准备/打包流程；结果见 results/RESULTS.md。核心淘汰算法保持原行为。
- GitHub源码仓库不包含 results/manifest_50.json、results/calibration_8.json、原始trace或tokenizer。当前用户电脑仍保留两份大清单；源码克隆需要按第7节准备，含数据实验包另见第10节。

## 1. 本地立即复现

在当前电脑可直接执行：

    cd /Users/zhangtianqi/Downloads/vllm-progressive-eviction
    .venv/bin/python -m pytest tests experiment/test_replay.py -v
    .venv/bin/python tools/gpu_smoke.py --self-test --output results/gpu-smoke-selftest.json

第二条只做 CPU 边界自检，发送 0 次 HTTP 请求。它不是 GPU 验收通过。

在另一台电脑重建测试环境（只安装测试工具）：

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements-local.lock.txt
    .venv/bin/python -m pytest tests experiment/test_replay.py -v

当前修订测试日志见 results/review_2287e39/cpu_tests.txt、cpu_tests.xml；results/cpu_tests.txt保留初次实施记录。测试覆盖100→80→48、独立重访命中80/48、出生保护、逻辑miss恢复热度、阈值合并、共享/活跃/别名保护、槽位代次、队列计数、native路径对照，以及首次lookup与实际采用K的区别。历史owner与保守共享owner分开，未执行或取消的lookup不产生历史计算量。新增回归覆盖sampler环境继承、直接runtime门槛、缺数据早停、校准选择/并发及两种打包模式。

## 2. 修改文件

| 文件 | 作用 |
| --- | --- |
| worktree/vllm/v1/core/progressive_policy.py | 逻辑前缀树、按块冷热/出生/owner、ghost、阈值、私有连续尾部选取 |
| worktree/vllm/v1/core/progressive_runtime.py | 配置门槛、请求ID适配、执行后提交历史、物理适配器、日志 |
| worktree/vllm/v1/core/sched/scheduler.py | 调度起点检查、成功admission、实际执行返回、正常完成、reset/shutdown |
| worktree/vllm/v1/core/kv_cache_manager.py | 首次逻辑lookup观测，保留原三项返回值 |
| worktree/vllm/v1/core/block_pool.py | 物理代次和原生失效通知，保留分配/引用/释放语义 |

kv_cache_utils.py 未改；直接使用原生 remove/prepend_n/evict_blocks。完整源码调用链和行号在 docs/source-audit.md。

patches/01-observe-only.patch 先只允许 native 观测；patches/02-enable-progressive.patch 再开放 progressive。也可一次应用 patches/progressive-eviction.patch。三者不能重复叠加。补丁分阶段应用、完整应用及反向回退已逐字节验证，见 results/patch-validation.json；分阶段补丁验证不代表已完成分阶段GPU验收。默认没有实验环境变量时仍走原生路径。CPU 的 native 对照包含未打补丁的原版 allocator 和启用相同观测的 native allocator。

## 3. 策略边界

原生 vLLM 本来就按块回收。本原型增加的是主动分比例 hard eviction：

- 正常完成请求跨过10的倍数后，只标记pending；在下一次同步schedule起点合并检查。
- 新节点跳过出生窗口；第一个完整冷窗口删剩余私有尾段20%，之后40%，向上取整。
- 访问按请求首次逻辑lookup记一次，已淘汰后的miss也恢复热度。
- 逐块冷热等价于在部分访问边界切段；共享owner使用已观察到的session集合保守保护，不读取未来trace。
- 完整hash+cache group是逻辑身份；物理slot另带generation。全部副本均安全才使逻辑块失效；任何活跃副本或未知别名均停止连续尾部裁剪。
- 调原生evict_blocks删除hash，再remove/prepend_n调整已空闲槽位。free count不变；之后原生分配可继续回收其他有效缓存。
- 持久ghost只用于已实际执行的历史内容；原生覆盖撤销resident关联。没有CPU备份、传输、预取或attention kernel修改。
- 比例按每次剩余逻辑块算；物理副本数量另记。累计计数是失效事件次数，同一内容重建后再次淘汰会再次计数。

首版只支持单GPU、单Engine、单组因果FullAttention、hash block size等于物理block size、同步调度且max_concurrent_batches=1。拒绝KV/EC connector、KV offload、speculative decoding、流式输入和多输出。两组都要求prefix caching开启、enforce_eager、显式非FlashInfer attention后端，并显式关闭FlashInfer sampler。Qwen2.5配置是否满足这些门槛，需要实际启动核验。

## 4. 自定义开关

以下均为本补丁的环境变量，不是vLLM官方CLI：

| 变量 | 默认 |
| --- | --- |
| VLLM_PE_POLICY | native |
| VLLM_PE_LOG_DIR | 不设则native不启用实验观测 |
| VLLM_PE_RUN_ID | 启动脚本给每轮唯一ID |
| VLLM_PE_CHECK_INTERVAL | 10 |
| VLLM_PE_FIRST_COLD_RATIO | 0.20 |
| VLLM_PE_LATER_COLD_RATIO | 0.40 |

native/progressive使用同一份最终构建、相同日志级别。没有session标识的缓存按未知owner保护；正式回放必须完成客户端→EngineCore ID关联，否则统计失败。

另外，vLLM原生环境变量 `VLLM_USE_FLASHINFER_SAMPLER=0` 必须在两组一致。manage.py在创建服务进程时强制设置为0，覆盖父环境遗留的1；EngineCore记录解析后的 `use_flashinfer_sampler=false`，启动核验和配对配置签名都检查它。直接绕过manage启动带实验观测的runtime，也必须先设置此变量，否则明确报错。非FlashInfer attention backend与sampler是两个独立开关。

    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_LOGGING_LEVEL=INFO

## 5. 保留现有可用环境的部署方式

以下命令供用户在拥有现有可用vLLM环境的Linux NVIDIA机器上手动运行。脚本不会SSH或远程执行。当前Mac上不能完成这一步。

先激活原来已经可用的vllm_env，令PE_PYTHON指向该环境的绝对解释器路径。不要升级Torch/CUDA，也不要对本源码运行main/nightly开发安装。

    export PE_PYTHON="/实际已验证环境/bin/python"
    "$PE_PYTHON" tools/manage.py overlay --dest "$PWD/overlay-v029"

这个命令先核对已安装版本和8个原始源码SHA；不一致即停止。通过后完整复制已安装vllm包及其兼容扩展/伴随库，再替换5个Python文件，保留原包。逐个校验复制的.so字节和子进程实际导入路径。它不会把零散.py文件直接塞进PYTHONPATH，也不承诺跨版本ABI兼容。

configs/server.example.json 的未知值为null，必须按现有启动记录填写为 configs/server.json，不能照猜。包括模型路径、解释器、实际dtype、block size、已核实总KV块数、调度预算、并发上限、chunked prefill、非FlashInfer后端、GPU和端口。gpu-memory-utilization相同不代表容量相同，因此启动时使用固定num-gpu-blocks-override，并再次核对EngineCore实际数量。

启动器只接受一张RTX5090、一个空闲专用端口；检测到该GPU已有计算进程就退出，不停止它们。服务绑定127.0.0.1，健康检查绕过HTTP代理，以唯一服务别名确认端口属于本次API进程。EngineCore路径和源码SHA必须属于选定完整overlay。

填好configs/server.json后，以下变量都从同一份配置读取：

    export PE_PORT=$("$PE_PYTHON" -c 'import json; print(json.load(open("configs/server.json"))["port"])')
    export PE_MODEL=$("$PE_PYTHON" -c 'import json; print(json.load(open("configs/server.json"))["model"])')
    export PE_BLOCK_SIZE=$("$PE_PYTHON" -c 'import json; print(json.load(open("configs/server.json"))["block_size"])')
    export PE_NUM_GPU_BLOCKS=$("$PE_PYTHON" -c 'import json; print(json.load(open("configs/server.json"))["num_gpu_blocks"])')

## 6. 先观测，再GPU机制验收

native观测：

    "$PE_PYTHON" tools/manage.py start --profile configs/server.json \
      --policy native --run-id smoke-native --run-dir runs/smoke-native

如果要放在screen里查看日志：

    screen -S pe-observe
    tail -F runs/smoke-native/server.log | tee runs/smoke-native/screen.log

在另一个终端运行机制烟测；PE_PORT和PE_MODEL必须与profile一致。请求使用本次进程随机模型别名：

    PE_SERVED_MODEL=$("$PE_PYTHON" -c 'import json; print(json.load(open("runs/smoke-native/process.json"))["launch_alias"])')

    "$PE_PYTHON" tools/gpu_smoke.py --run \
      --base-url "http://127.0.0.1:$PE_PORT" \
      --model "$PE_SERVED_MODEL" \
      --engine-log-dir runs/smoke-native/engine \
      --server-log runs/smoke-native/server.log \
      --tokenizer "$PE_MODEL" --output runs/smoke-native/gpu-smoke.json
    "$PE_PYTHON" tools/manage.py stop --record runs/smoke-native/process.json

随后progressive使用同一构建和profile：

    "$PE_PYTHON" tools/manage.py start --profile configs/server.json \
      --policy progressive --run-id smoke-progressive --run-dir runs/smoke-progressive
    PE_SERVED_MODEL=$("$PE_PYTHON" -c 'import json; print(json.load(open("runs/smoke-progressive/process.json"))["launch_alias"])')
    "$PE_PYTHON" tools/gpu_smoke.py --run \
      --base-url "http://127.0.0.1:$PE_PORT" \
      --model "$PE_SERVED_MODEL" \
      --engine-log-dir runs/smoke-progressive/engine \
      --server-log runs/smoke-progressive/server.log \
      --tokenizer "$PE_MODEL" --output runs/smoke-progressive/gpu-smoke.json
    "$PE_PYTHON" tools/manage.py stop --record runs/smoke-progressive/process.json

烟测要求至少104个总块、上下文容纳100*block_size+2 tokens。用独立fixture分别检查80、48；100B+1输入避免末token重算边界误判。原生组应保持100块，实验组验证真实请求采用的K和真实hash失效日志。仅打印“evicted”不能通过此脚本。

烟测还要求本次服务日志包含原生CUDA sampler初始化信息 `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.`，并拒绝启用FlashInfer采样的冲突信息；记录日志路径、读取时的SHA和匹配行。缺少INFO worker日志时验收报错，不能用EngineCore配置记录替代。这是初始化路径证据，不声称temperature=0请求执行了随机采样。

## 7. 数据预检、校准和六次配对实验

### 先准备清单：选一种路径

本轮工作量固定为50个完整Agent session、1090次LLM调用，每个session为5–50次调用。默认prepare筛选仍为10–12，当前数据必须显式传5–50。原导出、tokenizer和两份大清单不在GitHub源码仓库中；小型预检报告、ID/哈希及调用数统计保留在results。以下CPU命令在项目根目录运行：

    export PE_PYTHON="$PWD/.venv/bin/python"

**路径A：复用用户已有清单。** 当前Mac中的results/manifest_50.json和results/calibration_8.json保留在原处。其他执行机可接收用户已有清单或含数据包，将清单放入同名results路径；也可以给--manifest传实际位置。完整清单已经含messages/token IDs，回放不使用其中记录的Mac原数据绝对路径。这条路径不需要tokenizer或模型权重：

    "$PE_PYTHON" experiment/replay.py validate \
      --manifest results/manifest_50.json --expected-sessions 50 \
      --concurrency 4 --expected-max-model-len 32768

**路径B：用用户已有原始导出和tokenizer重建。** 本机原导出实际位置是 `/Users/zhangtianqi/Downloads/lmcache_50_audit`。Linux上应将PE_EXPORT设为该执行机实际保存导出的目录，不使用Mac绝对路径。需要requests.jsonl、original_manifest.json、tokenizer/，均不随源码提供。分词不需要Torch、权重或GPU；隔离依赖版本见requirements-local.lock.txt。

    # 仅在当前Mac使用这个已存在的路径；其他执行机设置自己的PE_EXPORT。
    export PE_EXPORT="/Users/zhangtianqi/Downloads/lmcache_50_audit"
    "$PE_PYTHON" experiment/replay.py prepare \
      --data "$PE_EXPORT/requests.jsonl" --tokenizer "$PE_EXPORT/tokenizer" \
      --reference-manifest "$PE_EXPORT/original_manifest.json" \
      --manifest results/manifest_50.json --max-model-len 32768 \
      --gap-unit s --gap-evidence docs/pre-gap-evidence.txt --order-field call \
      --sessions 50 --min-calls 5 --max-calls 50 --selection-seed 20261001

已有清单时prepare拒绝覆盖；重新预检使用新输出路径。原回放将pre_gap当作上一响应结束后的秒数等待，按已有回放约定保留；缺少上游时间戳/字段字典，未独立核实LMCache源定义。旧分词审计的最大输入32152，固定128后32280≤32768；预算合计139520输出tokens，实际GPU输出仍须逐条核验。本次修订不把旧分词审计记成新执行。

### 从正式清单生成或核验8任务校准集

如果缺calibration_8.json，先用已有选择记录重建。这条命令验证旧记录的父SHA、选择seed和ID，保持原来的8个完整session、163次调用及相对顺序：

    "$PE_PYTHON" experiment/replay.py subset \
      --manifest results/manifest_50.json --sessions 8 \
      --selection results/calibration_selection.json --out results/calibration_8.json

若已经有该文件，直接执行下方validate，不覆盖它。若路径B重建使父清单的字节SHA变化，旧选择记录会被拒绝；显式沿用seed创建新记录：

    "$PE_PYTHON" experiment/replay.py subset \
      --manifest results/manifest_50.json --sessions 8 --seed 20261001 \
      --out results/calibration_8.json

新生成的旁置calibration_8.selection.json记录父SHA、输出SHA、seed、ID及调用数；含数据打包优先核对它，没有旁置记录时使用原calibration_selection.json。上述两种subset路径择一，输出已经存在时命令拒绝覆盖。

    "$PE_PYTHON" experiment/replay.py validate \
      --manifest results/calibration_8.json --expected-sessions 8 --concurrency 4

8个任务只用于C=4、必要时C=8的校准。C是活跃session名额，工具等待也占名额；实际同时在途的HTTP请求数可低于C，不能将C称为GPU并发实测值。CLI拒绝C大于任务数，日志分别记录配置上限、worker数和客户端观测到的在途HTTP峰值。

确需C=16时，先创建至少16个完整任务的子集，再验证；不要求例行多扫一轮：

    "$PE_PYTHON" experiment/replay.py subset \
      --manifest results/manifest_50.json --sessions 16 --seed 20261001 \
      --out results/calibration_16.json
    "$PE_PYTHON" experiment/replay.py validate \
      --manifest results/calibration_16.json --expected-sessions 16 --concurrency 16

当前本机已按此命令生成16个完整任务、321次调用，记录在results/calibration_16.selection.json；大清单只在本地/含数据包中提供。若文件已经存在，执行validate即可。

### 清单校验通过后再启动校准服务

GPU执行前将PE_PYTHON重新设为第5节现有vllm_env解释器，使用已填写的profile。以下用8任务/C=4；C=16时改为16任务清单、concurrency=16及全新的run-id/目录。同步调度保留continuous batching，不把max_num_seqs设为1。

    "$PE_PYTHON" tools/manage.py start --profile configs/server.json \
      --policy native --run-id calibration-native --run-dir runs/calibration-native
    PE_SERVED_MODEL=$("$PE_PYTHON" -c 'import json; print(json.load(open("runs/calibration-native/process.json"))["launch_alias"])')
    "$PE_PYTHON" experiment/replay.py run \
      --manifest results/calibration_8.json \
      --base-url "http://127.0.0.1:$PE_PORT" --model "$PE_SERVED_MODEL" \
      --policy native --seed 17 --concurrency 4 \
      --expected-block-size "$PE_BLOCK_SIZE" --expected-num-gpu-blocks "$PE_NUM_GPU_BLOCKS" \
      --engine-log-dir runs/calibration-native/engine --out runs/calibration-native/client
    "$PE_PYTHON" tools/manage.py stop --record runs/calibration-native/process.json

检查client/summary.json的成功数、cache_pressure_observed和引擎抢占/原生有效覆盖，再固定正式C。校准清单不进入正式三对统计。

机制通过且C确定后：

    "$PE_PYTHON" tools/manage.py paired --profile configs/server.json \
      --manifest results/manifest_50.json --concurrency 4 \
      --seeds 17 29 43 --out runs/formal

paired在任何GPU启动和输出目录创建之前检查正式清单存在、schema、50个完整任务、输入+128预算、token哈希/顺序、C与任务数，以及profile的max_model_len。缺文件时输出prepare命令。通过后同一GPU顺序运行native17/progressive17、progressive29/native29、native43/progressive43；每轮新进程、新日志和相同profile。每轮预热后确认reset再开始t0。若某轮失败，就保存其真实错误并停止。全部成功才输出paired_results.json和CSV。运行总日志可用：

    "$PE_PYTHON" tools/manage.py paired --profile configs/server.json \
      --manifest results/manifest_50.json --concurrency 4 \
      --seeds 17 29 43 --out runs/formal-2 2>&1 | tee runs/formal-2-console.log

固定128输出并验证实际长度；完整trace输入不加入本地生成答案；客户端单调时钟统一计时。汇总包括任务吞吐、从t0算的平均完成时间、实际输出token、H/K历史损失、主动/原生失效、检查开销和成功/失败数。允许零/负收益；没有主动失效或竞争证据时明确限制结论。

## 8. 日志、停止与回退

- engine/run_config.json：EngineCore真实导入路径/SHA、版本、PID、GPU/KV/调度配置及use_flashinfer_sampler=false。worker初始化证据另保存在烟测输出中。
- engine/requests.jsonl：一次请求的session/call、H、首次lookup诊断、实际执行K、输出和状态。
- engine/policy_checks.jsonl：完成数、实际窗口、合并阈值、目标/实际逻辑与物理淘汰、原因、耗时、前后free count。
- engine/eviction_summary.json：reset以来主动/原生失效累计。原生logical只含已登记的逻辑节点；physical包含所有有效hash覆盖。
- engine/reset.jsonl：实际reset成功及policy/ghost/计数清空证据。
- client/：客户端任务到达/admission/完成、完整逐请求关联、metrics、原始与配对汇总。

只停止有本实验process.json身份记录的进程组：

    "$PE_PYTHON" tools/manage.py stop --record runs/某一轮/process.json

停止后不使用overlay即可回退：原vllm_env包、模型、原始源码和旧回放脚本均未改。沿用原来的启动命令；不要对其他服务执行pkill或killall。API/EngineCore启动失败时保留server.log和process.json供核对。

## 9. 尚待验证

导出报告声明服务器vLLM=0.29.0；这与本地源码版本一致，但不是对当前运行中EngineCore的直接核验。实际Linux已安装构建、5090池容量、启动配置与GPU模型权重路径仍未被本机证实；overlay匹配、EngineCore实际导入、GPU生成、HTTP元数据全链路与六次对照，都由上述脚本验收。真实数据/tokenizer预检已经完成；当前CPU通过不能替代GPU验收，也不能据此宣称任务吞吐改善或显存下降。

## 10. 源码包与含数据实验包

GitHub源码版和source包包含代码、补丁、必要原版快照及小型报告，可以运行CPU测试。它们不包含完整请求清单、原始导出或tokenizer。data包另外包含已准备的正式/校准清单；打包前校验schema、任务数、父SHA、ID、完整calls和选择记录。发现可选16任务清单时一并核验；不是只检查文件存在。

    .venv/bin/python tools/bundle.py plan --mode source
    .venv/bin/python tools/bundle.py create --mode source \
      --output dist/vllm-progressive-eviction-review-source.tar.gz
    .venv/bin/python tools/bundle.py plan --mode data
    .venv/bin/python tools/bundle.py create --mode data \
      --output dist/vllm-progressive-eviction-review-data.tar.gz
    .venv/bin/python tools/bundle.py verify \
      --archive dist/vllm-progressive-eviction-review-data.tar.gz

已有归档时使用新输出名。默认模式仍为data；缺数据时给出第7节prepare/subset命令，source模式仍可正常创建。两份大清单和新校准大文件已排除普通Git跟踪，小型selection/hash/预检报告保留。

两个包都只带本补丁5个Python文件、原kv_cache_utils.py、8个核验用原文件及许可证/工程配置，不重打包完整vLLM仓库。它们足够做CPU测试和从执行机现有完整安装包准备overlay，不可直接作为完整vLLM源码pip安装。BUNDLE_MANIFEST.json逐文件保存SHA256，旁置.verification.json记录校验结果。本机完整worktree和旧归档保留；模型权重、.venv、原始导出和tokenizer都不随包复制。

迁移后按第1节建立CPU测试环境；GPU部署沿用第5节现有可用vllm_env。data清单已含完整输入/token IDs，GPU烟测使用执行机现有模型目录的tokenizer。迁移/上传由用户操作，本工具只在本地生成文件。
