# vLLM 等待命中升序与按需 KV 驱逐

基于本机 vLLM 0.29.0 对应源码，在独立副本中实施。原版和原 progressive 项目的源码、Git 状态及环境均保留；未连接服务器、未推送、未运行 GPU。

## 已完成与结果

- 两项独立开关：等待请求按实际可复用前缀 token 数升序；真实物理分配时冷/暖分层、最长会话分阶段尾部回收。
- 共享物理块必须无活动引用，并同时处于所有已知 owner 的安全尾部；未知归属保护并可查询数量。供给不足返回延期，不回退原生 victim。
- 无 hash 空闲块优先；计划中的 cached IDs 就是实际分配的 IDs。失败回滚原 hash 映射、重复副本顺序、引用和 free queue。
- **CPU：110 项测试通过，2 项子测试通过**。包括原有 59 项和本次新增 51 项；结果见 [cpu-tests.txt](results/demand/cpu-tests.txt)。
- **CPU 微基准：60 个案例，每项 10 次预热 + 60 次稳态，共 3600 个稳态样本**。wall/CPU p50/p95/p99、原始样本、计数器及源码 SHA 见 [报告](results/demand/CPU_BENCHMARKS.md)、[JSON](results/demand/cpu-bench.json)、[CSV](results/demand/cpu-bench.csv)。
- 两种补丁均已正向应用和反向回退，并逐字节校验；见 [patch-validation.json](results/demand/patch-validation.json)。

实测开销不能忽略：S=128、B=8192、D=32 时，分配 wall p50 为原生 **12.19 µs**、新策略 **247.46 µs**；选择器从慢参考的 **2303.58 µs** 降至索引版 **139.38 µs**。W=128 的 CPU 调度样例，原生 **44.56 µs**，两开关启用 **966.75 µs**。这些不是吞吐、JCT 或 GPU 性能结论，也未设置性能通过阈值。

测试执行从真实源码加载的 BlockPool、KVCacheManager、FullAttention manager/coordinator 和 `Scheduler.schedule`；外围配置、编码器接口和 worker 输出封装用桩。完整 EngineCore、模型推理与 GPU 集成未运行，GPU 不在本次要求范围内。

## 当前电脑直接复现

```sh
cd '/Users/zhangtianqi/Downloads/schduling/agent-KV cache/progressive-eviction/预实验/vllm-demand-policy'
DP_PYTHON="$PWD/../vllm-progressive-eviction/.venv/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$DP_PYTHON" -m pytest -p no:cacheprovider tests experiment/test_replay.py -q
PYTHONDONTWRITEBYTECODE=1 "$DP_PYTHON" tools/bench_demand_cpu.py --repeats 60 --warmup 10
"$DP_PYTHON" tools/summarize_demand_cpu.py
"$DP_PYTHON" tools/check_demand_compat.py --source-root worktree
```

使用原有隔离 Python 3.12.7；无需权重、tokenizer、Torch/CUDA 安装或服务启动。重新测量会更新微基准结果，受机器负载影响可能不同。

## 开关

| 等待排序 `VLLM_WAITING_HIT_ASC` | 驱逐 `VLLM_DEMAND_EVICTION` | 行为 |
| --- | --- | --- |
| 0 | 0 | 原生等待顺序与分配，不创建新 runtime；旧周期策略也不运行 |
| 1 | 0 | 命中 token 数升序，原生物理分配 |
| 0 | 1 | 原生等待顺序，新的按需驱逐 |
| 1 | 1 | 两项同时启用 |

默认均为 0，只认字符串 `1` 为开启。`VLLM_DEMAND_TELEMETRY=0` 关闭新计数器，默认开启。启动已适配副本前设置环境变量，例如：

```sh
export VLLM_WAITING_HIT_ASC=1
export VLLM_DEMAND_EVICTION=1
export VLLM_DEMAND_TELEMETRY=1
```

这三行只设置开关，不启动服务。现有模型、采样器、CUDA graph、MPS 和服务参数未改动。复制保留的旧 `manage.py`/GPU 回放工具属于历史策略基础设施；本交付使用下面的完整补丁与只读检查，不通过旧 overlay 文件清单部署新模块。

## 补丁与回退

- 对当前本地 progressive 内容：[progressive-to-demand.patch](results/demand/progressive-to-demand.patch)。
- 对核验过的原生参考内容：[native-to-demand.patch](results/demand/native-to-demand.patch)。该补丁包含新模块及仅用于标识解析/回退保留的旧模块。
- 对应版本 SHA：[source-versions.json](results/demand/source-versions.json)。

两个补丁各自对应不同基线。在另一个已复制、已通过兼容性检查的源码根目录中，用 `git apply --check /绝对路径/补丁.patch` 检查后再 `git apply /绝对路径/补丁.patch`；回退使用同一补丁的 `git apply --reverse`。补丁路径以 `vllm/` 开头，不能直接在 `site-packages/vllm/` 内应用。没有覆盖用户原目录。

## 部署前只读检查

将本交付的 `tools/check_demand_compat.py` 和 `results/demand/source-versions.json` 保持相对目录关系放到目标机，由用户用现有服务解释器执行：

```sh
"/home/zhaohairui/vllm_env/bin/python" tools/check_demand_compat.py > server-source-check.json
```

上述解释器是用户提供的历史路径，本轮未验证服务器是否仍使用它。检查程序只读取版本元数据和解析出的 vLLM 文件路径、计算源码 SHA；不导入引擎、不安装、不上传、不连接服务。`matches` 为空或退出码非 0 时应先适配源码。此处没有执行部署或启动命令。

## 机制、验收与限制

见 [DESIGN_AND_VALIDATION.md](docs/DESIGN_AND_VALIDATION.md)。初始核对记录在 [demand-source-audit.md](docs/demand-source-audit.md)；其“待确认/尚未实施”段落是实施前快照，用户随后已全部确认，当前状态以本 README 和验收报告为准。历史 README 保存在 `docs/legacy-README.md`。
