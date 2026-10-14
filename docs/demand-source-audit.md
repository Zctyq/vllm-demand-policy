# 新需求实施前审计（2026-10-12）

## 位置、版本与保护

实际父目录为 `/Users/zhangtianqi/Downloads/schduling/agent-KV cache/progressive-eviction/预实验`，磁盘目录名是 `schduling`。

| 对象 | 相对父目录的位置 | 本地证据 |
| --- | --- | --- |
| 用户原版 | `vllm-0.29.0` | 无 Git、无生成的 `_version.py`；目录名不证明已安装运行版本 |
| progressive 项目 | `vllm-progressive-eviction` | HEAD `96ec749efd9fe64d9b2bcafb2362aa815725a936`，有未提交和暂存修改 |
| 实际源码 | `vllm-progressive-eviction/worktree` | HEAD `b356c4f32976677994e9bfef4e09e31366da1a77`；三文件有修改、两策略文件未跟踪 |
| 本地参考 | `vllm-progressive-eviction/reference/vllm-v0.29.0` | HEAD `98dff2a81d747d1dba01a47f939f48c3526d4206`，tag `v0.29.0` |
| 独立副本 | `vllm-demand-policy` | 复制 progressive 当前工作文件，保留原有改动 |

重新计算的 10 个关键文件 SHA 显示用户原版与本地参考逐字节一致；独立副本与 progressive 对应文件一致。版本与 SHA 在 `audit/provenance.json`，原 dirty 状态与已/未暂存 diff 在 `audit/*-status.txt`、`*-staged.txt`、`*-unstaged.txt`。复制文件摘要共 13642 项，见 `audit/copy-sha256.json`。未复制虚拟环境、Git 元数据、大数据清单和 dist 包；原数据保留。

GitHub 参考提交 `2287e39` 不在当前根仓库对象库中，未视为本地 HEAD，未远程获取或比较。已读取两份源码的 AGENTS.md、原开发说明和相关测试。使用已有 `.venv/bin/python` 3.12.7，没有安装依赖或修改模型环境。

## 分配调用链与接入风险

以下行号为此次源码审计时的行号。

1. `sched/scheduler.py:507` 的 `schedule` 先 running（含继续 chunked prefill），后 waiting；分配在约 666、1085 行。
2. `KVCacheManager.allocate_slots` 保留 full-sequence-fit、水位、reserved blocks、lookahead、已有/命中块等检查。
3. `kv_cache_coordinator.py:225` 的 `allocate_new_computed_blocks` 先 touch 全部组的本地命中，再处理外部缓存分配。
4. `single_type_kv_cache_manager.py:148` 的 `get_num_blocks_to_allocate` 返回新增块加尚空闲的命中块占用，还可能含 partial-hit CoW；**不能直接当作规格 N**。
5. 同文件 `allocate_new_blocks:335` 可能先为 partial-hit 单独分配 CoW，再按 block table 计算新增块。
6. `block_pool.py:648` 的 `get_new_blocks` 是实际物理分配入口：弹出确定 IDs，失效对应旧 hash，再增加 refcount。`touch` 移除空闲命中块并增加引用。

新事务必须预检安全供给，不能在底层选择失败后留下已 touch 引用或已修改 block table。实际分配需重验 IDs、代次、hash/别名、owner、空闲状态。若首版排除多组/CoW 等配置，必须在启用时明确拒绝，不能隐式走原生 victim。

## Waiting 与只读查找

- waiting 在约 784 行开始，受 token/input budget、running 上限、LoRA、阻塞状态、stale outputs、并发 prefill、编码器等条件约束。
- `num_computed_tokens == 0` 才做原生本地 prefix lookup；抢占恢复通常回到此分支。
- waiting 中 `num_computed_tokens > 0` 分支用于远程 KV 接收后的恢复，不重新 lookup。继续 chunked prefill 属于 running，不能借 waiting 排序改写。
- 当前 `KVCacheManager.get_computed_blocks` 会调用旧 `on_lookup`，且可能发 KV 事件，**不能直接用于只读排序打分**。应复用底层原生命中逻辑、`num_tokens - 1` 上限和 skip-reading 条件。
- 删除缓存可降低 H，缓存插入/替换可提高 H。依赖失效机制需要覆盖已命中链和第一个 miss，仅处理删除或重查堆顶不足以确保排序一致。

## 旧策略与共享元数据

旧实现改动三个原生文件共 37 行 hook，另有两个 policy/runtime 文件。`schedule` 起点的 `maybe_check` 运行旧周期比例驱逐；lookup 更新旧 epoch 历史；旧 adapter 先失效再调整 free queue。这些机制不能直接充当新的 demand-time 事务。

旧源码完整保留，尚未修改开关或接线。新 off/off 必须验证原生行为，不能用旧 native 观测模式替代无新元数据基线。

旧请求标记可提供 session 身份，实际命中/缓存挂接提供物理 owner 证据；排序 lookup 不能增加实际 owner。同一 session 的多轮和分支需要明确维护。未知 owner、过期代次、别名和无法证明的共享关系必须保护。

## 待确认（已发出问题，尚未收到答案）

1. 按块计时、成功采用命中刷新、新有效缓存开始计时、单调时钟、`>=1s` 冷边界、一次分配固定时刻。
2. 当前完整保留块计长度（共享计各 owner、活跃计长度但保护）；共享块仅在所有 owner 的安全尾部可复用，或全部保护共享块。
3. 同长逐物理块稳定轮转与跨 allocation 游标；部分最长并列稳定选主；等 H 按稳定入队 FCFS。
4. 无有效 hash 空闲块优先；首版同步、单组 FullAttention、hash/物理块大小一致；未覆盖配置启用即拒绝。

以上是推荐方案，**不是已确认策略**。依据规格第二节，不在用户确认前把这些选择写成最终实现。

## 验证、数据与范围

- 独立副本现有 `tests` 和 `experiment/test_replay.py` 实测：**59 passed，2 subtests passed，0.75s**。输出与 JUnit XML 在 `audit/baseline-tests.*`。
- allocator 测试从真实源码提取定义，替换外围重依赖；不是完整 EngineCore/GPU 验证。
- 新参考模型、优化实现、四开关组合、随机状态机及 CPU 微基准尚未实施/运行；不能沿用旧测试数字冒充。
- 检查原项目 results、同父目录实验数据和其中 `kv-preexperiment.tar.gz` 的文件名列表，未找到 `manifest_10.json` / `stratified10` 子集。未据 ID 重建输入；结论只覆盖上述位置。
- GPU 实验按用户要求排除，不需要模型权重。未连接服务器、部署或推送。
- 服务器安装版本、实际 import 路径和核心文件 SHA 尚待未来部署前只读核对；本地版本证据不能替代服务器证据。

## 下一步与回退

收到语义答案后，依次实现确定性参考模型、增量数据结构、原生事务绑定、waiting 一致性、功能测试与 CPU 对照。独立副本可直接放弃，原版和 progressive 保持起始内容，无须 reset 原仓库。
