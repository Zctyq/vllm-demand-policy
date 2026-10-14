# 机制、确认语义与验收

## 已确认语义

2026-10-12 用户确认四组问题，并强调共享块必须满足**所有受影响会话的尾部条件**，仅所有 owner 空闲不足以回收。

- 按物理块计时。成功分配并采用命中块才刷新；排序探测不刷新。新块首次进入原生完整缓存索引开始计时。
- 单调时钟；`elapsed >= 1s` 为冷；每次需要 victim 的分配只读取一次决策时刻。
- 按当前保留完整物理 KV 块计会话长度；共享块计入每个 owner，活动块计长度但保护；未形成完整缓存的块不计长度。
- 同长逐个物理块 round-robin，稳定 owner 顺序、跨 allocation 保留游标。部分最长并列稳定选主并做完整阶段。等 H 以稳定入队序号 FCFS；没有 aging。
- 优先无有效 hash 空闲块。首版启用时要求同步、单组 causal FullAttention、hash/物理 block size 相同、无 connector/offload/speculation/context parallelism。稀疏窗口、部分缓存别名等不支持配置明确拒绝。

## 函数与状态对应

| 语义 | 实现位置 |
| --- | --- |
| 慢确定性参考 | `demand_policy.ReferencePolicy.plan`：复制链、全会话枚举，独立规划循环 |
| 增量选择 | `DemandPolicy` 的 blocks/chains/members、两个 `LengthIndex`、到期 heap |
| 真实复用与冷度 | `DemandRuntime.attach(reused=True)` → `DemandPolicy.reuse`；`advance` 在固定时刻推进到期 |
| 阶段端点 | `DemandPolicy.plan` 固定 owner/target；到目标、安全边界或 D 停止 |
| 同长轮转 | `plan` 保存本轮原长度和游标；逐物理 ID 等价执行，不混成每块全局追平 |
| 多 owner 安全 | `reason` 检查全部 owner 当前尾部、active、known/full；`commit/remove` 更新全部长度 |
| 无压力快路径 | `reserve` 在 U≥N 时直接取 clean 索引，不推进到期堆或建立候选集合 |
| 原生需求 | `KVCacheManager.allocate_slots` 在原生容量检查之后扣除即将 touch 的空闲命中块；保留 watermark/full-fit/reserved/lookahead 检查 |
| 事务 | `reserve` 规划、验证，再 `_commit` 移除确切 free-queue 节点、解除对应 hash、设置 refcount；`get_new_blocks` 消费同一批预留块 |
| 回滚 | `_commit` 记录局部队列邻接、原 hash/覆盖 token 数与重复副本字典顺序；物理阶段失败恢复并返回延期 |
| 只读命中 | `score` 调用真实 coordinator 的最长合法命中；遵守 skip-reading 和 `num_tokens-1` 上限 |
| 等待一致性 | `HitQueue` 的 hash→请求依赖，含已命中链和第一个 miss；插入/移除都标脏，下一选择前重算受影响键 |
| 避免重复查找 | 只读 probe 暂存至 `get_computed_blocks`，hash 变化撤销；成功分配始终 `pop_selected` 原请求 |
| 原生运行约束 | 只改 waiting 队列选择/成功弹出；running、预算、chunked prefill 等执行原始 schedule 代码 |
| 关闭行为 | Scheduler 不创建旧 runtime；新两开关关闭时 create_runtime 返回 None，pool/manager 走原生分支 |

原生 free queue 里 cached/refcount=0 块本来就可分配，hash 失效不增加 free_count。这里实际移除 N 个物理节点、正好丢弃 D 个 cached 物理块；没有先清一批再原生覆盖另一批。

规划可临时缩短逻辑链以维护候选，结束时撤销，未成功提交不会推进 RR 游标。成功提交只清理真实选中的物理块，多个 owner 共享一个物理块只计一次。冷层遇到暖尾、活动/未知尾或共享中间边界不越过它；暖层处理暴露的新冷候选在下一阶段优先考虑。

## 共享、分支与错误范围

会话身份来自原生 `session_id`，或旧回放的严格 `pe1` marker 解析；不从 prompt 猜会话。实际采用的命中和新缓存挂接登记 owner。未知身份保持保护，`diagnostics()` 的 `unknown_cached_blocks` 给出当前物理数量，另有累计保护检查计数，两者单位不同。

一个 session 的多个已观察分支按首次附着的完整物理块顺序保留；只能从该顺序尾部回收，因此可能比逐分支叶子策略更保守。没有通过跳过不可回收尾部来增加供给。外部中间失效后，无法证明连续关系的剩余链标记为未知。非预期 alias 保留关系并保护，不降级到 native victim。

代次用递增版本防止旧 expiry 事件命中新槽位；提交前校验真实 refcount、free 链连接、hash/alias、原生 hash-map 成员资格、版本及 owner 集合。预留采用同步单线程事件顺序，无新增全局锁。

物理提交成功后的可观测性通知错误以 `notification_errors` 计数，不把成功分配报成失败而触发重试/额外缓存损失。用户注入的物理提交失败测试发生在提交阶段，验证完整回滚。不可恢复的 Python 进程故障（如 OOM/进程终止）不属于可回滚保证。

## 复杂度与有界状态

定义 W 为等待请求数，S 为已观察会话数，B 为缓存物理块数，E 为新到期事件数，D 为本次物理 victims，O 为受影响 owner 引用数，P 为各请求实际遍历的 prefix hash 数，R 为本次 hash 变化影响的等待请求集合大小。

- 初始化 clean 索引 O(物理池容量)，只在 runtime 创建/reset 做一次。U 足够时分配 O(N)，没有全池扫描。
- 到期推进 O(E log B)，批量刷新受影响 owners；没有每 allocation 扫描全池。懒 heap 超过 `2*当前缓存块数+64` 时在复用事件重建，单次 O(B)，报告单列实测清理耗时。连续删除后可能暂留旧 heap 项，上限受之前池容量限制。
- victim 规划和提交必须处理 O(D+O) 个物理/owner 关系。长度索引采用有序 Python 列表和桶，查找长度档位无需全会话排序，但插入/移除有 O(S) 移位最坏成本；不能声称整体 O(log S) 或常数复杂度。
- 首次 waiting 建键 O(P + W log W)。失效标记成本为变更 hash 的 watcher 引用数；下一选择前只重算 R 个键，O(P_R + R log R + R log W)。没有每物理块重查/重排整个 waiting。
- 每请求保存命中链和首个 miss 的依赖，空间 O(P)。队列 lazy heap 超过 `2*live+64` 时重建；版本用队列全局序号避免 request-ID 重入的 ABA。过期 heap、lazy queue 不随事件数无限积累。
- owner 名称/稳定序号保留到 reset，空间随**已观察会话 S** 增长，不只随当前 resident session 数增长。这是跨分配稳定顺序的状态成本；长期服务应依据实际 S 评估，未宣称内存只由 B 决定。
- 计数器键集合有界、没有热路径磁盘写入/完整 prompt 日志。等待记录 max_wait、oldest_pending 和超过 1 秒的观察次数，报告可能长期等待，不改变优先级。`diagnostics()`/`metadata_bytes()` 为显式离线扫描，不能在每次 allocation 上调用。

## CPU 验收矩阵

完整输出 `results/demand/cpu-tests.txt`、JUnit XML：**110 passed, 2 subtests passed**。本次新增 51 项，旧基线 59 项。

| 要求 | 实际验证 |
| --- | --- |
| 绝对 H、0、不同输入长度、并列与 FCFS | `test_waiting_absolute_hits_ties_and_same_iteration_invalidation` 等 |
| sorting-only 不改变缓存/ref/free/计时 | 四开关真实只读 scoring；manager 连续 lookup 前后 snapshot |
| 同次调度 hash 删除/插入影响 H | 首 miss 依赖、真实 pool 失效、选中请求被自身插入改变排名的回归 |
| 100/80/50、D=12 | 精确阶段端点/physical IDs/保留长度断言 |
| 冷优先、同长仍冷、暖层与边界 | 混合冷暖、共享中间尾、保护边界测试 |
| 1 秒和固定时刻、新槽位 | 阈值三点、单次 clock 读取、复用槽位旧 deadline 不生效 |
| RR 游标和参考等价 | 非零游标顺序；seed 1701 的 120 组随机形态×4 次分配规划 |
| shared/active/unknown/partial/stale | 模型和真实 BlockPool；共享物理一次回收、未知数量、别名保护 |
| cached-free hash 清除不增 free_count | 原有真实 native-core 测试及新精确物理分配测试 |
| N/U/D、计划与实际绑定 | 真实 12 个 victim IDs、exact physical loss、clean fast path |
| 0、供给不足、验证/提交失败 | zero、excluded tail、容量边界、owner/hash 失配、3 个提交失败位置、重复 hash 顺序恢复 |
| native 约束与四开关 | 实际 schedule 的 running-first/chunking/decode、preempted resume、预算/并发、只读查找复用；off/off 对原版逐状态比较 |
| 随机生命周期 | seed 311，500 次真实 allocate/cache/hit/touch/free 交错，持续检查链表/ref/hash/owner/长度 |
| 不擅自防饥饿 | 连续 100 次低命中到达，高命中仍等待且产生观测计数 |
| CPU 性能 | 60 案例×60 稳态样本，wall+CPU p50/p95/p99，见独立微基准报告 |

已修复测试中发现的候选恢复、已预留块与原生 free-count 检查、队列重排导致弹出错误请求、槽位旧计时事件、重复 hash 回滚顺序问题。最终套件无失败。

## 未运行和部署边界

- 未运行完整 vLLM 导入/EngineCore、HTTP 服务、worker KV 读写、模型输出一致性、GPU/JCT/TTFT/吞吐实验；GPU 已由用户明确排除。
- 本地检查范围内未发现指定 stratified10 manifest/selection，不重新抽样，不把合成 CPU 输入称为 237 次真实回放。
- 未核验服务器实际运行环境。只读兼容性脚本检查解析路径、版本元数据与源码 SHA；它不等于部署/启动验证。
- 原版、原 progressive、原环境保持不变；旧实现源码保留。本交付 patch 只对记录的精确源码适用。两补丁正反应用均逐字节通过，结果见 `patch-validation.json`。
