# 本地源码审计：vLLM 0.29.0

审计对象是用户提供的 `/Users/zhangtianqi/Downloads/vllm-0.29.0`。这是源码目录，不是已证明正在某 GPU 服务中加载的安装包；本轮不连接远程服务器。

此前取得的官方 v0.29.0 tag 指向 `98dff2a81d747d1dba01a47f939f48c3526d4206`。以下八个本地文件与该 tag 逐字节相同：scheduler、KVCacheManager、BlockPool、kv_cache_utils、Request、InputProcessor、chat serving、base serving。用户目录没有 `.git`。实现仅写入独立 worktree；原始源码保留。

## 已核对的调用链（行号为修改前的用户源码）

| 位置 | 已核对行为 | 集成要求 |
| --- | --- | --- |
| `vllm/v1/engine/core.py:597` | 同步 `step()`：schedule → execute_model → future.result → update_from_output | 安全检查必须在下一次 schedule 起点 |
| `vllm/v1/engine/core.py:210`、`:235` | max_concurrent_batches > 1 会使用 batch_queue 路径 | 原型同时拒绝 async_scheduling=True 和 max_concurrent_batches != 1 |
| `vllm/v1/core/sched/scheduler.py:501` | `schedule(self, throttle_prefills=False)`，随后执行所有 lookup 与 allocation | 只在此函数起点消费 pending 检查 |
| `scheduler.py:838` | 首次或抢占后 num_computed_tokens == 0 时查找前缀 | 逻辑访问按唯一请求去重，不能按 lookup 次数累计 |
| `scheduler.py:1077`、`:1092`、`:1161` | allocate_slots 可失败；成功后请求进入 running | 实际 K 绑定首次成功 admission，再由实际 GPU 返回确认执行 |
| `scheduler.py:1461` | `_update_after_schedule` 在执行 GPU 前增加 computed 和 in_flight token 数 | computed 数本身不是执行完成证据 |
| `scheduler.py:1871`、`:1915` | GPU 返回后扣除 in_flight tokens，并处理 speculative rollback | H 的历史登记只能取处理输出后的 computed - in_flight 完整块边界 |
| `scheduler.py:2053`、`:2061` | `stopped` 后 `_handle_stopped_request`；真正 finished 才 `_free_request` | 仅正常终止状态计完成；忽略取消、错误、抢占和继续 streaming |
| `scheduler.py:2480`、`:2522` | `_free_request` 可延迟释放；`_free_request_blocks` 处理 GPU 在途 fence | 不把响应完成等同物理引用释放；实验拒绝 connectors/多在途批次 |
| `vllm/v1/core/kv_cache_manager.py:228` | `get_computed_blocks` 返回 `(KVCacheBlocks, num_new_computed_tokens, shared_prefix_boundary)` 三项 | 保留三项返回和原查找语义 |
| `kv_cache_manager.py:257` | 最长命中上限 `request.num_tokens - 1` | 机制 fixture 多加探测 token，H 同样受此上限限制 |
| `kv_cache_manager.py:554` | `allocate_slots` 在 GPU 执行前调用 `coordinator.cache_blocks` | `cache_full_blocks` 只能登记暂存身份，不能直接宣称已经计算并登记 H |
| `vllm/v1/core/block_pool.py:225` | `cache_full_blocks` 解析 hash_block_size 与 group 的完整块身份 | 只支持 full-attention 单组、hash_block_size 等于实际块大小的原型 |
| `block_pool.py:56`、`:88` | hash 映射值可为一个 KVCacheBlock 或 `{block_id: block}` | 全部副本安全才允许逻辑失效，逻辑计数与物理计数分开 |
| `block_pool.py:185`、`:443`、`:606` | `cached_block_hashes_by_block` 保存附加别名；partial cache 会产生别名 | 无法证明全部别名安全就拒绝；不得只看 primary hash |
| `block_pool.py:571` | `_remove_cached_block_hashes` 删除主 hash 和全部别名并 reset_hash | 适合通知 resident 映射失效；必须区分原生覆盖、策略失效、其他清理 |
| `block_pool.py:646` | `get_new_blocks` 从 free queue 弹出，原生失效旧 hash 后 ref_cnt += 1 | 保留原生容量和引用语义；物理 generation 按分配递增 |
| `block_pool.py:749` | `evict_blocks` 对指定 block 调用 `_maybe_evict_cached_block`，不会拒绝 ref_cnt > 0 | 调用者必须预先验证所有副本、free queue 和完整身份 |
| `vllm/v1/core/kv_cache_utils.py:351`、`:394` | `remove` 减 free count；`prepend_n` 加 count | 失效空闲块后逐个 remove、统一 prepend_n，前后 free count 相同 |

## 请求标识通道

`vllm/entrypoints/serve/engine/serving.py:117` 的 `_base_request_id` 确实读取 `X-Request-Id`；chat serving (`:281`) 在前方增加 `chatcmpl-`；`InputProcessor.assign_request_id` (`input_processor.py:262`) 默认再追加 `-<8随机字符>`。最终 `Request.from_engine_core_request` (`request.py:237`) 保留内部 ID。回放标识应包含可严格解析的定界标记，并容许官方添加的前后缀；不可假定任意 HTTP header 都能透传，也无需关闭 ID 随机化。

该版本还提供 `ChatCompletionRequest.session_id`（protocol.py:397），经 serving → AsyncLLM → EngineCoreRequest → Request 传递。初版可以沿用明确验证的 ID 标记，不需要往 prompt 或 cache_salt 填 session。

以上是源码调用链验证，尚不是 HTTP → GPU EngineCore 的运行验证。GPU smoke 必须核对 EngineCore 日志里的实际 import 路径和解析后的 request/session/call 标识。

## 额外边界

- ignore_eos 是真实的 chat protocol 字段；但 `to_sampling_params` 会合并服务端默认 stop_token_ids (`protocol.py:689`)。要固定输出长度，还应使用 min_tokens=max_tokens 并检查实际长度。
- 原生回收本来就是 block 级，free queue 中可分配但仍带 hash 的块被提前失效不增加 free count，也不使预分配 KV 显存下降。
- 禁止把 cache_full_blocks 的乐观登记当成 GPU 已执行。应在处理成功输出时登记实际完整块；失败/取消的未执行内容不能进入历史分母。
- 原型需要明确拒绝当前未覆盖的 attention/connector/并行配置，不能静默退化成假设。

## 实现中的逻辑树、分段和生命周期

`progressive_policy.py` 的每个节点对应唯一 `(prefix hash, group)` 完整块位置，保存 parent/children、已观察 owner 集合、访问 epoch、出生 epoch、cold_streak，以及 resident 物理槽位和 generation。后续多轮请求遇到相同链就沿用节点，不按请求复制整条历史。

相邻节点仅在同一 owner 集合、同一冷却级别、连续有效、没有其他仍驻留分支依赖时组成临时私有尾段。部分访问更新前部节点的访问 epoch，下一检查时冷热边界自然将前部与尾部分开；无需复制节点。公共前缀和未知 owner 均被保护。部分淘汰后保留 ghost 节点与历史 owner 信息，后来的 miss 同样会重置已知内容热度；出生所在窗口不计冷。

物理槽位每次原生分配 generation 增加。原生移除 hash 的回调同步撤销旧 resident 关联，按 reason 区分 native 覆盖与 policy 失效。检查前重新读取实际 hash→全部副本，身份失配更新 resident；任何副本活跃时保护整逻辑块。带附加 hash 别名的物理块保守跳过。失效调用原生 `evict_blocks` 后使用队列 `remove` / `prepend_n`；从不再次 `free_blocks`，前后 free count 必须相等。

完整历史只在 GPU 返回后的 `on_computed` 登记；`cache_full_blocks` 的乐观 hash 挂接不进入 H。首个 lookup 记录访问及历史 H；首次成功 admission 记录拟采用命中，首次实际计算输出返回后才确认 K。抢占恢复重新登记新物理 generation，但不会重复首次访问、H、K 或完成数。正常完成只累计一次；安全边界合并跨过的阈值，只执行一次冷却检查。

`on_finish` / checkpoint / reset 输出日志，decode step 不写磁盘。原生模式保留相同必要观察但不调用主动失效。没有设置任何实验环境变量时不创建 runtime，不改变原生调度/回收语义。当前实验 runtime 对两组应用相同支持边界：单 GPU、单组 causal full attention、hash size=block size、同步调度且 max_concurrent_batches=1、无 transfer/offload/speculative、prefix cache开启、enforce_eager开启、显式非 FlashInfer attention backend。

## Completion token-ID 烟测通道补充

`vllm/entrypoints/openai/completion/serving.py:156` 先增加 `cmpl-`，`:180` 对每个 prompt 追加 `-0` 等索引；随后 InputProcessor 再追加随机后缀。因此真实内部标识形如 `cmpl-pe1.<base64url-json>-0-<8hex>`。runtime 的 marker 解析同时覆盖该形式和 chat 通道。CompletionResponse 的 `prompt_token_ids` 在 `choices[0]` 内（不是 chat 的顶层字段），GPU 烟测按此核对实际输入 IDs。

`tools/gpu_smoke.py --self-test` 只验证 fixture 算术和标识解析，不发 HTTP、不载入 tokenizer/model，也不代表 GPU 机制通过。`--run` 是留给用户手动在实验实例执行的真实 smoke：两个独立 reset fixture 分别验证 80 与 48，避免重访恢复尾部后继续期待 48。脚本使用用户本地 tokenizer 的合法 token IDs，100*block_size+1 输入保证 logits 重算边界最多命中 100 个完整块。每个 filler 只有一个输入 token、生成一个 token，实际计算输入小于块大小，不创建可缓存完整块。正式回放前仍需真实运行该脚本；本机未执行任何服务器请求。

## 2026-10-03审查修订：独立的sampler开关

本地原版 `vllm/envs.py` 的 `VLLM_USE_FLASHINFER_SAMPLER` 未设置时默认True。`vllm/v1/sample/ops/topk_topp_sampler.py` 的 `flashinfer_sampler_supported()` 在CUDA sampler初始化时独立读取它；非FlashInfer attention后端不足以关闭sampler。

启动器对子进程明确设置该变量为0；runtime同时核验原始环境值和vllm.envs解析结果，将use_flashinfer_sampler=False记录到EngineCore配置，客户端配置签名也包含此字段。CPU已验证父1→子0和直接runtime错误设置的拒绝行为。

真实GPU烟测还要读取本次服务日志，找到该原生函数的sampler-disabled初始化信息，保存读取时的日志SHA及匹配行。这部分worker证据尚未在本机GPU执行，不能由EngineCore的配置记录单独替代；也不代表temperature=0时执行了随机采样。
