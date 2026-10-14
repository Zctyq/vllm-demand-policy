"""Execute the unchanged schedule method with CPU data/config/output stubs.

All token-budget, running/waiting and allocation decisions are real source.
Worker payload serialization, encoder work and GPU execution are not modeled.
"""

import ast
import copy
import time
from collections import deque
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace

from native_loader import WORKTREE, make_cache_manager


class Queue(deque):
    add_request = deque.append
    prepend_request = deque.appendleft
    pop_request = deque.popleft

    def peek_request(self):
        return self[0]

    def prepend_requests(self, requests):
        self.extendleft(reversed(list(requests)))

    remove_request = deque.remove

    def remove_requests(self, requests):
        for request in requests:
            if request in self:
                self.remove(request)


class Output(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(has_structured_output_requests=False, **kwargs)


class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def scheduler(core, pool, *, source=WORKTREE, budget=128, max_running=32):
    class PauseState(Enum):
        UNPAUSED = 0
        PAUSED_ALL = 1

    class SchedulingPolicy(Enum):
        FCFS = 0
        PRIORITY = 1

    names = {
        "schedule",
        "_get_local_prefix_cache_hit",
        "_select_waiting_queue_for_scheduling",
        "_reserve_prefill_lookahead",
        "_is_blocked_waiting_status",
        "_update_after_schedule",
        "_get_new_block_ids_to_zero",
    }
    path = source / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text())
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    body = [copy.deepcopy(n) for n in original.body if getattr(n, "name", "") in names]
    cls = ast.ClassDef(
        name="CPUScheduler", bases=[], keywords=[], body=body, decorator_list=[]
    )
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        cls,
    ]
    ast.fix_missing_locations(tree)
    namespace = {
        "time": time,
        "record_function_or_nullcontext": lambda _: nullcontext(),
        "PauseState": PauseState,
        "SchedulingPolicy": SchedulingPolicy,
        "RequestStatus": core.RequestStatus,
        "create_request_queue": lambda _: Queue(),
        "SchedulerOutput": Output,
        "NewRequestData": SimpleNamespace(
            from_request=lambda r, *a, **k: SimpleNamespace(req_id=r.request_id)
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    s = namespace["CPUScheduler"]()
    s.progressive_runtime = s.demand_runtime = None
    s.kv_cache_manager = make_cache_manager(core, pool)
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
    )
    s.vllm_config = SimpleNamespace(speculative_config=None)
    s.kv_cache_config = SimpleNamespace(kv_cache_groups=[0])
    s.current_step = 0
    s.max_num_scheduled_tokens = budget
    s.max_num_running_reqs = max_running
    s.max_model_len = 32768
    s.block_size = 16
    s.max_num_encoder_input_tokens = 0
    s._pause_state = PauseState.UNPAUSED
    s.policy = SchedulingPolicy.FCFS
    s.waiting, s.skipped_waiting = Queue(), Queue()
    s.running, s.requests = [], {}
    s._inflight_prefills = set()
    s.finished_req_ids = set()
    s.reset_preempted_req_ids = set()
    s.prev_step_scheduled_req_ids = set()
    for key in (
        "prefill_capacity_bound",
        "log_stats",
        "requires_kv_delivery",
        "need_mamba_block_aligned_split",
        "is_encoder_decoder",
        "scheduler_reserve_full_isl",
        "needs_kv_cache_zeroing",
        "use_v2_model_runner",
        "model_uses_mrope",
        "model_uses_xdrope",
        "defer_block_free",
        "enable_return_routed_experts",
    ):
        setattr(s, key, False)
    for key in (
        "num_waiting_for_streaming_input",
        "num_spec_tokens",
        "num_prefill_lookahead",
        "num_lookahead_tokens",
        "sched_step_seq",
    ):
        setattr(s, key, 0)
    s.num_sampled_tokens_per_step = 1
    s.lora_config = s.connector = s.ec_connector = s.dynamic_sd_lookup = None
    s.connector_prefix_cache_stats = None
    s.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=list, get_manager_metadata=lambda: None
    )
    s._make_cached_request_data = lambda *a: SimpleNamespace(
        req_ids=[], new_block_ids=[]
    )
    return s


def enqueue(s, request):
    r = Request(**vars(request))
    r.num_output_placeholders = r.num_stale_output_tokens = 0
    r.drop_stale_output = False
    r.lora_request = r.prefill_stats = None
    r.num_preemptions = 0
    r.has_encoder_inputs = r.use_structured_output = False
    r.max_tokens = 128
    r.num_tokens_with_spec = r.num_tokens
    r.next_decode_eligible_step = 0
    r.is_prefill_chunk = True
    r.spec_token_ids = []
    s.waiting.add_request(r)
    s.requests[r.request_id] = r
    return r
