"""CPU contracts: safe exact physical reuse, ordering and deterministic stages.

Inputs are real source BlockPool lifecycles and synthetic owner chains. These
tests catch cache loss/refcount/selection errors without model execution.
"""

import copy
import random
import sys
from types import SimpleNamespace

import pytest
from native_loader import (
    WORKTREE,
    assert_queue_integrity,
    load_cache_manager_core,
    load_native_core,
    load_source_module,
    make_cache_manager,
)
from test_native_core import chain_hashes


@pytest.fixture
def modules(monkeypatch):
    root = WORKTREE / "vllm/v1/core"
    policy = load_source_module(root / "demand_policy.py", "vllm.v1.core.demand_policy")
    queue = load_source_module(root / "demand_queue.py", "vllm.v1.core.demand_queue")
    load_source_module(root / "demand_metadata.py", "vllm.v1.core.demand_metadata")
    runtime = load_source_module(
        root / "demand_runtime.py", "vllm.v1.core.demand_runtime"
    )
    return policy, queue, runtime


def model(policy, lengths, *, cls="DemandPolicy", now=0):
    result = getattr(policy, cls)()
    bid = 1
    for owner, length in enumerate(lengths):
        ids = list(range(bid, bid + length))
        for i in ids:
            result.cache(i, now)
        result.attach(str(owner), ids)
        bid += length
    return result


def fixture_pool(modules, lengths, spare=0):
    _, _, runtime = modules
    core = load_native_core()
    pool = core.BlockPool(sum(lengths) + spare + 1, True, 16)
    clock = [0.0]
    rt = runtime.DemandRuntime(pool, eviction=True, clock=lambda: clock[0])
    pool.demand_runtime = rt
    cached = []
    for owner, length in enumerate(lengths):
        blocks = pool.get_new_blocks(length)
        req = SimpleNamespace(
            request_id=str(owner),
            session_id=str(owner),
            block_hashes=chain_hashes(length, str(owner)),
        )
        pool.cache_full_blocks(req, blocks, 0, length, 16, 0)
        pool.free_blocks(reversed(blocks))
        cached.append(blocks)
    return core, pool, rt, clock, cached


def test_stage_endpoints_and_early_stop(modules):
    policy, _, _ = modules
    m = model(policy, [100, 80, 50])
    plan = m.plan(12, 1)
    assert plan.ids == list(range(100, 88, -1))
    assert plan.stages == [("cold", 0, 100, 88)]
    assert list(map(len, m.chains.values())) == [100, 80, 50]
    full = m.plan(80, 1)
    assert full.stages == [
        ("cold", 0, 100, 80),
        ("cold", 0, 80, 50),
        ("cold", 1, 80, 50),
    ]
    m.commit(plan)
    assert list(map(len, m.chains.values())) == [88, 80, 50]


def test_cold_equal_keeps_round_robin_and_cursor(modules):
    m = model(modules[0], [2, 2, 2])
    p = m.plan(1, 1)
    assert p.ids == [2]
    assert m.cursor["cold"] == -1
    m.commit(p)
    # Unequal remaining lengths form stages before equality is restored.
    p = m.plan(2, 1)
    assert p.ids == [4, 6]
    m.commit(p)
    assert m.plan(2, 1).ids == [3, 5]


@pytest.mark.parametrize("now,cold", [(0.999999, False), (1.0, True), (1.000001, True)])
def test_one_second_boundary_and_new_cache(modules, now, cold):
    m = model(modules[0], [2])
    p = m.plan(1, now)
    assert p.stages[0][0] == ("cold" if cold else "warm")
    assert m.blocks[2].used == 0


def test_cold_exhaustion_then_warm_without_skipping_boundaries(modules):
    m = model(modules[0], [2, 2])
    m.reuse(4, 0.5)
    p = m.plan(4, 1)
    assert p.ids == [2, 1, 4, 3]
    assert [s[0] for s in p.stages] == ["cold", "cold", "warm", "cold"]


def test_shared_interior_remains_until_all_owner_tails(modules):
    m = model(modules[0], [2])
    m.cache(3, 0)
    m.attach("other", [1, 2, 3])
    m.set_active(3, 1)
    assert m.plan(1, 1) is None
    assert m.stats["protected_shared_interior"] > 0
    m.set_active(3, 0)
    p = m.plan(3, 1)
    assert p.ids == [3, 2, 1]
    m.commit(p)
    assert all(not c for c in m.chains.values())


@pytest.mark.parametrize("reason", ["unknown", "active", "partial"])
def test_protected_tail_never_skips_to_prefix(modules, reason):
    m = model(modules[0], [3])
    b = m.blocks[3]
    if reason == "unknown":
        b.known = False
    elif reason == "active":
        b.active = 1
    else:
        b.full = False
    m.refresh(b.owners)
    assert m.plan(1, 1) is None


def test_real_allocator_uses_exact_planned_ids_and_no_extra_hash_loss(modules):
    _core, pool, rt, clock, cached = fixture_pool(modules, [100, 80, 50])
    clock[0] = 1
    expected = [b.block_id for b in reversed(cached[0][-12:])]
    actual = pool.get_new_blocks(12)
    assert [b.block_id for b in actual] == expected
    assert sum(b.block_hash is not None for b in pool.blocks) == 218
    assert rt.stats["planned_victims"] == rt.stats["actual_victims"] == 12
    assert pool.get_num_free_blocks() == 218
    assert_queue_integrity(pool)


def test_clean_fast_path_does_not_advance_expiration_or_select(modules):
    _, pool, rt, clock, _ = fixture_pool(modules, [3], spare=5)
    clock[0] = 10
    before = dict(rt.policy.stats)
    pool.get_new_blocks(2)
    assert dict(rt.policy.stats) == before
    assert sum(b.block_hash is not None for b in pool.blocks) == 3


def snapshot(pool):
    return (
        assert_queue_integrity(pool),
        [(b.ref_cnt, b.block_hash, b.block_hash_num_tokens) for b in pool.blocks],
        {
            k: tuple(v) if isinstance(v, dict) else v.block_id
            for k, v in pool.cached_block_hash_to_block._cache.items()
        },
    )


@pytest.mark.parametrize("failed_at", [1, 2, 3])
def test_commit_failure_restores_native_hashes_refs_and_queue(modules, failed_at):
    _, pool, rt, _clock, _ = fixture_pool(modules, [4])
    before = snapshot(pool)
    rt.fail_after = failed_at
    assert rt.reserve(3) is False
    assert snapshot(pool) == before
    assert rt.policy.chains[0] == [1, 2, 3, 4]
    assert rt.stats["rollback"] == 1


def test_insufficient_eligible_supply_has_no_cache_loss(modules):
    _, pool, rt, _, cached = fixture_pool(modules, [4])
    before = snapshot(pool)
    assert rt.reserve(2, excluded=[cached[0][-1].block_id]) is False
    assert snapshot(pool) == before
    assert rt.policy.blocks[4].active == 0


def test_reference_equivalence_random_active_unknown_shared_and_reuse(modules):
    policy = modules[0]
    rng = random.Random(1701)
    for _ in range(120):
        lengths = [rng.randrange(1, 14) for _ in range(rng.randrange(1, 12))]
        m = model(policy, lengths)
        for bid, b in m.blocks.items():
            b.active = int(rng.random() < 0.08)
            b.known = rng.random() > 0.04
            m.reuse(bid, rng.choice([0, 0.2, 0.9]))
        if len(lengths) > 1:
            m.attach("shared", m.chains[0][: rng.randrange(1, lengths[0] + 1)])
        m.refresh(m.chains)
        ref = copy.deepcopy(m)
        ref.__class__ = policy.ReferencePolicy
        for _ in range(4):
            count = rng.randrange(0, 12)
            p, q = m.plan(count, 1), ref.plan(count, 1)
            assert p == q
            if p:
                m.commit(p)
                ref.commit(q)
            assert m.chains == ref.chains


def test_waiting_absolute_hits_ties_and_same_iteration_invalidation(modules):
    _, queue, _ = modules
    scores = {"A": (200, {"a"}), "B": (100, {"b"}), "C": (100, {"c"}), "Z": (0, {"z"})}
    runtime = SimpleNamespace(sequence=0, score=lambda r: scores[r.request_id])
    q = queue.HitQueue(runtime)
    for rid, tokens in [("A", 1000), ("B", 200), ("C", 500), ("Z", 70)]:
        q.add_request(SimpleNamespace(request_id=rid, num_tokens=tokens))
    assert q.pop_request().request_id == "Z"
    assert q.pop_request().request_id == "B"
    scores["A"] = (0, {"a"})
    q.invalidate("a")
    assert q.pop_request().request_id == "A"
    assert q.pop_request().request_id == "C"
    assert q.stats["lookups"] == 5


def test_score_increase_invalidates_first_miss_dependencies(modules):
    _, queue, _ = modules
    values = {"A": 0, "B": 16}
    rt = SimpleNamespace(
        sequence=0, score=lambda r: (values[r.request_id], {r.request_id})
    )
    q = queue.HitQueue(rt)
    for rid in values:
        q.add_request(SimpleNamespace(request_id=rid))
    assert q.peek_request().request_id == "A"
    values["A"] = 32
    q.invalidate("A")
    assert q.peek_request().request_id == "B"


def test_singleton_becomes_competing_queue_with_fresh_scores(modules):
    """Skipping a sole candidate's score must not leave stale future ordering."""
    values = {"A": 16, "B": 0, "C": 16}
    rt = SimpleNamespace(
        sequence=0, score=lambda r: (values[r.request_id], {r.request_id})
    )
    q = modules[1].HitQueue(rt)
    a, b, c = [SimpleNamespace(request_id=rid) for rid in values]
    q.add_request(a)
    assert q.peek_request() is a
    assert q.peek_key() == (16, a._demand_sequence)
    values["A"] = 0
    q.invalidate("A")
    q.add_request(b)
    assert q.pop_request() is a  # Equal H keeps the original FCFS sequence.
    assert q.peek_request() is b
    values["B"] = 32
    q.invalidate("B")
    q.add_request(c)
    assert q.pop_request() is c
    assert q.pop_request() is b


def request(core, rid, count, hashes=None):
    return SimpleNamespace(
        request_id=rid,
        session_id=rid,
        num_tokens=count,
        num_prompt_tokens=count,
        num_computed_tokens=0,
        num_in_flight_tokens=0,
        status=core.RequestStatus.WAITING,
        skip_reading_prefix_cache=False,
        block_hashes=hashes or chain_hashes(count // 16, rid),
        shared_prefix_boundary=0,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_cache_manager_hit_capacity_chunked_and_decode(modules, enabled):
    core = load_cache_manager_core()
    pool = core.BlockPool(10, True, 16)
    manager = make_cache_manager(core, pool)
    rt = modules[2].DemandRuntime(pool, eviction=True, clock=lambda: 0)
    if enabled:
        pool.demand_runtime = manager.demand_runtime = rt
    first = request(core, "A", 65)
    assert len(manager.allocate_slots(first, 32).blocks[0]) == 2
    first.num_computed_tokens = 32
    assert len(manager.allocate_slots(first, 33).blocks[0]) == 3
    manager.free(first)
    second = request(core, "B", 65, first.block_hashes)
    blocks, hits, _ = manager.get_computed_blocks(second)
    assert hits == 64
    before = snapshot(pool)
    assert manager.get_computed_blocks(second)[1] == 64
    assert snapshot(pool) == before
    fresh = manager.allocate_slots(
        second, 1, num_new_computed_tokens=hits, new_computed_blocks=blocks
    )
    assert len(fresh.blocks[0]) == 1
    second.num_computed_tokens = 65
    assert not manager.allocate_slots(second, 1).blocks[0]
    if enabled:
        assert rt.reserved is None
        assert rt.stats["D"] == 0
    manager.free(second)
    assert_queue_integrity(pool)


def test_true_manager_defers_without_touching_hits_if_tail_protected(modules):
    core = load_cache_manager_core()
    pool = core.BlockPool(6, True, 16)
    manager = make_cache_manager(core, pool)
    rt = modules[2].DemandRuntime(pool, eviction=True, clock=lambda: 0)
    pool.demand_runtime = manager.demand_runtime = rt
    first = request(core, "A", 80)
    manager.allocate_slots(first, 80)
    manager.free(first)
    # Unknown tail blocks recycling all earlier blocks of this chain.
    tail = rt.policy.chains[0][-1]
    rt.policy.blocks[tail].known = False
    rt.policy.refresh([0])
    second = request(core, "B", 32, first.block_hashes[:1] + chain_hashes(1, "x"))
    blocks, hits, _ = manager.get_computed_blocks(second)
    assert hits == 16
    before = snapshot(pool)
    assert (
        manager.allocate_slots(
            second, 16, num_new_computed_tokens=hits, new_computed_blocks=blocks
        )
        is None
    )
    assert snapshot(pool) == before
    assert "B" not in manager.coordinator.single_type_managers[0].req_to_blocks


def test_rr_rotates_within_a_round_from_nonzero_cursor(modules):
    m = model(modules[0], [2, 2, 2])
    m.cursor["cold"] = 0
    assert m.plan(3, 1).ids == [4, 6, 2]


def test_recycled_slot_old_deadline_cannot_age_new_cache(modules):
    m = model(modules[0], [1])
    m.commit(m.plan(1, 0))
    m.cache(1, 0.8)
    m.attach("0", [1])
    assert m.plan(1, 1).stages[0][0] == "warm"
    for i in range(500):
        m.reuse(1, 1 + i / 1000)
    assert len(m.deadlines) <= 2 * len(m.blocks) + 64


def test_stale_identity_and_unknown_quantity(modules):
    _, pool, rt, _, _ = fixture_pool(modules, [3])
    rt.policy.blocks[3].key = b"stale"
    before = snapshot(pool)
    assert rt.reserve(1) is False
    assert snapshot(pool) == before
    rt.policy.blocks[3].known = False
    assert rt.diagnostics()["unknown_cached_blocks"] == 1


def test_real_shared_block_is_recycled_once_with_all_owner_lengths(modules):
    _, pool, rt, _, cached = fixture_pool(modules, [3])
    rt.policy.attach("other", [b.block_id for b in cached[0]])
    allocated = pool.get_new_blocks(2)
    assert [b.block_id for b in allocated] == [3, 2]
    assert list(map(len, rt.policy.chains.values())) == [1, 1]
    assert rt.stats["actual_victims"] == 2
    assert_queue_integrity(pool)


def fake_scheduler(core, pool):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=False),
            max_concurrent_batches=1,
            kv_transfer_config=None,
            ec_transfer_config=None,
            speculative_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=True),
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=core.FullAttentionSpec())]
        ),
        kv_cache_manager=make_cache_manager(core, pool),
        hash_block_size=16,
        is_encoder_decoder=False,
        is_mm_encoder_only=False,
        dcp_world_size=1,
        pcp_world_size=1,
    )


@pytest.mark.parametrize("waiting,eviction", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_four_switch_combinations_and_real_readonly_scoring(
    modules, monkeypatch, waiting, eviction
):
    core = load_cache_manager_core()
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", core)
    monkeypatch.setenv("VLLM_WAITING_HIT_ASC", str(waiting))
    monkeypatch.setenv("VLLM_DEMAND_EVICTION", str(eviction))
    pool = core.BlockPool(20, True, 16)
    scheduler = fake_scheduler(core, pool)
    rt = modules[2].create_runtime(scheduler)
    assert (rt is None) == (not waiting and not eviction)
    if rt is None:
        return
    assert (rt.policy is not None) == bool(eviction)
    pool.demand_runtime = scheduler.kv_cache_manager.demand_runtime = rt
    first = request(core, "A", 65)
    scheduler.kv_cache_manager.allocate_slots(first, 65)
    scheduler.kv_cache_manager.free(first)
    if waiting:
        second = request(core, "B", 65, first.block_hashes)
        before = snapshot(pool)
        state = copy.deepcopy(rt.policy.blocks) if rt.policy else None
        scheduler.waiting.add_request(second)
        assert scheduler.waiting.peek_key()[0] == 64
        assert snapshot(pool) == before
        assert (rt.policy.blocks if rt.policy else None) == state
        pool.evict_blocks({1})
        assert scheduler.waiting.peek_key()[0] == 0


def test_native_budget_watermark_and_full_sequence_checks_unchanged(modules):
    core = load_cache_manager_core()
    pool = core.BlockPool(6, True, 16)
    manager = make_cache_manager(core, pool)
    rt = modules[2].DemandRuntime(pool, eviction=True, clock=lambda: 0)
    pool.demand_runtime = manager.demand_runtime = rt
    r = request(core, "A", 96)
    before = snapshot(pool)
    assert manager.allocate_slots(r, 16, full_sequence_must_fit=True) is None
    manager.watermark_blocks = 5
    assert manager.allocate_slots(r, 16, has_scheduled_reqs=True) is None
    manager.watermark_blocks = 0
    assert manager.allocate_slots(r, 16, reserved_blocks=5) is None
    assert snapshot(pool) == before
    assert not rt.stats


def test_real_allocator_random_lifecycle_invariants(modules):
    _core, pool, rt, clock, _ = fixture_pool(modules, [7, 6, 5], spare=10)
    rng = random.Random(311)
    held = []
    for tick in range(500):
        clock[0] = tick / 100
        roots = [b for b in pool.blocks if b.block_hash_num_tokens == 16]
        if roots and rng.random() < 0.2:
            block = rng.choice(roots)
            pool.touch([block])
            rt.attach(
                SimpleNamespace(request_id=f"hit{tick}", session_id=f"hit{tick}"),
                [block],
                reused=True,
            )
            held.append([block])
        elif held and rng.random() < 0.45:
            pool.free_blocks(held.pop(rng.randrange(len(held))))
        elif pool.get_num_free_blocks():
            count = min(rng.randrange(1, 4), pool.get_num_free_blocks())
            if rt.reserve(count):
                blocks = pool.get_new_blocks(count)
                if rng.random() < 0.7:
                    req = SimpleNamespace(
                        request_id=str(tick),
                        session_id=str(tick % 8),
                        block_hashes=chain_hashes(count, str(tick)),
                    )
                    pool.cache_full_blocks(req, blocks, 0, count, 16, 0)
                held.append(blocks)
        assert_queue_integrity(pool)
        for bid, b in rt.policy.blocks.items():
            assert pool.blocks[bid].ref_cnt == b.active
            assert pool.blocks[bid].block_hash == b.key
            for o in b.owners:
                assert bid in rt.policy.chains[o]
        for o, chain in rt.policy.chains.items():
            assert len(chain) == len(set(chain)) == len(rt.policy.members[o])
            assert all(o in rt.policy.blocks[bid].owners for bid in chain)


@pytest.mark.parametrize("waiting,eviction", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_actual_scheduler_running_first_chunked_budget_and_waiting_order(
    modules, monkeypatch, waiting, eviction
):
    from scheduler_cpu import enqueue, scheduler

    core = load_cache_manager_core()
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", core)
    pool = core.BlockPool(40, True, 16)
    s = scheduler(core, pool, budget=16, max_running=3)
    rt = modules[2].DemandRuntime(pool, eviction=bool(eviction), waiting=bool(waiting))
    rt.manager = s.kv_cache_manager
    if waiting or eviction:
        s.demand_runtime = pool.demand_runtime = s.kv_cache_manager.demand_runtime = rt
    if waiting:
        s.waiting, s.skipped_waiting = modules[1].HitQueue(rt), modules[1].HitQueue(rt)
        rt.queues = [s.waiting, s.skipped_waiting]
    cached = request(core, "cache", 32)
    s.kv_cache_manager.allocate_slots(cached, 32)
    s.kv_cache_manager.free(cached)
    a = enqueue(s, request(core, "A", 33, cached.block_hashes))
    b = enqueue(s, request(core, "B", 33))
    out = s.schedule()
    if waiting:
        assert out.num_scheduled_tokens == {"B": 16}
        assert b.num_computed_tokens == 16
        # A lower-hit newcomer cannot interrupt an already running prefill.
        enqueue(s, request(core, "C", 17))
        b.num_in_flight_tokens = 0
        out = s.schedule()
        assert out.num_scheduled_tokens == {"B": 16}
    else:
        assert list(out.num_scheduled_tokens) == ["A", "B"]
        assert out.num_scheduled_tokens == {"A": 1, "B": 15}
        assert a.num_computed_tokens == 33


def test_selected_request_stays_bound_when_its_cache_insertion_changes_rank(
    modules, monkeypatch
):
    from scheduler_cpu import enqueue, scheduler

    core = load_cache_manager_core()
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", core)
    pool = core.BlockPool(100, True, 16)
    s = scheduler(core, pool, budget=32)
    rt = modules[2].DemandRuntime(pool, eviction=False, waiting=True)
    rt.manager = s.kv_cache_manager
    s.demand_runtime = pool.demand_runtime = s.kv_cache_manager.demand_runtime = rt
    s.waiting, s.skipped_waiting = modules[1].HitQueue(rt), modules[1].HitQueue(rt)
    rt.queues = [s.waiting, s.skipped_waiting]
    enqueue(s, request(core, "A", 65))
    enqueue(s, request(core, "B", 65))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"A": 32}
    assert [r.request_id for r in s.running] == ["A"]
    assert [r.request_id for r in s.waiting] == ["B"]


def test_scheduler_off_off_matches_untouched_reference(modules):
    from native_loader import REFERENCE
    from scheduler_cpu import enqueue, scheduler

    outputs = []
    for source in (REFERENCE, WORKTREE):
        core = load_cache_manager_core(source)
        pool = core.BlockPool(20, True, 16)
        s = scheduler(core, pool, source=source, budget=17, max_running=2)
        a = enqueue(s, request(core, "A", 33))
        enqueue(s, request(core, "B", 33))
        first = s.schedule()
        a.num_in_flight_tokens = 0
        second = s.schedule()
        outputs.append(
            (
                first.num_scheduled_tokens,
                second.num_scheduled_tokens,
                [(r.request_id, r.num_computed_tokens) for r in s.running],
                snapshot(pool),
            )
        )
    assert outputs[0] == outputs[1]


def test_unexpected_alias_is_protected_without_losing_owner_metadata(modules):
    core, pool, rt, _, cached = fixture_pool(modules, [3])
    alias = core.make_block_hash_with_group_id(b"alias", 0)
    pool._insert_block_hash(alias, cached[0][-1], num_tokens=8)
    before = snapshot(pool)
    assert rt.reserve(1) is False
    assert snapshot(pool) == before
    assert rt.policy.chains[0] == [1, 2, 3]


def test_validation_after_planning_rejects_changed_ownership(modules, monkeypatch):
    _, pool, rt, _, _ = fixture_pool(modules, [3])
    before = snapshot(pool)
    original = rt.policy.plan

    def race(*args, **kwargs):
        plan = original(*args, **kwargs)
        rt.policy.blocks[plan.ids[0]].known = False
        return plan

    monkeypatch.setattr(rt.policy, "plan", race)
    assert rt.reserve(1) is False
    assert snapshot(pool) == before
    assert rt.stats["validation_failure"] == 1


def test_zero_demand_has_no_expiry_or_metadata_work(modules):
    _, pool, rt, clock, _ = fixture_pool(modules, [3])
    clock[0] = 100
    before = snapshot(pool)
    stats = dict(rt.policy.stats)
    assert pool.get_new_blocks(0) == []
    assert dict(rt.policy.stats) == stats
    assert snapshot(pool) == before


def test_continuous_low_hit_arrivals_report_wait_without_aging(modules):
    import time

    rt = SimpleNamespace(sequence=0, score=lambda r: (r.hits, {r.request_id}))
    q = modules[1].HitQueue(rt)
    high = SimpleNamespace(request_id="high", hits=100)
    q.add_request(high)
    high._demand_enqueued = time.monotonic() - 2
    # Replace the age entry to use the deterministic past enqueue time.
    q.age_heap = [(high._demand_enqueued, "high")]
    for i in range(100):
        q.add_request(SimpleNamespace(request_id=str(i), hits=0))
        assert q.pop_request().request_id == str(i)
    assert q.peek_request() is high
    assert q.stats["pending_over_1s_observations"] >= 100
    assert len(q.heap) <= 2 * len(q) + 64


@pytest.mark.parametrize("change", ["async", "alias", "connector", "groups"])
def test_unsupported_config_is_rejected_before_runtime(modules, monkeypatch, change):
    core = load_cache_manager_core()
    s = fake_scheduler(core, core.BlockPool(5, True, 16))
    monkeypatch.setenv("VLLM_DEMAND_EVICTION", "1")
    if change == "async":
        s.vllm_config.scheduler_config.async_scheduling = True
    elif change == "alias":
        s.hash_block_size = 8
    elif change == "connector":
        s.vllm_config.kv_transfer_config = object()
    else:
        s.kv_cache_config.kv_cache_groups *= 2
    with pytest.raises(ValueError, match="Demand policy"):
        modules[2].create_runtime(s)


def test_rollback_restores_duplicate_hash_lookup_order(modules):
    _, pool, rt, _, _ = fixture_pool(modules, [1], spare=1)
    second = pool.get_new_blocks(1)
    r = SimpleNamespace(
        request_id="other", session_id="other", block_hashes=chain_hashes(1, "0")
    )
    pool.cache_full_blocks(r, second, 0, 1, 16, 0)
    pool.free_blocks(second)
    before = snapshot(pool)
    first = pool.get_cached_block(r.block_hashes[0], [0])
    rt.fail_after = 1
    assert rt.reserve(1) is False
    assert pool.get_cached_block(r.block_hashes[0], [0]) == first
    assert snapshot(pool) == before


def test_telemetry_failure_does_not_retry_committed_allocation(modules, monkeypatch):
    _, pool, rt, _, _ = fixture_pool(modules, [3])

    def fail(_):
        raise RuntimeError("event sink unavailable")

    monkeypatch.setattr(pool, "_emit_block_removed_events", fail)
    blocks = pool.get_new_blocks(1)
    assert [b.block_id for b in blocks] == [3]
    assert rt.stats["notification_errors"] == 1
    assert sum(b.block_hash is not None for b in pool.blocks) == 2
    assert_queue_integrity(pool)


def test_stale_native_hash_index_is_rejected(modules):
    _, pool, rt, _, cached = fixture_pool(modules, [2])
    tail = cached[0][-1]
    pool.cached_block_hash_to_block.pop(tail.block_hash, tail.block_id)
    before = snapshot(pool)
    assert rt.reserve(1) is False
    assert snapshot(pool) == before


def test_clock_is_sampled_once_for_victim_decision(modules):
    _, pool, rt, _, _ = fixture_pool(modules, [2])
    times = iter([0.999999, 20.0])
    rt.clock = lambda: next(times)
    pool.get_new_blocks(1)
    assert rt.policy.stats["selected_warm"] == 1
    assert next(times) == 20.0


def test_resumed_request_rechecks_prefix_and_running_decode_keeps_priority(
    modules, monkeypatch
):
    from scheduler_cpu import enqueue, scheduler

    core = load_cache_manager_core()
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", core)
    pool = core.BlockPool(20, True, 16)
    s = scheduler(core, pool, budget=1, max_running=1)
    rt = modules[2].DemandRuntime(pool, eviction=True, waiting=True)
    rt.manager = s.kv_cache_manager
    s.demand_runtime = pool.demand_runtime = s.kv_cache_manager.demand_runtime = rt
    s.waiting, s.skipped_waiting = modules[1].HitQueue(rt), modules[1].HitQueue(rt)
    rt.queues = [s.waiting, s.skipped_waiting]
    old = request(core, "old", 17)
    s.kv_cache_manager.allocate_slots(old, 17)
    cached_block = s.kv_cache_manager.get_blocks(old.request_id).blocks[0][0]
    s.kv_cache_manager.free(old)
    resumed = enqueue(s, request(core, "resumed", 17, old.block_hashes))
    resumed.status = core.RequestStatus.PREEMPTED
    resumed.num_preemptions = 1
    assert s.schedule().num_scheduled_tokens == {"resumed": 1}
    assert resumed.num_computed_tokens == 17
    assert (
        s.kv_cache_manager.get_blocks(resumed.request_id).blocks[0][0] is cached_block
    )
    resumed.num_tokens = resumed.num_tokens_with_spec = 18
    resumed.num_in_flight_tokens = 0
    enqueue(s, request(core, "waiting", 17))
    assert s.schedule().num_scheduled_tokens == {"resumed": 1}
    assert len(s.waiting) == 1
