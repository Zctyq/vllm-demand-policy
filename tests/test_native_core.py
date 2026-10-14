"""CPU acceptance of the real allocator and progressive eviction metadata.

Purpose: detect invalid hash eviction, free-list corruption, unsafe references
and incorrect cold-window accounting. Inputs are deterministic logical chains
plus the real allocator's lifecycle calls; outputs are actual native lookup,
allocation and queue states. CPU tests are the cheapest level for those risks.
They do not verify EngineCore scheduling, model generation or GPU performance.
"""

from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest
from native_loader import (
    REFERENCE,
    assert_queue_integrity,
    load_native_core,
    load_policy,
    load_runtime,
)


@pytest.fixture
def core():
    return load_native_core()


def chain_hashes(count, seed="private"):
    parent = seed.encode()
    result = []
    for index in range(count):
        parent = hashlib.sha256(parent + index.to_bytes(4, "big")).digest()
        result.append(parent)
    return result


def cache_chain(core, pool, hashes, *, block_size=16, group_id=0):
    blocks = pool.get_new_blocks(len(hashes))
    request = SimpleNamespace(block_hashes=hashes)
    pool.cache_full_blocks(request, blocks, 0, len(blocks), block_size, group_id)
    return blocks


def native_snapshot(core):
    pool = core.BlockPool(9, True, 16)
    hashes = chain_hashes(4)
    blocks = cache_chain(core, pool, hashes)
    assert all(block.ref_cnt == 1 for block in blocks)
    pool.free_blocks(reversed(blocks))
    before = assert_queue_integrity(pool)
    hits = pool.get_cached_block(hashes[0], [0])
    pool.touch(hits)
    touched = assert_queue_integrity(pool)
    assert hits[0].ref_cnt == 1
    pool.free_blocks(hits)
    after = assert_queue_integrity(pool)
    allocated = pool.get_new_blocks(8)
    assert all(pool.get_cached_block(key, [0]) is None for key in hashes)
    assert pool.get_num_free_blocks() == 0
    with pytest.raises(ValueError, match="Cannot get"):
        pool.get_new_blocks(1)
    pool.free_blocks(allocated)
    final = assert_queue_integrity(pool)
    return before, touched, after, [b.block_id for b in allocated], final


def test_disabled_hooks_preserve_original_allocation_lookup_and_queue(core):
    """The patched default path matches the untouched v0.29 source."""
    assert native_snapshot(core) == native_snapshot(load_native_core(REFERENCE))


def test_real_hash_map_keeps_second_copy_until_both_are_evicted(core):
    pool = core.BlockPool(5, True, 16)
    key = chain_hashes(1)
    first = cache_chain(core, pool, key)[0]
    second = cache_chain(core, pool, key)[0]
    pool.free_blocks([first, second])
    before = pool.get_num_free_blocks()
    pool.evict_blocks({first.block_id})
    assert pool.get_cached_block(key[0], [0]) == [second]
    pool.evict_blocks({second.block_id})
    assert pool.get_cached_block(key[0], [0]) is None
    assert pool.get_num_free_blocks() == before
    assert_queue_integrity(pool)


def test_native_alias_removal_invalidates_every_hash_of_slot(core):
    pool = core.BlockPool(5, True, 16)
    hashes = chain_hashes(2)
    block = cache_chain(core, pool, hashes[:1])[0]
    alias = core.make_block_hash_with_group_id(hashes[1], 0)
    pool._insert_block_hash(alias, block, num_tokens=8)
    assert pool.get_cached_block(hashes[1], [0]) == [block]
    pool.free_blocks([block])
    before = pool.get_num_free_blocks()
    pool.evict_blocks({block.block_id})
    assert all(pool.get_cached_block(key, [0]) is None for key in hashes)
    assert pool.get_num_free_blocks() == before
    assert_queue_integrity(pool)


def test_hash_identity_contains_cache_group(core):
    raw = chain_hashes(1)[0]
    a = core.make_block_hash_with_group_id(raw, 0)
    b = core.make_block_hash_with_group_id(raw, 1)
    assert a != b
    assert core.get_block_hash(a) == core.get_block_hash(b) == raw
    assert core.get_group_id(a) == 0
    assert core.get_group_id(b) == 1


def test_success_count_deduplicates_and_coalesces_pending_thresholds(core):
    """Finishing a batch only arms one check, without extra cooling rounds."""
    policy = load_policy().ProgressivePolicy(16, mode="progressive")
    pool = core.BlockPool(5, True, 16)
    empty_adapter = SimpleNamespace(free_count=pool.get_num_free_blocks)
    policy.finish("cancelled", False)
    policy.finish("cancelled", True)
    for i in range(9):
        policy.finish(f"r{i}", True)
        policy.finish(f"r{i}", True)
    assert policy.completed == 9
    assert policy.maybe_check(empty_adapter) is None
    for i in range(9, 35):
        policy.finish(f"r{i}", True)
    event = policy.maybe_check(empty_adapter)
    assert event["completed_requests"] == 35
    assert event["window_completed"] == 35
    assert event["merged_thresholds"] == 3
    assert policy.epoch == 1
    assert policy.next_threshold == 40
    assert policy.maybe_check(empty_adapter) is None
    assert_queue_integrity(pool)


def test_history_uses_only_prior_commits_keeps_ghosts_and_obeys_hit_cap(core):
    """Misses still count as logical accesses; retries cannot refresh heat."""
    policy = load_policy().ProgressivePolicy(16, mode="progressive")
    keys = [core.make_block_hash_with_group_id(x, 0) for x in chain_hashes(5)]
    assert policy.observe_access("first", "s", keys, 79) == 0
    policy.commit("first", "s", keys[:4], {})
    policy.epoch = 1
    assert policy.observe_access("revisit", "s", keys, 63) == 48
    assert all(policy.nodes[key].last_access_epoch == 1 for key in keys[:4])
    assert keys[4] not in policy.nodes  # An observed input alone is not history.
    policy.epoch = 2
    assert policy.observe_access("revisit", "s", keys, 79) == 48
    assert all(policy.nodes[key].last_access_epoch == 1 for key in keys[:4])
    assert policy.observe_access("other-owner", "other", keys, 79) == 0
    assert policy.observe_access("unknown-owner", None, keys, 79) == 0


def test_unadmitted_shared_lookup_does_not_create_session_history(core):
    """Conservative sharing ownership does not prove completed session work."""
    policy = load_policy().ProgressivePolicy(16, mode="progressive")
    keys = [core.make_block_hash_with_group_id(x, 0) for x in chain_hashes(2)]
    policy.commit("a-computed", "a", keys, {})
    assert policy.observe_access("b-cancelled", "b", keys, 32) == 0
    policy.finish("b-cancelled", False)
    assert policy.observe_access("b-first-executed", "b", keys, 32) == 0
    assert all(policy.nodes[key].owners == {"a", "b"} for key in keys)
    policy.commit("b-first-executed", "b", keys, {})
    assert policy.observe_access("b-next", "b", keys, 32) == 32


@pytest.fixture
def runtime_factory(core, tmp_path):
    runtimes = []

    def create(capacity=150, mode="progressive"):
        pool = core.BlockPool(capacity + 1, True, 16)
        scheduler = SimpleNamespace(
            kv_cache_manager=SimpleNamespace(block_pool=pool),
            requests={},
            deferred_frees=[],
        )
        runtime = load_runtime().ProgressiveRuntime(
            scheduler,
            {
                "block_size": 16,
                "mode": mode,
                "check_interval": 10,
                "first_cold_ratio": 0.2,
                "later_cold_ratio": 0.4,
                "log_dir": str(tmp_path / f"run-{len(runtimes)}"),
                "run_id": "cpu-mechanism",
            },
        )
        pool.progressive_runtime = runtime
        runtimes.append(runtime)
        return runtime

    yield create
    for runtime in runtimes:
        runtime.close()


def make_request(session, index, hashes):
    marker = json.dumps({"r": "cpu-mechanism", "s": session, "c": index}).encode()
    marker = base64.urlsafe_b64encode(marker).decode().rstrip("=")
    return SimpleNamespace(
        request_id="chatcmpl-pe1." + marker + "-ab12cd34",
        block_hashes=hashes,
        num_prompt_tokens=len(hashes) * 16 + 1,
        sampling_params=None,
        output_token_ids=[1],
        num_preemptions=0,
        status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"),
        num_in_flight_tokens=0,
    )


def test_request_metadata_survives_actual_chat_and_completion_wrappers():
    runtime = load_runtime()
    request = make_request("session", 1, [])
    marker = request.request_id.removeprefix("chatcmpl-").removesuffix("-ab12cd34")
    for wrapped in (
        "chatcmpl-" + marker + "-ab12cd34",
        "cmpl-" + marker + "-0-ab12cd34",
        "cmpl-" + marker + "-999-ab12cd34",
    ):
        assert runtime.parse_request_id(wrapped) == {
            "run_id": "cpu-mechanism",
            "session_id": "session",
            "call_index": 1,
            "request_id": marker,
        }


def seed_runtime(core, runtime, count=100, *, seed="private", session="s"):
    hashes = chain_hashes(count, seed)
    request = make_request(session, 0, hashes)
    runtime.on_lookup(request, 0)
    blocks = cache_chain(core, runtime.pool, hashes)
    runtime.on_admit(request, 0)
    runtime.on_computed(request, (blocks,), count * 16)
    runtime.pool.free_blocks(reversed(blocks))
    runtime.on_finish(request)
    return hashes, blocks


def advance_to(runtime, completed):
    for index in range(runtime.policy.completed, completed):
        runtime.policy.finish(f"independent-completion-{index}", True)
    runtime.maybe_check()
    return json.loads(
        (runtime.log_dir / "policy_checks.jsonl").read_text().splitlines()[-1]
    )


def prefix_length(pool, hashes):
    """Read real native hash lookups; this is not a GPU prefill measurement."""
    count = 0
    for key in hashes:
        if pool.get_cached_block(key, [0]) is None:
            break
        count += 1
    return count


def test_real_hard_eviction_100_to_80_to_48_and_birth_window(runtime_factory, core):
    runtime = runtime_factory()
    hashes, blocks = seed_runtime(core, runtime)
    free = runtime.pool.get_num_free_blocks()
    birth = advance_to(runtime, 10)
    assert birth["evicted_logical_blocks"] == 0
    assert prefix_length(runtime.pool, hashes) == 100
    first = advance_to(runtime, 20)
    assert first["evicted_logical_blocks"] == 20
    assert first["evicted_physical_blocks"] == 20
    assert prefix_length(runtime.pool, hashes) == 80
    assert all(runtime.pool.get_cached_block(key, [0]) is None for key in hashes[80:])
    assert runtime.pool.get_num_free_blocks() == free
    assert_queue_integrity(runtime.pool)
    second = advance_to(runtime, 30)
    assert second["evicted_logical_blocks"] == 32
    assert second["evicted_physical_blocks"] == 32
    assert prefix_length(runtime.pool, hashes) == 48
    assert all(runtime.pool.get_cached_block(key, [0]) is None for key in hashes[48:])
    assert runtime.pool.get_num_free_blocks() == free
    ids = assert_queue_integrity(runtime.pool)
    assert ids[0] == blocks[48].block_id
    assert runtime.policy.totals["active_logical"] == 52
    assert runtime.policy.totals["active_physical"] == 52


@pytest.mark.parametrize("completed,expected", [(20, 80), (30, 48)])
def test_independent_revisits_observe_real_prefix_misses_and_reset_heat(
    runtime_factory, core, completed, expected
):
    runtime = runtime_factory()
    hashes, _ = seed_runtime(core, runtime)
    for threshold in range(10, completed + 1, 10):
        advance_to(runtime, threshold)
    assert prefix_length(runtime.pool, hashes) == expected
    revisit = make_request("s", 1, hashes)
    runtime.on_lookup(revisit, expected * 16)
    state = runtime.requests[revisit.request_id]
    assert state["historical_prefix_tokens"] == 1600
    assert state["first_lookup_hit_tokens"] == expected * 16
    assert all(node.cold_streak == 0 for node in runtime.policy.nodes.values())
    event = advance_to(runtime, completed + 10)
    assert event["evicted_logical_blocks"] == 0
    assert prefix_length(runtime.pool, hashes) == expected


def test_duplicate_active_copy_blocks_eviction_then_all_copies_are_removed(
    runtime_factory, core
):
    runtime = runtime_factory(capacity=10)
    hashes, original = seed_runtime(core, runtime, count=1)
    duplicate = cache_chain(core, runtime.pool, hashes)[0]
    key = core.make_block_hash_with_group_id(hashes[0], 0)
    runtime.policy.commit("duplicate", "s", [key], {key: runtime.adapter.copies(key)})
    advance_to(runtime, 10)
    event = advance_to(runtime, 20)
    assert event["evicted_logical_blocks"] == 0
    assert runtime.pool.get_cached_block(hashes[0], [0]) is not None
    assert runtime.policy.nodes[key].cold_streak == 0
    runtime.pool.free_blocks([duplicate])
    free = runtime.pool.get_num_free_blocks()
    event = advance_to(runtime, 30)
    assert event["evicted_logical_blocks"] == 1
    assert event["evicted_physical_blocks"] == 2
    assert runtime.pool.get_cached_block(hashes[0], [0]) is None
    assert original[0].block_hash is duplicate.block_hash is None
    assert runtime.pool.get_num_free_blocks() == free
    assert_queue_integrity(runtime.pool)


def test_alias_is_a_contiguous_tail_barrier(runtime_factory, core):
    runtime = runtime_factory(capacity=15)
    hashes, blocks = seed_runtime(core, runtime, count=10)
    alias = core.make_block_hash_with_group_id(chain_hashes(1, "alias")[0], 0)
    runtime.pool._insert_block_hash(alias, blocks[-1], num_tokens=8)
    advance_to(runtime, 10)
    event = advance_to(runtime, 20)
    assert event["target_logical_blocks"] == 2
    assert event["evicted_logical_blocks"] == 0
    assert event["skip_reasons"]["alias"] == 1
    assert prefix_length(runtime.pool, hashes) == 10
    assert_queue_integrity(runtime.pool)


def test_native_recycled_slot_cannot_be_evicted_by_ghost_generation(
    runtime_factory, core
):
    runtime = runtime_factory(capacity=10)
    hashes, blocks = seed_runtime(core, runtime, count=10)
    advance_to(runtime, 10)
    advance_to(runtime, 20)
    key = core.make_block_hash_with_group_id(hashes[8], 0)
    stale = [(blocks[8].block_id, runtime.generations[blocks[8].block_id])]
    assert not runtime.policy.nodes[key].copies
    new_hash = chain_hashes(1, "new-slot-content")[0]
    new_block = cache_chain(core, runtime.pool, [new_hash])[0]
    assert new_block.block_id == stale[0][0]
    assert runtime.generations[new_block.block_id] == stale[0][1] + 1
    runtime.pool.free_blocks([new_block])
    assert runtime.adapter.evict(key, stale) == 0
    advance_to(runtime, 30)
    assert runtime.pool.get_cached_block(new_hash, [0]) == [new_block]
    assert runtime.policy.totals["native_physical"] == 0
    assert_queue_integrity(runtime.pool)


def test_shared_prefix_survives_while_private_branches_trim(runtime_factory, core):
    runtime = runtime_factory(capacity=30)
    hashes, _ = seed_runtime(core, runtime, count=10, session="a")
    branch = hashes[:5] + chain_hashes(5, "branch-b")
    request = make_request("b", 0, branch)
    runtime.on_lookup(request, 5 * 16)
    shared = [runtime.pool.get_cached_block(key, [0])[0] for key in branch[:5]]
    runtime.pool.touch(shared)
    private = runtime.pool.get_new_blocks(5)
    blocks = shared + private
    runtime.pool.cache_full_blocks(request, blocks, 5, 10, 16, 0)
    runtime.on_admit(request, 80)
    runtime.on_computed(request, (blocks,), 160)
    runtime.pool.free_blocks(reversed(blocks))
    runtime.on_finish(request)
    advance_to(runtime, 10)
    event = advance_to(runtime, 20)
    assert event["evicted_logical_blocks"] == 2
    assert prefix_length(runtime.pool, hashes) == 9
    assert prefix_length(runtime.pool, branch) == 9
    assert all(runtime.pool.get_cached_block(key, [0]) for key in hashes[:5])
    assert_queue_integrity(runtime.pool)


def test_partial_prefix_visit_keeps_hot_front_and_cools_remaining_tail(
    runtime_factory, core
):
    runtime = runtime_factory()
    hashes, _ = seed_runtime(core, runtime)
    advance_to(runtime, 10)
    short = make_request("s", 1, hashes[:40])
    runtime.on_lookup(short, 40 * 16)
    event = advance_to(runtime, 20)
    assert event["candidate_logical_blocks"] == 60
    assert event["target_logical_blocks"] == 12
    assert event["evicted_logical_blocks"] == 12
    assert prefix_length(runtime.pool, hashes) == 88
    assert_queue_integrity(runtime.pool)


def test_runtime_refuses_mutation_with_in_flight_or_deferred_references(
    runtime_factory,
):
    runtime = runtime_factory()
    runtime.scheduler.requests["active"] = SimpleNamespace(num_in_flight_tokens=1)
    with pytest.raises(RuntimeError, match="in-flight"):
        runtime.maybe_check()
    runtime.scheduler.requests.clear()
    runtime.scheduler.deferred_frees.append("request")
    with pytest.raises(RuntimeError, match="in-flight"):
        runtime.maybe_check()


def test_native_pressure_counts_only_additional_valid_cache_reclamation(
    runtime_factory, core
):
    runtime = runtime_factory(capacity=10)
    hashes, _ = seed_runtime(core, runtime, count=10)
    advance_to(runtime, 10)
    advance_to(runtime, 20)
    assert runtime.policy.totals["active_physical"] == 2
    allocated = runtime.pool.get_new_blocks(4)
    assert len(allocated) == 4
    assert runtime.policy.totals["native_physical"] == 2
    assert runtime.policy.totals["native_logical"] == 2
    assert prefix_length(runtime.pool, hashes) == 6
    assert all(block.ref_cnt == 1 for block in allocated)
    runtime.pool.free_blocks(allocated)
    assert_queue_integrity(runtime.pool)


def test_observed_native_mode_does_not_change_cache_or_free_queue(
    runtime_factory, core
):
    runtime = runtime_factory(mode="native")
    hashes, _ = seed_runtime(core, runtime)
    original_queue = assert_queue_integrity(runtime.pool)
    for threshold in (10, 20, 30):
        event = advance_to(runtime, threshold)
        assert event["target_logical_blocks"] == 0
        assert event["evicted_logical_blocks"] == 0
        assert prefix_length(runtime.pool, hashes) == 100
        assert assert_queue_integrity(runtime.pool) == original_queue


def test_adopted_hit_is_bound_to_execution_and_survives_preemption(
    runtime_factory, core
):
    runtime = runtime_factory(capacity=5, mode="native")
    hashes, _ = seed_runtime(core, runtime, count=5)
    revisit = make_request("s", 1, hashes)
    runtime.on_lookup(revisit, 80)
    state = runtime.requests[revisit.request_id]
    assert state["actual_prefix_hit_tokens"] is None
    pressure = runtime.pool.get_new_blocks(2)
    runtime.pool.free_blocks(pressure)
    assert prefix_length(runtime.pool, hashes) == 3
    runtime.on_lookup(revisit, 48)
    assert state["first_lookup_hit_tokens"] == 80
    shared = [runtime.pool.get_cached_block(key, [0])[0] for key in hashes[:3]]
    runtime.pool.touch(shared)
    fresh = runtime.pool.get_new_blocks(2)
    blocks = shared + fresh
    runtime.pool.cache_full_blocks(revisit, blocks, 3, 5, 16, 0)
    runtime.on_admit(revisit, 48)
    assert state["actual_prefix_hit_tokens"] is None
    runtime.on_computed(revisit, (blocks,), 80)
    assert state["actual_prefix_hit_tokens"] == 48
    assert state["historical_cache_hit_tokens"] == 48
    assert state["historical_prefix_tokens"] == 80
    old_copies = {key: dict(node.copies) for key, node in runtime.policy.nodes.items()}
    runtime.pool.free_blocks(reversed(blocks))
    replacement = cache_chain(core, runtime.pool, hashes)
    runtime.on_admit(revisit, 0)
    runtime.on_computed(revisit, (replacement,), 80)
    assert state["actual_prefix_hit_tokens"] == 48
    assert all(
        node.copies != old_copies[key] for key, node in runtime.policy.nodes.items()
    )
    runtime.pool.free_blocks(reversed(replacement))
    runtime.on_finish(revisit)
    runtime.on_finish(revisit)
    assert runtime.policy.completed == 2
    records = (runtime.log_dir / "requests.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert json.loads(records[-1])["historical_cache_hit_tokens"] == 48
    assert_queue_integrity(runtime.pool)


def test_reset_clears_history_and_cache_without_rewinding_slot_generations(
    runtime_factory, core
):
    runtime = runtime_factory(capacity=5)
    hashes, _ = seed_runtime(core, runtime, count=5)
    generations = runtime.generations[:]
    assert runtime.pool.reset_prefix_cache()
    runtime.reset()
    assert runtime.generations == generations
    assert runtime.policy.nodes == {}
    assert runtime.requests == {}
    assert runtime.policy.completed == 0
    assert prefix_length(runtime.pool, hashes) == 0
    revisit = make_request("s", 1, hashes)
    runtime.on_lookup(revisit, 0)
    assert runtime.requests[revisit.request_id]["historical_prefix_tokens"] == 0
    blocks = cache_chain(core, runtime.pool, hashes)
    assert all(
        runtime.generations[block.block_id] == generations[block.block_id] + 1
        for block in blocks
    )
    runtime.pool.free_blocks(blocks)
    assert_queue_integrity(runtime.pool)
