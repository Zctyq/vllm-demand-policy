"""Load real vLLM allocator definitions without importing torch/CUDA.

This is a CPU mechanism harness, not an EngineCore or GPU integration test.
The allocator, linked queue, hash map and hash helpers are compiled unchanged
from the selected source files. Only peripheral logging, event and annotation
dependencies are replaced; model execution is neither loaded nor simulated.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NewType, overload

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT / "worktree"
REFERENCE = ROOT / "reference" / "vllm-v0.29.0"


class EventStub:
    """Capture event payloads only; no connector or asynchronous work exists."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _compile_selected(path: Path, names: set[str], module: ModuleType) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    seen = set()
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        if name in names:
            selected.append(node)
            seen.add(name)
    if seen != names:
        raise RuntimeError(f"Missing definitions in {path}: {names - seen}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree.body = [future, *selected]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102


def load_native_core(source_root: Path = WORKTREE) -> ModuleType:
    """Return actual source classes in an isolated namespace."""
    suffix = "reference" if source_root.resolve() == REFERENCE.resolve() else "worktree"
    module = ModuleType(f"cpu_allocator_{suffix}")
    module.__dict__.update(
        dataclass=dataclass,
        NewType=NewType,
        overload=overload,
        logger=logging.getLogger(module.__name__),
        envs=SimpleNamespace(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=False),
        MEDIUM_GPU="GPU",
        BlockRemoved=EventStub,
        BlockStored=EventStub,
        AllBlocksCleared=EventStub,
    )
    sys.modules[module.__name__] = module
    utils_path = source_root / "vllm/v1/core/kv_cache_utils.py"
    pool_path = source_root / "vllm/v1/core/block_pool.py"
    _compile_selected(
        utils_path,
        {
            "BlockHash",
            "BlockHashWithGroupId",
            "ExternalBlockHash",
            "make_block_hash_with_group_id",
            "get_block_hash",
            "get_group_id",
            "maybe_convert_block_hash",
            "KVCacheBlock",
            "FreeKVCacheBlockQueue",
            "BlockHashListWithBlockSize",
            "BlockHashList",
            "resolve_block_hashes",
        },
        module,
    )
    _compile_selected(pool_path, {"BlockHashToBlockMap", "BlockPool"}, module)
    module.source_files = [str(utils_path), str(pool_path)]
    return module


def load_policy() -> ModuleType:
    """Import the complete new policy module, with no vLLM import side effects."""
    return load_source_module(
        WORKTREE / "vllm/v1/core/progressive_policy.py", "cpu_progressive_policy"
    )


def load_runtime() -> ModuleType:
    """Load the complete runtime with its real policy as the only substituted import."""
    name = "vllm.v1.core.progressive_policy"
    previous = sys.modules.get(name)
    sys.modules[name] = load_policy()
    try:
        return load_source_module(
            WORKTREE / "vllm/v1/core/progressive_runtime.py",
            "cpu_progressive_runtime",
        )
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_source_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_queue_integrity(pool) -> list[int]:
    """Check both links, uniqueness, reference counts and the native count."""
    queue = pool.free_block_queue
    current = queue.fake_free_list_head
    ids = []
    while current.next_free_block is not queue.fake_free_list_tail:
        following = current.next_free_block
        assert following is not None
        assert following.prev_free_block is current
        assert following.block_id not in ids, "Duplicate free node or cycle"
        assert following.ref_cnt == 0
        assert not following.is_null
        ids.append(following.block_id)
        assert len(ids) < pool.num_gpu_blocks
        current = following
    assert queue.fake_free_list_tail.prev_free_block is current
    assert len(ids) == pool.get_num_free_blocks()
    assert ids == [b.block_id for b in queue.get_all_free_blocks()]
    return ids


def load_cache_manager_core(source_root=WORKTREE):
    """Compile actual manager/coordinator definitions with CPU config stubs."""
    import itertools
    from abc import ABC, abstractmethod
    from collections import defaultdict
    from collections.abc import Sequence
    from enum import Enum

    module = load_native_core(source_root)

    @dataclass
    class FullAttentionSpec:
        block_size: int = 16
        sliding_window: None = None
        attention_chunk_size: None = None
        non_causal: bool = False

    class RequestStatus(Enum):
        WAITING = 1
        PREEMPTED = 2
        RUNNING = 3
        WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 4
        WAITING_FOR_REMOTE_KVS = 5
        WAITING_FOR_STREAMING_REQ = 6

    module.__dict__.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        itertools=itertools,
        defaultdict=defaultdict,
        Sequence=Sequence,
        cdiv=lambda a, b: (a + b - 1) // b,
        FullAttentionSpec=FullAttentionSpec,
        ChunkedLocalAttentionSpec=type("ChunkedLocalAttentionSpec", (), {}),
        CrossAttentionManager=type("CrossAttentionManager", (), {}),
        RequestStatus=RequestStatus,
    )
    _compile_selected(
        source_root / "vllm/v1/core/single_type_kv_cache_manager.py",
        {"SingleTypeKVCacheManager", "FullAttentionManager"},
        module,
    )
    _compile_selected(
        source_root / "vllm/v1/core/kv_cache_coordinator.py",
        {"KVCacheCoordinator", "UnitaryKVCacheCoordinator"},
        module,
    )
    _compile_selected(
        source_root / "vllm/v1/core/kv_cache_manager.py",
        {"KVCacheBlocks", "KVCacheManager"},
        module,
    )
    return module


def make_cache_manager(core, pool):
    """Supply explicit configuration to real FullAttention manager methods."""
    spec = core.FullAttentionSpec()
    single = core.FullAttentionManager(spec, pool, True, 0, 16)
    coordinator = object.__new__(core.UnitaryKVCacheCoordinator)
    coordinator.single_type_managers = [single]
    coordinator.block_pool = pool
    coordinator.kv_cache_spec = spec
    coordinator.block_size = 16
    coordinator.eagle_group_ids = set()
    coordinator.dcp_world_size = coordinator.pcp_world_size = 1
    coordinator.enable_caching = True
    coordinator.num_reprefillable_tokens = 0
    coordinator.retention_interval = None
    manager = object.__new__(core.KVCacheManager)
    manager.coordinator = coordinator
    manager.block_pool = pool
    manager.max_model_len = 32768
    manager.watermark_blocks = 0
    manager.enable_caching = True
    manager.enable_kv_cache_events = manager.log_stats = False
    manager.progressive_runtime = manager.demand_runtime = None
    manager.num_kv_cache_groups = 1
    manager.empty_kv_cache_blocks = core.KVCacheBlocks(((),))
    return manager
