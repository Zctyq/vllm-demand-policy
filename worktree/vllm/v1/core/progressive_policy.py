# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observed-prefix policy; all physical mutation is delegated to BlockPool."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

Copy = tuple[int, int]
UNKNOWN_OWNER = "<unattributed>"


class PoolAdapter(Protocol):
    def copies(self, key: bytes) -> list[Copy]: ...
    def check(self, key: bytes, expected: list[Copy]) -> str | None: ...
    def evict(self, key: bytes, expected: list[Copy]) -> int: ...
    def free_count(self) -> int: ...


@dataclass
class PrefixNode:
    key: bytes
    parent: bytes | None
    depth: int
    born_epoch: int
    owners: set[str] = field(default_factory=set)
    history_owners: set[str] = field(default_factory=set)
    children: set[bytes] = field(default_factory=set)
    copies: dict[int, int] = field(default_factory=dict)
    last_access_epoch: int = -1
    cold_streak: int = 0


class ProgressivePolicy:
    """Track committed history and trim only observed private terminal chains.

    Keys include the engine prefix hash and cache group. Per-block heat makes
    partial prefix visits equivalent to splitting a segment at the visit edge.
    Ghost nodes retain history and session ownership after physical eviction.
    """

    def __init__(
        self,
        block_size: int,
        mode: str = "native",
        check_interval: int = 10,
        first_ratio: float = 0.20,
        later_ratio: float = 0.40,
    ):
        if mode not in {"native", "progressive"}:
            raise ValueError("policy must be native or progressive")
        if block_size < 1 or check_interval < 1:
            raise ValueError("block size and check interval must be positive")
        if not (0 < first_ratio <= 1 and 0 < later_ratio <= 1):
            raise ValueError("ratios must be in (0, 1]")
        self.block_size = block_size
        self.mode = mode
        self.check_interval = check_interval
        self.first_ratio = first_ratio
        self.later_ratio = later_ratio
        self.reset()

    def reset(self) -> None:
        self.nodes: dict[bytes, PrefixNode] = {}
        self.lookups: dict[str, int] = {}
        self.finished: set[str] = set()
        self.completed = 0
        self.last_completed = 0
        self.next_threshold = self.check_interval
        self.epoch = 0
        self.checkpoint = 0
        self.totals = {
            "active_logical": 0,
            "active_physical": 0,
            "native_logical": 0,
            "native_physical": 0,
        }

    def observe_access(
        self,
        request_id: str,
        session_id: str | None,
        keys: list[bytes],
        max_cacheable_tokens: int,
    ) -> int:
        """Record a first logical lookup, including misses to ghost contents."""
        if request_id in self.lookups:
            return self.lookups[request_id]
        owner = session_id if session_id is not None else UNKNOWN_OWNER
        history_blocks = 0
        history_open = session_id is not None
        parent = None
        for key in keys:
            node = self.nodes.get(key)
            if (
                history_open
                and node is not None
                and node.parent == parent
                and owner in node.history_owners
            ):
                history_blocks += 1
            else:
                history_open = False
            if node is not None:
                node.owners.add(owner)
                node.last_access_epoch = self.epoch
                node.cold_streak = 0
            parent = key
        cap = max(0, max_cacheable_tokens // self.block_size) * self.block_size
        history = min(history_blocks * self.block_size, cap)
        self.lookups[request_id] = history
        return history

    def commit(
        self,
        request_id: str,
        session_id: str | None,
        keys: list[bytes],
        copies: dict[bytes, list[Copy]],
    ) -> None:
        """Register only full blocks whose GPU execution has completed.

        A logical node is committed even if its physical mapping has already
        gone; it remains eligible history, never a fictitious resident entry.
        """
        owner = session_id if session_id is not None else UNKNOWN_OWNER
        parent = None
        for depth, key in enumerate(keys):
            node = self.nodes.get(key)
            if node is None:
                node = PrefixNode(key, parent, depth, self.epoch)
                self.nodes[key] = node
                if parent is not None:
                    self.nodes[parent].children.add(key)
            elif node.parent != parent or node.depth != depth:
                raise ValueError("prefix hash identity has conflicting ancestry")
            node.owners.add(owner)
            node.history_owners.add(owner)
            # The adapter supplies all confirmed physical copies for this key.
            # Existing copies remain until a native invalidation or reconcile.
            node.copies.update(dict(copies.get(key, [])))
            parent = key

    def finish(self, request_id: str, success: bool) -> None:
        """Count a normal request once; cancellation and preemption do not count."""
        if request_id in self.finished:
            return
        self.finished.add(request_id)
        if success:
            self.completed += 1

    def on_remove(
        self,
        block_id: int,
        generation: int,
        keys: list[bytes],
        reason: str,
    ) -> None:
        """Forget old slot contents before a slot is used for another prefix."""
        physical_removed = False
        logical_removed = 0
        for key in set(keys):
            node = self.nodes.get(key)
            if node is not None and node.copies.get(block_id) == generation:
                del node.copies[block_id]
                physical_removed = True
                if not node.copies:
                    logical_removed += 1
        prefix = "active" if reason == "policy" else "native"
        if reason in {"policy", "native"}:
            # Include native eviction of provisional/unmanaged valid contents.
            if physical_removed or keys:
                self.totals[prefix + "_physical"] += 1
            self.totals[prefix + "_logical"] += logical_removed

    def maybe_check(self, adapter: PoolAdapter) -> dict | None:
        """Run at most one check at a synchronous scheduler safe boundary."""
        if self.completed < self.next_threshold:
            return None
        started = time.perf_counter()
        merged = (self.completed - self.next_threshold) // self.check_interval + 1
        self.next_threshold = (
            self.completed // self.check_interval + 1
        ) * self.check_interval
        window_completed = self.completed - self.last_completed
        self.last_completed = self.completed
        self.checkpoint += 1
        before = adapter.free_count()
        skips: Counter[str] = Counter()
        statuses: dict[bytes, str | None] = {}
        live: set[bytes] = set()

        ordered = sorted(self.nodes.values(), key=lambda node: node.depth)
        for node in ordered:
            actual = dict(adapter.copies(node.key))
            if actual:
                live.add(node.key)
            stale = [
                slot
                for slot, generation in node.copies.items()
                if actual.get(slot) != generation
            ]
            for slot in stale:
                del node.copies[slot]
            if stale:
                skips["identity_mismatch"] += 1
            status = adapter.check(node.key, list(node.copies.items()))
            statuses[node.key] = status
            if (
                node.last_access_epoch == self.epoch
                or node.born_epoch == self.epoch
                or status == "active"
            ):
                node.cold_streak = 0
            else:
                node.cold_streak += 1

        # Effective prefix is rebuilt after native reclamation, so a hole cannot
        # masquerade as a retained contiguous hit. Even orphaned resident
        # descendants still protect their ancestors from active trimming.
        effective: set[bytes] = set()
        descendant_live: set[bytes] = set()
        live_children: dict[bytes, set[bytes]] = {}
        for node in reversed(ordered):
            if node.parent is not None and (
                node.key in live or node.key in descendant_live
            ):
                descendant_live.add(node.parent)
                live_children.setdefault(node.parent, set()).add(node.key)
        for node in ordered:
            if (
                node.copies
                and node.key in live
                and (node.parent is None or node.parent in effective)
            ):
                effective.add(node.key)

        segments: list[dict] = []
        candidate_count = target_count = actual_count = physical_count = 0
        for leaf in ordered:
            if leaf.key not in effective or leaf.key in descendant_live:
                continue
            chain: list[PrefixNode] = []
            node = leaf
            while True:
                if len(node.owners) != 1 or UNKNOWN_OWNER in node.owners:
                    skips["shared_or_unattributed"] += 1
                    break
                if node.cold_streak <= 0:
                    skips[statuses[node.key] or "accessed_or_birth_protected"] += 1
                    break
                if node.cold_streak != leaf.cold_streak:
                    break
                if node.owners != leaf.owners:
                    break
                chain.append(node)
                if node.parent is None or node.parent not in effective:
                    break
                parent = self.nodes[node.parent]
                if live_children.get(parent.key, set()) != {node.key}:
                    break
                node = parent
            if not chain:
                continue

            n = len(chain)
            candidate_count += n
            ratio = self.first_ratio if leaf.cold_streak == 1 else self.later_ratio
            target = min(n, math.ceil(ratio * n)) if self.mode == "progressive" else 0
            target_count += target
            removed = 0
            stopped = None
            for node in chain[:target]:
                expected = list(node.copies.items())
                stopped = adapter.check(node.key, expected)
                if stopped is not None:
                    skips[stopped] += 1
                    break
                count = adapter.evict(node.key, expected)
                if count != len(expected) or not count:
                    stopped = "execution_recheck_refused"
                    skips[stopped] += 1
                    break
                if adapter.copies(node.key):
                    raise AssertionError("hard eviction left a hittable duplicate")
                # Production BlockPool calls on_remove. This fallback also lets
                # a small independent adapter implement the protocol correctly.
                for slot, generation in expected:
                    if node.copies.get(slot) == generation:
                        self.on_remove(slot, generation, [node.key], "policy")
                removed += 1
                physical_count += count
            actual_count += removed
            segments.append(
                {
                    "before": n,
                    "target": target,
                    "evicted": removed,
                    "remaining": n - removed,
                    "cold_streak": leaf.cold_streak,
                    "stop_reason": stopped,
                }
            )

        after = adapter.free_count()
        if before != after:
            raise AssertionError("pure invalidation changed allocatable block count")
        event = {
            "checkpoint": self.checkpoint,
            "completed_requests": self.completed,
            "window_completed": window_completed,
            "merged_thresholds": merged,
            "candidate_logical_blocks": candidate_count,
            "target_logical_blocks": target_count,
            "evicted_logical_blocks": actual_count,
            "evicted_physical_blocks": physical_count,
            "skip_reasons": dict(skips),
            "segments": segments,
            "free_before": before,
            "free_after": after,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
        self.epoch += 1
        return event
