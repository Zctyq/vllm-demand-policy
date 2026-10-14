"""Deterministic demand-time policy, independent of vLLM and GPU imports."""

from __future__ import annotations

import bisect
import heapq
from collections import Counter
from dataclasses import dataclass, field


class Counters(Counter):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def __setitem__(self, key, value):
        if self.enabled:
            super().__setitem__(key, value)


@dataclass
class Block:
    owners: set[int] = field(default_factory=set)
    used: float = 0.0
    active: int = 0
    known: bool = True
    full: bool = True
    cold: bool = False
    version: int = 0
    key: bytes | None = None


@dataclass
class Plan:
    ids: list[int]
    stages: list[tuple[str, int, int, int]]
    cursor: dict[str, int]
    stamps: dict[int, tuple] = field(default_factory=dict)


class LengthIndex:
    """Incremental sorted length buckets; no allocation-time session scan.

    Python list insertion/removal has linear shift cost. This is deliberately
    exposed in benchmarks, rather than claiming logarithmic index updates.
    """

    def __init__(self):
        self.lengths: list[int] = []
        self.buckets: dict[int, list[int]] = {}
        self.keys: dict[int, int] = {}

    def set(self, owner, length=None):
        if self.keys.get(owner) == length:
            return
        old = self.keys.pop(owner, None)
        if old is not None:
            bucket = self.buckets[old]
            bucket.pop(bisect.bisect_left(bucket, owner))
            if not bucket:
                del self.buckets[old]
                self.lengths.pop(bisect.bisect_left(self.lengths, old))
        if length is not None:
            if length not in self.buckets:
                bisect.insort(self.lengths, length)
                self.buckets[length] = []
            bisect.insort(self.buckets[length], owner)
            self.keys[owner] = length

    def choose(self, cursor):
        if not self.lengths:
            return None
        longest = self.lengths[-1]
        bucket = self.buckets[longest]
        if len(self.lengths) > 1:
            return bucket[0], self.lengths[-2], False
        pos = bisect.bisect_right(bucket, cursor)
        return bucket[pos % len(bucket)], longest - 1, True


class DemandPolicy:
    def __init__(self, telemetry=True):
        self.blocks: dict[int, Block] = {}
        self.chains: dict[int, list[int]] = {}
        self.members: dict[int, set[int]] = {}
        self.names: dict[str, int] = {}
        self.next_owner = 0
        self.next_version = 0
        self.index = {layer: LengthIndex() for layer in ("cold", "warm")}
        self.cursor = {"cold": -1, "warm": -1}
        self.deadlines = []
        self.stats = Counter()
        self.telemetry = telemetry
        self.now = 0.0

    def count(self, name, value=1):
        if self.telemetry:
            self.stats[name] += value

    def owner(self, name):
        if name not in self.names:
            owner = self.next_owner
            self.next_owner += 1
            self.names[name] = owner
            self.chains[owner] = []
            self.members[owner] = set()
        return self.names[name]

    def cache(self, bid, now, *, active=0, known=True, full=True):
        self.blocks[bid] = Block(used=now, active=active, known=known, full=full)
        self.reuse(bid, now)

    def attach(self, name, bids):
        owner = self.owner(name)
        affected = {owner}
        if self.chains[owner]:
            affected.update(self.blocks[self.chains[owner][-1]].owners)
        for bid in bids:
            block = self.blocks.get(bid)
            if block is None or not block.full or bid in self.members[owner]:
                continue
            affected.update(block.owners)
            block.owners.add(owner)
            self.chains[owner].append(bid)
            self.members[owner].add(bid)
        self.refresh(affected)
        return owner

    def reason(self, bid):
        self.count("tail_checks")
        block = self.blocks[bid]
        if block.active:
            return "active"
        if not block.known or not block.owners:
            return "unknown"
        if not block.full:
            return "partial"
        if any(not self.chains[o] or self.chains[o][-1] != bid for o in block.owners):
            return "shared_interior"
        return None

    def refresh(self, owners):
        affected = set(owners)
        for owner in tuple(affected):
            chain = self.chains[owner]
            if chain:
                affected.update(self.blocks[chain[-1]].owners)
        for owner in affected:
            self.count("candidate_updates")
            chain = self.chains[owner]
            reason = self.reason(chain[-1]) if chain else None
            if reason:
                self.count("protected_" + reason)
            layer = (
                ("cold" if self.blocks[chain[-1]].cold else "warm")
                if chain and reason is None
                else None
            )
            for name, index in self.index.items():
                index.set(owner, len(chain) if name == layer else None)

    def set_active(self, bid, active):
        block = self.blocks.get(bid)
        if block is not None and block.active != active:
            previous = block.active
            block.active = active
            if bool(previous) != bool(active) and self._common_tail(bid, block):
                self.refresh(block.owners)

    def _common_tail(self, bid, block):
        return bool(block.owners) and all(
            self.chains[o] and self.chains[o][-1] == bid for o in block.owners
        )

    def reuse(self, bid, now):
        block = self.blocks[bid]
        was_cold = block.cold
        block.used = now
        self.next_version += 1
        block.version = self.next_version
        block.cold = False
        heapq.heappush(self.deadlines, (now + 1, bid, block.version))
        if was_cold and not block.active and self._common_tail(bid, block):
            self.refresh(block.owners)
        if len(self.deadlines) > 2 * len(self.blocks) + 64:
            self.count("heap_rebuild_items", len(self.blocks))
            self.deadlines = [
                (b.used + 1, i, b.version) for i, b in self.blocks.items() if not b.cold
            ]
            heapq.heapify(self.deadlines)

    def advance(self, now):
        if now < self.now:
            raise ValueError("decision clock must be monotonic")
        self.now = now
        affected = set()
        while self.deadlines and self.deadlines[0][0] <= now:
            _, bid, version = heapq.heappop(self.deadlines)
            self.count("expiry_pops")
            block = self.blocks.get(bid)
            if block is None or block.version != version:
                self.count("expiry_stale")
                continue
            block.cold = True
            affected.update(block.owners)
        self.refresh(affected)

    def remove(self, bid):
        block = self.blocks.pop(bid, None)
        if block is None:
            return
        for owner in block.owners:
            chain = self.chains[owner]
            if chain and chain[-1] == bid:
                chain.pop()
            else:
                chain.remove(bid)
                # An external invalidation creates an unproven chain. Protect
                # the remaining suffix instead of treating an interior as tail.
                for remaining in chain:
                    self.blocks[remaining].known = False
            self.members[owner].remove(bid)
        self.refresh(block.owners)

    def _choose(self, layer, cursor):
        return self.index[layer].choose(cursor)

    def _round_robin(self, layer, length, cursor):
        bucket = self.index[layer].buckets.get(length)
        if not bucket:
            return None
        pos = bisect.bisect_right(bucket, cursor)
        return bucket[pos % len(bucket)], length - 1, True

    def plan(self, demand, now, excluded=()):
        """Plan without committing lengths, reuse clocks or RR cursor."""
        self.advance(now)
        protected = []
        for bid in set(excluded):
            block = self.blocks.get(bid)
            if block is not None:
                protected.append((bid, block.active))
                self.set_active(bid, block.active + 1)
        ids, stages, trail = [], [], []
        cursor = dict(self.cursor)
        round_layer, round_length = None, None
        try:
            while len(ids) < demand:
                layer = "cold"
                choice = None
                if round_layer == "warm" and self._choose("cold", cursor["cold"]):
                    round_layer = None
                if round_layer is not None:
                    layer = round_layer
                    choice = self._round_robin(layer, round_length, cursor[layer])
                    if choice is None:
                        round_layer = None
                if choice is None:
                    layer = "cold"
                    choice = self._choose(layer, cursor[layer])
                if choice is None:
                    layer = "warm"
                    choice = self._choose(layer, cursor[layer])
                if choice is None:
                    break
                owner, target, rr = choice
                if rr and round_layer is None:
                    round_layer, round_length = layer, target + 1
                start = len(self.chains[owner])
                assert start and self.reason(self.chains[owner][-1]) is None
                self.count("stages")
                while len(ids) < demand and len(self.chains[owner]) > target:
                    chain = self.chains[owner]
                    if not chain:
                        break
                    bid = chain[-1]
                    block = self.blocks[bid]
                    if self.reason(bid) or block.cold != (layer == "cold"):
                        break
                    self.count("selected_" + layer)
                    self.count("shared_selected", int(len(block.owners) > 1))
                    ids.append(bid)
                    trail.append((bid, tuple(block.owners)))
                    for affected in block.owners:
                        assert self.chains[affected].pop() == bid
                    self.refresh(block.owners)
                    if rr:
                        cursor[layer] = owner
                        break
                stages.append((layer, owner, start, len(self.chains[owner])))
            stamps = {
                bid: (
                    self.blocks[bid].version,
                    tuple(sorted(self.blocks[bid].owners)),
                    self.blocks[bid].known,
                    self.blocks[bid].full,
                )
                for bid in ids
            }
            return Plan(ids, stages, cursor, stamps) if len(ids) == demand else None
        finally:
            for bid, owners in reversed(trail):
                affected = set(owners)
                for owner in owners:
                    if self.chains[owner]:
                        affected.update(self.blocks[self.chains[owner][-1]].owners)
                    self.chains[owner].append(bid)
                self.refresh(affected)
            for bid, active in protected:
                self.set_active(bid, active)

    def commit(self, plan):
        for bid in plan.ids:
            self.remove(bid)
        self.cursor = dict(plan.cursor)


class ReferencePolicy(DemandPolicy):
    """Slow independent candidate enumeration for exact selection comparison."""

    def plan(self, demand, now, excluded=()):
        self.advance(now)
        chains = {owner: list(chain) for owner, chain in self.chains.items()}
        excluded = set(excluded)
        cursor, ids, stages = dict(self.cursor), [], []
        rr_layer = rr_length = None

        def eligible(bid):
            b = self.blocks[bid]
            self.count("reference_tail_checks")
            return (
                bid not in excluded
                and not b.active
                and b.known
                and b.full
                and b.owners
                and all(chains[o] and chains[o][-1] == bid for o in b.owners)
            )

        def candidates(layer):
            return {
                o: len(c)
                for o, c in chains.items()
                if c
                and eligible(c[-1])
                and self.blocks[c[-1]].cold == (layer == "cold")
            }

        while len(ids) < demand:
            group = {}
            if rr_layer == "warm" and candidates("cold"):
                rr_layer = None
            if rr_layer is not None:
                layer = rr_layer
                group = {o: n for o, n in candidates(layer).items() if n == rr_length}
                if not group:
                    rr_layer = None
            if not group:
                layer = "cold"
                group = candidates(layer)
                if not group:
                    layer = "warm"
                    group = candidates(layer)
            if not group:
                return None
            levels = sorted(set(group.values()))
            rr = rr_layer is not None or len(levels) == 1
            if rr:
                owners = sorted(group)
                owner = next((o for o in owners if o > cursor[layer]), owners[0])
                target = group[owner] - 1
                rr_layer, rr_length = layer, group[owner]
            else:
                owner = min(o for o, n in group.items() if n == levels[-1])
                target = levels[-2]
            start = len(chains[owner])
            while len(ids) < demand and len(chains[owner]) > target:
                bid = chains[owner][-1]
                if not eligible(bid) or self.blocks[bid].cold != (layer == "cold"):
                    break
                ids.append(bid)
                for affected in self.blocks[bid].owners:
                    chains[affected].pop()
                if rr:
                    cursor[layer] = owner
                    break
            stages.append((layer, owner, start, len(chains[owner])))
        stamps = {
            i: (
                self.blocks[i].version,
                tuple(sorted(self.blocks[i].owners)),
                self.blocks[i].known,
                self.blocks[i].full,
            )
            for i in ids
        }
        return Plan(ids, stages, cursor, stamps)

    def _choose(self, layer, cursor):
        candidates = []
        for owner, chain in self.chains.items():
            if (
                chain
                and self.reason(chain[-1]) is None
                and self.blocks[chain[-1]].cold == (layer == "cold")
            ):
                candidates.append((len(chain), owner))
        if not candidates:
            return None
        lengths = sorted({length for length, _ in candidates})
        owners = sorted(o for length, o in candidates if length == lengths[-1])
        if len(lengths) > 1:
            return owners[0], lengths[-2], False
        return next((o for o in owners if o > cursor), owners[0]), lengths[-1] - 1, True

    def _round_robin(self, layer, length, cursor):
        owners = sorted(
            o
            for o, chain in self.chains.items()
            if len(chain) == length
            and self.reason(chain[-1]) is None
            and self.blocks[chain[-1]].cold == (layer == "cold")
        )
        if not owners:
            return None
        return next((o for o in owners if o > cursor), owners[0]), length - 1, True
