"""Compare optimized maintenance against the frozen, GPU-tested implementation."""

import dataclasses
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from native_loader import load_source_module
from test_demand import model

before_path = ROOT / "audit/before-gpu-optimization/demand_policy.py"
after_path = ROOT / "worktree/vllm/v1/core/demand_policy.py"
before = load_source_module(before_path, "frozen_policy")
after = load_source_module(after_path, "optimized_policy")
rng = random.Random(228739)
plans = events = 0
for trace in range(200):
    lengths = [rng.randrange(2, 15) for _ in range(rng.randrange(2, 12))]
    old, new = model(before, lengths), model(after, lengths)
    for m in (old, new):
        m.attach("shared", m.chains[0][: lengths[0] // 2])
    now = 0.0
    for event in range(200):
        events += 1
        now += rng.choice([0, 0.001, 0.2, 1.0])
        bid = rng.choice(list(old.blocks)) if old.blocks else None
        action = rng.randrange(5)
        if bid is not None and action == 0:
            active = rng.randrange(4)
            old.set_active(bid, active)
            new.set_active(bid, active)
        elif bid is not None and action == 1:
            old.reuse(bid, now)
            new.reuse(bid, now)
        elif bid is not None and action == 2:
            for m in (old, new):
                m.blocks[bid].known = not m.blocks[bid].known
                m.refresh(m.blocks[bid].owners)
        else:
            count = rng.randrange(0, 12)
            p, q = old.plan(count, now), new.plan(count, now)
            assert (dataclasses.asdict(p) if p else None) == (
                dataclasses.asdict(q) if q else None
            ), (trace, event)
            plans += 1
            if p and rng.randrange(2):
                old.commit(p)
                new.commit(q)
        old.advance(now)
        new.advance(now)
        assert old.chains == new.chains and old.cursor == new.cursor
        assert {k: v.keys for k, v in old.index.items()} == {
            k: v.keys for k, v in new.index.items()
        }, (trace, event)
        assert {k: dataclasses.asdict(v) for k, v in old.blocks.items()} == {
            k: dataclasses.asdict(v) for k, v in new.blocks.items()
        }

timings = {}
for name, module in [("before", before), ("optimized", after)]:
    m = model(module, [1024] * 8)
    m.advance(1)
    bids = list(m.blocks)
    start = time.perf_counter()
    for cycle in range(30):
        now = 1 + cycle * 0.001
        for bid in bids:
            m.set_active(bid, 1)
            m.reuse(bid, now)
        for bid in bids:
            m.set_active(bid, 0)
    timings[name] = {
        "seconds": time.perf_counter() - start,
        "physical_blocks": len(bids),
        "cycles": 30,
        "candidate_updates": m.stats["candidate_updates"],
    }
result = {
    "traces": 200,
    "events": events,
    "plans": plans,
    "same_blocks_indices_chains_cursors_and_plans": True,
    "cpu_hit_reference_maintenance": timings,
    "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [before_path, after_path]
    },
}
out = ROOT / "results/demand/gpu-optimization-verification.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
