"""Version-gated replacement, with a reversible archive and native overlay."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path("/data/zhaohairui/vllm-progressive-eviction")
RECORD = HERE.parent / "deployment.json"
VERSIONS = json.loads((HERE / "source-versions.json").read_text())
INSTALLED = Path(importlib.util.find_spec("vllm").origin).parent.parent
LIVE = [PROJECT / "worktree", PROJECT / "overlay-v029-verified/site-packages"]
CHANGED = ["sched/scheduler.py", "block_pool.py", "kv_cache_manager.py"]
NEW = ["demand_policy.py", "demand_queue.py", "demand_runtime.py", "demand_metadata.py"]
OLD = ["progressive_policy.py", "progressive_runtime.py"]
PREFIX = "vllm/v1/core/"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RECORD.exists():
        raise RuntimeError("Deployment already recorded; refusing overwrite")
    for rel, digest in VERSIONS["native"].items():
        assert sha(INSTALLED / rel) == digest, ("native mismatch", rel)
    for base in LIVE:
        for rel, digest in VERSIONS["progressive"].items():
            assert sha(base / rel) == digest, ("live mismatch", str(base), rel)
    before, backups = {}, {}
    for number, base in enumerate(LIVE):
        archive = HERE.parent / "backup" / str(number)
        for name in CHANGED + OLD:
            rel = PREFIX + name
            path, dest = base / rel, archive / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            before[str(path)] = sha(path)
            backups[str(path)] = str(dest)
    native = HERE.parent / "native-overlay/site-packages"
    native.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        INSTALLED / "vllm",
        native / "vllm",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    if (INSTALLED / "vllm.libs").exists():
        shutil.copytree(INSTALLED / "vllm.libs", native / "vllm.libs")
    for base in LIVE:
        for name in CHANGED + NEW:
            rel = PREFIX + name
            shutil.copy2(HERE / "worktree" / rel, base / rel)
        for name in OLD:
            src = base / PREFIX / name
            # A move into a named archive retires the old live modules reversibly.
            retired = HERE.parent / "retired" / str(LIVE.index(base)) / name
            retired.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(retired))
    after = {
        str(base / PREFIX / n): sha(base / PREFIX / n)
        for base in LIVE
        for n in CHANGED + NEW
    }
    record = {
        "python": sys.executable,
        "installed": str(INSTALLED),
        "original_hashes": before,
        "backups": backups,
        "new_hashes": after,
        "native_overlay": str(native),
        "demand_overlay": str(LIVE[1]),
        "deleted_files": [],
        "retired_strategy_modules": OLD,
    }
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "deployed": True,
                "native_overlay": str(native),
                "backup": str(HERE.parent / "backup"),
                "deleted_files": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
