"""Build and byte-verify patches without touching either original checkout."""

import difflib
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "vllm-progressive-eviction"
DEST = ROOT / "results/demand"
DEST.mkdir(parents=True, exist_ok=True)
CORE = ["block_pool.py", "kv_cache_manager.py", "sched/scheduler.py"]
NEW = ["demand_policy.py", "demand_runtime.py", "demand_queue.py", "demand_metadata.py"]
LEGACY = ["progressive_policy.py", "progressive_runtime.py"]
PREFIX = "vllm/v1/core/"
checks = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


for label, base, names in (
    ("progressive-to-demand", SOURCE / "worktree", CORE + NEW),
    ("native-to-demand", ROOT / "reference/vllm-v0.29.0", CORE + NEW + LEGACY),
):
    patch = []
    for name in names:
        rel = PREFIX + name
        old = base / rel
        new = ROOT / "worktree" / rel
        patch.extend(difflib.unified_diff(
            old.read_text().splitlines(keepends=True) if old.exists() else [],
            new.read_text().splitlines(keepends=True),
            fromfile="a/" + rel if old.exists() else "/dev/null", tofile="b/" + rel,
        ))
    path = DEST / (label + ".patch")
    path.write_text("".join(patch))
    with tempfile.TemporaryDirectory(prefix="demand-patch-") as temp:
        copy = Path(temp)
        for name in names:
            rel = PREFIX + name
            if (base / rel).exists():
                (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base / rel, copy / rel)
        subprocess.run(["git", "apply", "--check", str(path)], cwd=copy, check=True)
        subprocess.run(["git", "apply", str(path)], cwd=copy, check=True)
        assert all((copy / (PREFIX + n)).read_bytes() ==
                   (ROOT / "worktree" / (PREFIX + n)).read_bytes() for n in names)
        subprocess.run(["git", "apply", "--reverse", str(path)], cwd=copy, check=True)
        for name in names:
            rel = PREFIX + name
            if (base / rel).exists():
                assert (copy / rel).read_bytes() == (base / rel).read_bytes()
            else:
                assert not (copy / rel).exists()
    checks.append({"patch": path.name, "sha256": sha(path.read_bytes()),
                   "forward_byte_equivalent": True, "reverse_byte_equivalent": True})

audit = json.loads((ROOT / "audit/provenance.json").read_text())
files = list(audit["source_comparison"])
versions = {}
for label, base in (("native", ROOT / "reference/vllm-v0.29.0"),
                    ("progressive", SOURCE / "worktree"),
                    ("demand", ROOT / "worktree")):
    versions[label] = {rel: sha((base / rel).read_bytes()) for rel in files}
    for name in NEW + LEGACY:
        rel = PREFIX + name
        if (base / rel).exists():
            versions[label][rel] = sha((base / rel).read_bytes())
(DEST / "source-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
(DEST / "patch-validation.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
