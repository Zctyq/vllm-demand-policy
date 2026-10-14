"""Version-gated, reversible replacement of only three demand modules."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = Path("/data/zhaohairui/vllm-progressive-eviction")
old = json.loads((HERE / "optimization-source-hashes.json").read_text())
names = ["demand_policy.py", "demand_runtime.py", "demand_queue.py"]
roots = [
    PROJECT / "worktree/vllm/v1/core",
    PROJECT / "overlay-v029-verified/site-packages/vllm/v1/core",
    ROOT / "payload/worktree/vllm/v1/core",
]
backups = ROOT / "before-optimization"
backups.mkdir(exist_ok=False)
entries = []
for index, root in enumerate(roots):
    for name in names:
        source = root / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == old["before"][name], (source, digest)
        assert (
            hashlib.sha256((HERE / "optimized" / name).read_bytes()).hexdigest()
            == old["after"][name]
        )
        backup = backups / str(index) / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        entries.append(
            {
                "target": str(source),
                "backup": str(backup),
                "before_sha256": digest,
                "after_sha256": old["after"][name],
            }
        )
# All gates and backups complete before replacing live code.
for entry in entries:
    shutil.copy2(HERE / "optimized" / Path(entry["target"]).name, entry["target"])
    assert (
        hashlib.sha256(Path(entry["target"]).read_bytes()).hexdigest()
        == entry["after_sha256"]
    )
(ROOT / "optimization-deployment.json").write_text(
    json.dumps({"entries": entries, "deleted_files": []}, indent=2) + "\n"
)
print("OPTIMIZATION_DEPLOYED", len(entries), flush=True)
