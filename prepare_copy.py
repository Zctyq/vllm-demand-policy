"""Capture local provenance and copy sources without altering their repositories."""

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parent
BASE = DEST.parent
SOURCE = BASE / "vllm-progressive-eviction"
ORIGINAL = BASE / "vllm-0.29.0"
AUDIT = DEST / "audit"
AUDIT.mkdir(exist_ok=True)


def git(path, *args):
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, check=False
    )
    return result.stdout.decode(), result.returncode


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


provenance = {"platform": platform.platform(), "python": platform.python_version()}
for name, root in (
    ("project", SOURCE),
    ("worktree", SOURCE / "worktree"),
    ("reference", SOURCE / "reference/vllm-v0.29.0"),
):
    provenance[name] = {"path": str(root), "head": git(root, "rev-parse", "HEAD")[0].strip()}
    for label, args in (
        ("status", ("status", "--short")),
        ("unstaged", ("diff", "--binary")),
        ("staged", ("diff", "--cached", "--binary")),
    ):
        (AUDIT / f"{name}-{label}.txt").write_text(git(root, *args)[0])

ignore = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"
)
for name in ("worktree", "reference", "tests", "tools", "experiment", "configs", "docs", "patches"):
    shutil.copytree(SOURCE / name, DEST / name, ignore=ignore)
for name in ("README.md", "requirements-local.lock.txt", ".gitignore"):
    shutil.copy2(SOURCE / name, DEST / name)
shutil.copy2(Path.home() / "Downloads/vllm_policy_codex_spec.txt", AUDIT / "spec.txt")

critical = [
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/request.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/engine/input_processor.py",
    "vllm/envs.py",
]
provenance["source_comparison"] = {}
for rel in critical:
    hashes = {
        label: digest(root / rel)
        for label, root in (
            ("original", ORIGINAL),
            ("reference", SOURCE / "reference/vllm-v0.29.0"),
            ("progressive", SOURCE / "worktree"),
            ("copy", DEST / "worktree"),
        )
    }
    provenance["source_comparison"][rel] = hashes

snapshot = {}
for prefix in ("worktree", "reference", "tests", "tools", "experiment", "configs"):
    for path in sorted((DEST / prefix).rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(DEST))] = digest(path)
(AUDIT / "copy-sha256.json").write_text(json.dumps(snapshot, indent=2) + "\n")
(AUDIT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({"destination": str(DEST), "copied_files": len(snapshot), "critical_files": len(critical), "original_matches_reference": all(v["original"] == v["reference"] for v in provenance["source_comparison"].values()), "copy_matches_progressive": all(v["copy"] == v["progressive"] for v in provenance["source_comparison"].values())}, indent=2))
