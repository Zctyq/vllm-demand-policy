#!/usr/bin/env python
"""Create or verify a reduced local experiment bundle; never uploads anything.

The bundle contains six selected worktree Python files and project metadata,
plus eight original reference files and LICENSE. It is not a full vLLM source tree.
The complete local worktree is retained without changes.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "vllm-progressive-eviction"
MANIFEST = "BUNDLE_MANIFEST.json"
DIRECTORIES = ("configs", "docs", "tools", "tests", "experiment", "patches", "results")
ROOT_FILES = ("README.md", "requirements-local.lock.txt")
DATA_MANIFESTS = ("results/manifest_50.json", "results/calibration_8.json")
SOURCE_REPORT_LIMIT = 8 * 1024 * 1024
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".tools",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    ".DS_Store",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tar", ".tar.gz", ".tgz", ".zip")
NEW_MODULES = (
    "vllm/v1/core/progressive_policy.py",
    "vllm/v1/core/progressive_runtime.py",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contains_payload(value) -> bool:
    """Distinguish full request content from numeric/hash preflight reports."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "messages",
                "prompt_token_ids",
                "input_ids",
                "token_ids",
                "input",
                "prompt",
                "output_text",
                "completion_text",
            } and isinstance(item, (dict, list, str)):
                return True
            if contains_payload(item):
                return True
    elif isinstance(value, list):
        return any(contains_payload(item) for item in value)
    return False


def source_safe(path: Path) -> bool:
    """Keep source and small reports; omit raw exports and request manifests."""
    if path.name in {"manifest_50.json", "requests.json", "original_manifest.json"}:
        return False
    if re.fullmatch(r"calibration_\d+\.json", path.name):
        return False
    if path.stat().st_size > SOURCE_REPORT_LIMIT:
        return False
    if path.suffix in {".jsonl", ".ndjson", ".parquet", ".arrow"}:
        return False
    if any(
        part in {"raw", "exports", "raw_export", "tokenizer"} for part in path.parts
    ):
        return False
    if path.suffix == ".json":
        try:
            return not contains_payload(json.loads(path.read_text()))
        except (ValueError, UnicodeError):
            return False
    return True


def preparation_commands() -> list[str]:
    return [
        (
            '"$PE_PYTHON" experiment/replay.py prepare '
            '--data "$PE_EXPORT/requests.jsonl" --tokenizer "$PE_EXPORT/tokenizer" '
            '--reference-manifest "$PE_EXPORT/original_manifest.json" '
            "--manifest results/manifest_50.json "
            "--sessions 50 --min-calls 5 --max-calls 50 "
            "--max-model-len 32768 --gap-unit s "
            "--gap-evidence docs/pre-gap-evidence.txt --order-field call"
        ),
        (
            '"$PE_PYTHON" experiment/replay.py subset '
            "--manifest results/manifest_50.json --sessions 8 "
            "--seed 20261001 "
            "--out results/calibration_8.json"
        ),
    ]


def calibration_files(root: Path) -> list[tuple[Path, Path, int]]:
    """Prefer the new manifest sidecar; an invalid sidecar never falls back."""
    paths = {root / DATA_MANIFESTS[1]}
    paths.update(
        path
        for path in (root / "results").glob("calibration_*.json")
        if re.fullmatch(r"calibration_\d+\.json", path.name)
    )
    calibrations = []
    for path in sorted(paths):
        count = int(path.stem.split("_")[-1])
        sidecar = path.with_suffix(".selection.json")
        selection = (
            sidecar
            if sidecar.exists() or count != 8
            else root / "results/calibration_selection.json"
        )
        calibrations.append((path, selection, count))
    return calibrations


def validate_data(root: Path) -> dict:
    """Validate both manifests and complete calibration membership before packing."""
    source = root / "experiment/replay.py"
    spec = importlib.util.spec_from_file_location("bundle_replay_validation", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for subset, selection, count in calibration_files(root):
        # This helper validates the parent as 50 sessions, both stored manifests,
        # token hashes/budgets, parent SHA, selected IDs and complete call contents.
        record = module.validate_subset_selection(
            root / DATA_MANIFESTS[0], subset, selection
        )
        if record["selected_sessions"] != count:
            raise ValueError(
                f"{subset.name} does not contain {count} complete sessions"
            )
        records.append(
            {
                "manifest": str(subset.relative_to(root)),
                "selection": str(selection.relative_to(root)),
                **record,
            }
        )
    return {"parent_sessions": 50, "calibrations": records}


def source_lists(root: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse((root / "tools/manage.py").read_text())
    core = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CORE" for t in node.targets)
    )
    if len(core) != 8 or not all(isinstance(name, str) for name in core):
        raise ValueError("manage.CORE must contain the audited eight original files")
    for name in core:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("manage.CORE contains an unsafe relative path")
    worktree = core[:3] + list(NEW_MODULES) + ["vllm/v1/core/kv_cache_utils.py"]
    worktree += ["pyproject.toml", "AGENTS.md", "LICENSE"]
    return core, worktree


def select_files(
    root: Path, archive: Path | None = None, mode: str = "data"
) -> tuple[list[str], list[str]]:
    if mode not in {"source", "data"}:
        raise ValueError("Bundle mode must be source or data")
    core, worktree = source_lists(root)
    required = list(ROOT_FILES) + [
        "results/RESULTS.md",
        "patches/01-observe-only.patch",
        "patches/02-enable-progressive.patch",
        "patches/progressive-eviction.patch",
        "results/patch-validation.json",
        "tests/native_loader.py",
        "tests/test_native_core.py",
        "tools/manage.py",
        "tools/bundle.py",
        "tools/gpu_smoke.py",
        "experiment/replay.py",
        "experiment/summarize.py",
    ]
    if mode == "data":
        required += [*DATA_MANIFESTS]
        for subset, selection, _ in calibration_files(root):
            required.extend(str(path.relative_to(root)) for path in (subset, selection))
    required += ["worktree/" + name for name in worktree]
    required += ["reference/vllm-v0.29.0/" + name for name in [*core, "LICENSE"]]
    selected = set(required)
    for directory in DIRECTORIES:
        for source in (root / directory).rglob("*"):
            relative = source.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if str(source).endswith(EXCLUDED_SUFFIXES):
                continue
            if archive is not None and source.resolve() == archive.resolve():
                continue
            if source.is_symlink():
                raise ValueError(f"Bundle does not follow symlinks: {relative}")
            if source.is_file():
                if mode == "source" and not source_safe(source):
                    continue
                selected.add(relative.as_posix())
    if (root / ".gitignore").is_file():
        selected.add(".gitignore")
    missing = sorted(name for name in required if not (root / name).is_file())
    for name in selected:
        if (root / name).is_symlink():
            raise ValueError(f"Bundle does not follow symlinks: {name}")
        if (
            mode == "source"
            and (root / name).is_file()
            and not source_safe(root / name)
        ):
            raise ValueError(f"Required source member contains request data: {name}")
    return sorted(selected), missing


def member_info(relative: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(f"{PREFIX}/{relative}")
    info.size = size
    info.mode = mode
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    return info


def verify_archive(archive: Path, required: set[str] | None = None) -> dict:
    with tarfile.open(archive, "r:gz") as packed:
        members = packed.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or path.parts[0] != PREFIX
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        manifest_member = packed.extractfile(f"{PREFIX}/{MANIFEST}")
        if manifest_member is None:
            raise ValueError("Archive has no member checksum manifest")
        manifest_bytes = manifest_member.read()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "progressive-eviction-reduced-bundle-1":
            raise ValueError("Unknown bundle schema")
        rows = manifest["members"]
        expected = {row["path"] for row in rows}
        if len(expected) != len(rows):
            raise ValueError("Duplicate manifest member records")
        actual = {str(PurePosixPath(name).relative_to(PREFIX)) for name in names}
        if actual != expected | {MANIFEST}:
            raise ValueError(
                "Archive members do not exactly match the checksum manifest"
            )
        needed = set(manifest["required_members"])
        if required is not None:
            needed |= required
        if not needed.issubset(expected):
            raise ValueError(
                f"Required archive members missing: {sorted(needed - expected)}"
            )
        for row in rows:
            stream = packed.extractfile(f"{PREFIX}/{row['path']}")
            if stream is None:
                raise ValueError(f"Cannot read archive member {row['path']}")
            content = stream.read()
            if len(content) != row["bytes"] or sha(content) != row["sha256"]:
                raise ValueError(f"Checksum mismatch: {row['path']}")
        with archive.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "status": "passed",
            "archive": str(archive),
            "archive_sha256": archive_hash,
            "payload_members": len(rows),
            "payload_bytes": sum(row["bytes"] for row in rows),
            "manifest_sha256": sha(manifest_bytes),
            "required_members_verified": len(needed),
            "all_member_sha256_verified": True,
            "reduced_bundle": True,
            "bundle_mode": manifest.get("bundle_mode", "data"),
            "installable_full_vllm_source": False,
            "manifest": manifest,
        }


def create_bundle(root: Path, output: Path, mode: str = "data") -> dict:
    output = output.resolve()
    if not str(output).endswith(".tar.gz"):
        raise ValueError("Output filename must end in .tar.gz")
    report_paths = [
        Path(str(output) + suffix)
        for suffix in (".manifest.json", ".verification.json")
    ]
    if output.exists() or any(path.exists() for path in report_paths):
        raise ValueError(
            "Archive or companion report already exists; choose a new name"
        )
    selected, missing = select_files(root, output, mode)
    if missing:
        hint = (
            "\nUse matching existing local manifests, or set PE_PYTHON and PE_EXPORT "
            "on the execution host and rebuild:\n"
            + "\n".join(preparation_commands())
            + "\nRebuilding changes the parent SHA when paths change; the new subset "
            "writes calibration_8.selection.json. "
            "The legacy calibration_selection.json "
            "is valid only with its original parent SHA."
        )
        raise ValueError(
            "Required delivery files are not ready: "
            + ", ".join(missing)
            + (hint if mode == "data" else "")
        )
    data_validation = validate_data(root) if mode == "data" else None
    core, worktree = source_lists(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".pe-bundle-", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        rows = []
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as packed,
        ):
            for relative in selected:
                source = root / relative
                content = source.read_bytes()
                file_mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                packed.addfile(
                    member_info(relative, len(content), file_mode),
                    io.BytesIO(content),
                )
                rows.append(
                    {
                        "path": relative,
                        "bytes": len(content),
                        "sha256": sha(content),
                        "mode": oct(file_mode),
                    }
                )
            manifest = {
                "schema": "progressive-eviction-reduced-bundle-1",
                "description": "Reduced experiment bundle; full source stays local.",
                "bundle_mode": mode,
                "data_validation": data_validation,
                "installable_full_vllm_source": False,
                "full_local_worktree_preserved": True,
                "worktree_files": worktree,
                "reference_files": [*core, "LICENSE"],
                "required_members": selected,
                "checksum_scope": (
                    "All payload files; this manifest excludes its own checksum."
                ),
                "members": rows,
            }
            data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
            packed.addfile(member_info(MANIFEST, len(data), 0o644), io.BytesIO(data))
        result = verify_archive(temporary, set(selected))
        for row in rows:
            if sha((root / row["path"]).read_bytes()) != row["sha256"]:
                raise ValueError(f"Source changed while creating bundle: {row['path']}")
        os.replace(temporary, output)
        result["archive"] = str(output)
        report_paths[0].write_text(
            json.dumps(result.pop("manifest"), ensure_ascii=False, indent=2) + "\n"
        )
        report_paths[1].write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "create", "verify"))
    parser.add_argument(
        "--mode",
        choices=("source", "data"),
        default="data",
        help="plan/create mode; default data for compatibility",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="new .tar.gz path for create")
    parser.add_argument("--archive", type=Path, help="existing .tar.gz path for verify")
    args = parser.parse_args()
    if args.action == "plan":
        selected, missing = select_files(args.root.resolve(), args.output, args.mode)
        result = {
            "action": "plan",
            "archive_created": False,
            "bundle_mode": args.mode,
            "selected_members": selected,
            "missing_required": missing,
            "ready": not missing,
            "data_validation": "performed by create before archive creation"
            if args.mode == "data"
            else "not required for source mode",
            "preparation_commands": preparation_commands()
            if missing and args.mode == "data"
            else [],
            "installable_full_vllm_source": False,
        }
    elif args.action == "create":
        name = "local" if args.mode == "data" else "source"
        output = (
            args.output or args.root / f"dist/vllm-progressive-eviction-{name}.tar.gz"
        )
        result = create_bundle(args.root.resolve(), output, args.mode)
    else:
        if args.archive is None:
            parser.error("verify requires --archive")
        result = verify_archive(args.archive.resolve())
        result.pop("manifest")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
