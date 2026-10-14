"""Package boundaries: source-only portability and fail-fast data validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bundle_under_test", ROOT / "tools/bundle.py"
)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def package_fixture(root):
    """Small file tree with the same required source member names as production."""
    core, _ = bundle.source_lists(ROOT)
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "manage.py").write_text("CORE = " + repr(core) + "\n")
    _, missing = bundle.select_files(root, mode="source")
    for name in missing:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# fixture\n")
    return root


def test_source_bundle_omits_payloads_and_keeps_cpu_sources_and_small_reports(tmp_path):
    root = package_fixture(tmp_path / "package")
    (root / "results/calibration_selection.json").write_text(
        json.dumps({"session_ids": ["task-a"], "selected_calls": 3})
    )
    (root / "results/export_audit").mkdir()
    (root / "results/export_audit/token_checks.csv").write_text(
        "session_id,hash\na,123\n"
    )
    # Known inputs and renamed request content must not leak into a source bundle.
    for name in (*bundle.DATA_MANIFESTS, "results/renamed_payload.json"):
        (root / name).write_text(json.dumps({"messages": [{"content": "trace"}]}))
    (root / "results/raw_export.jsonl").write_text('{"input": "trace"}\n')
    output = tmp_path / "source.tar.gz"
    result = bundle.create_bundle(root, output, mode="source")
    evidence = bundle.verify_archive(output)
    names = {row["path"] for row in evidence["manifest"]["members"]}
    assert result["bundle_mode"] == "source"
    assert evidence["all_member_sha256_verified"]
    assert not (set(bundle.DATA_MANIFESTS) & names)
    assert "results/renamed_payload.json" not in names
    assert "results/raw_export.jsonl" not in names
    assert "results/calibration_selection.json" in names
    assert "results/export_audit/token_checks.csv" in names
    assert "tests/native_loader.py" in names
    _, worktree = bundle.source_lists(root)
    assert {"worktree/" + name for name in worktree}.issubset(names)
    assert "reference/vllm-v0.29.0/vllm/v1/core/kv_cache_utils.py" in names


def test_data_bundle_missing_manifests_fails_before_output_with_prepare_commands(
    tmp_path,
):
    root = package_fixture(tmp_path / "package")
    output = tmp_path / "uncreated/data.tar.gz"
    with pytest.raises(ValueError, match="prepare --data|prepare.*--data") as error:
        bundle.create_bundle(root, output)
    assert "replay.py subset" in str(error.value)
    assert "manifest_50.json" in str(error.value)
    assert not output.parent.exists()


def test_data_bundle_runs_consistency_validator_before_writing_archive(
    tmp_path, monkeypatch
):
    root = package_fixture(tmp_path / "package")
    for name in (*bundle.DATA_MANIFESTS, "results/calibration_selection.json"):
        (root / name).write_text("{}\n")
    calls = []

    def inconsistent_data(path):
        calls.append(path)
        raise ValueError("calibration parent SHA mismatch")

    monkeypatch.setattr(bundle, "validate_data", inconsistent_data)
    output = tmp_path / "uncreated/data.tar.gz"
    with pytest.raises(ValueError, match="parent SHA mismatch"):
        bundle.create_bundle(root, output, mode="data")
    assert calls == [root]
    assert not output.parent.exists()


def test_data_validation_checks_optional_16_and_never_ignores_a_bad_sidecar(tmp_path):
    root = package_fixture(tmp_path / "package")
    source = root / "experiment/replay.py"
    source.write_bytes((ROOT / "experiment/replay.py").read_bytes())
    replay_spec = importlib.util.spec_from_file_location("bundle_replay_test", source)
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    call = {
        "call_index": 0,
        "dataset_row": 0,
        "messages": [{"role": "user", "content": "x"}],
        "prompt_token_ids": [1],
        "input_tokens": 1,
        "prompt_sha256": replay.hash_ids([1]),
        "pre_gap_seconds": 0,
    }
    parent = root / "results/manifest_50.json"
    parent.write_text(
        json.dumps(
            {
                "schema": replay.SCHEMA,
                "output_tokens": 128,
                "max_model_len": 256,
                "selected_sessions": 50,
                "selected_calls": 50,
                "sessions": [
                    {"session_id": f"task-{i:02}", "calls": [call]} for i in range(50)
                ],
            }
        )
    )
    for count in (8, 16):
        replay.subset(
            SimpleNamespace(
                manifest=parent,
                sessions=count,
                selection=None,
                seed=20261001,
                out=root / f"results/calibration_{count}.json",
            )
        )
    sidecar = root / "results/calibration_8.selection.json"
    original = sidecar.read_bytes()
    (root / "results/calibration_selection.json").write_bytes(original)
    evidence = bundle.validate_data(root)
    assert sorted(row["selected_sessions"] for row in evidence["calibrations"]) == [
        8,
        16,
    ]
    broken = json.loads(original)
    broken["calibration_parent_manifest_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="parent SHA256"):
        bundle.validate_data(root)
    sidecar.write_bytes(original)
    child_path = root / "results/calibration_16.json"
    child = json.loads(child_path.read_text())
    child["sessions"][0]["calls"][0]["messages"][0]["content"] = "changed"
    child_path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match="complete calls"):
        bundle.validate_data(root)
