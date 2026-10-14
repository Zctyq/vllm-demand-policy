"""CPU tests for deployment identity checks; no GPU or remote service is used."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from native_loader import load_runtime


@pytest.fixture
def manage():
    path = Path(__file__).resolve().parents[1] / "tools/manage.py"
    spec = importlib.util.spec_from_file_location("manage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deployment(manage, tmp_path):
    site = tmp_path / "site-packages"
    model = tmp_path / "model"
    model.mkdir()
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    patched = {}
    for rel in manage.CHANGED:
        target = site / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rel)
        patched[rel] = manage.sha(target)
    binary = site / "vllm/native.so"
    binary.write_bytes(b"original installed extension fixture")
    manifest = {
        "version": "0.29.0",
        "python": str(python),
        "site_packages": str(site),
        "patched_sha256": patched,
        "compiled_extensions_unchanged": {"vllm/native.so": manage.sha(binary)},
    }
    (tmp_path / "overlay_manifest.json").write_text(json.dumps(manifest))
    profile = {
        "python": str(python),
        "overlay_site": str(site),
        "model": str(model),
        "served_model_name": "experiment",
        "gpu": "0",
        "port": 18080,
        "dtype": "float16",
        "max_model_len": 2048,
        "block_size": 16,
        "num_gpu_blocks": 200,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 8,
        "enable_chunked_prefill": True,
        "attention_backend": "FLASH_ATTN",
        "seed": 17,
        "gpu_memory_utilization": 0.8,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path, profile, binary


def test_profile_refuses_compiled_overlay_drift(manage, deployment):
    path, profile, binary = deployment
    assert manage.profile(path) == profile
    binary.write_bytes(b"changed compiled extension")
    with pytest.raises(ValueError, match="compiled extension changed"):
        manage.profile(path)


def test_engine_evidence_binds_model_environment_and_all_five_sources(
    manage, deployment
):
    _, p, _ = deployment
    paths = {
        rel.removeprefix("vllm/v1/core/").removesuffix(".py").replace("/", "."): str(
            Path(p["overlay_site"]) / rel
        )
        for rel in manage.CHANGED
    }
    evidence = {
        **p,
        "version": "0.29.0",
        "dtype": "torch.float16",
        "cuda_visible_devices": "0",
        "prefix_caching": True,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_concurrent_batches": 1,
        "use_flashinfer_sampler": False,
        "core_paths": paths,
        "core_sha256": {name: manage.sha(path) for name, path in paths.items()},
    }
    manage.verify_engine_profile(p, evidence)
    for field, wrong in (
        ("python", "/different/python"),
        ("model", "/different/model"),
        ("gpu_memory_utilization", 0.9),
        ("core_paths", {}),
        ("use_flashinfer_sampler", True),
        ("use_flashinfer_sampler", None),
        ("use_flashinfer_sampler", "0"),
        ("use_flashinfer_sampler", 0),
    ):
        with pytest.raises(RuntimeError):
            manage.verify_engine_profile(p, {**evidence, field: wrong})


def test_stop_checks_surviving_group_members_and_refuses_foreign_processes(
    manage, monkeypatch, tmp_path
):
    proc = tmp_path / "proc"
    leader, child = proc / "100", proc / "101"
    leader.mkdir(parents=True)
    child.mkdir()
    (leader / "stat").write_text("100 (api) Z other fields")
    (child / "stat").write_text("101 (worker) S other fields")
    (child / "environ").write_bytes(b"VLLM_PE_LAUNCH_TOKEN=ours\0")
    monkeypatch.setattr(
        manage, "Path", lambda value: proc if value == "/proc" else Path(value)
    )
    monkeypatch.setattr(manage.os, "getpgid", lambda pid: 100)
    sent = []

    def terminate(group, signum):
        sent.append((group, signum))
        (child / "stat").write_text("101 (worker) Z other fields")

    monkeypatch.setattr(manage.os, "killpg", terminate)
    record = {"pid": 100, "launch_token": "ours"}
    assert manage.owned_group_members(record) == [101]
    (child / "environ").write_bytes(b"VLLM_PE_LAUNCH_TOKEN=someone-else\0")
    with pytest.raises(RuntimeError, match="refusing to signal"):
        manage.stop_record(record)
    assert sent == []
    (child / "environ").write_bytes(b"VLLM_PE_LAUNCH_TOKEN=ours\0")
    leader.rename(proc / "leader-no-longer-a-pid")
    manage.stop_record(record)
    assert sent == [(100, manage.signal.SIGTERM)]


def test_start_refuses_occupied_port_before_spawning_or_contacting_http(
    manage, monkeypatch, tmp_path
):
    monkeypatch.setattr(manage, "profile", lambda _: {"gpu": "0", "port": 18080})
    monkeypatch.setattr(manage.platform, "system", lambda: "Linux")
    monkeypatch.setattr(manage.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        manage, "command", lambda cmd: "RTX 5090" if "--query-gpu=name" in cmd else ""
    )

    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def bind(self, address):
            raise OSError("Address already in use")

    monkeypatch.setattr(manage.socket, "socket", lambda *_: OccupiedSocket())
    monkeypatch.setattr(
        manage.subprocess, "Popen", lambda *_a, **_k: pytest.fail("must not launch")
    )
    monkeypatch.setattr(
        manage.LOCAL_HTTP,
        "open",
        lambda *_a, **_k: pytest.fail("must not contact existing service"),
    )
    with pytest.raises(OSError, match="already in use"):
        manage.start(SimpleNamespace(profile="unused", run_dir=str(tmp_path / "run")))


@pytest.mark.parametrize("mode", ["native", "progressive"])
def test_start_passes_sampler_disabled_to_service_when_parent_enabled(
    manage, deployment, monkeypatch, tmp_path, mode
):
    profile_path, _, _ = deployment
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.setattr(manage.platform, "system", lambda: "Linux")
    monkeypatch.setattr(manage.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        manage, "command", lambda cmd: "RTX 5090" if "--query-gpu=name" in cmd else ""
    )
    monkeypatch.setattr(
        manage.socket,
        "socket",
        lambda *_: nullcontext(SimpleNamespace(bind=lambda _: None)),
    )
    captured = {}

    def service_process(command, **kwargs):
        captured.update(command=command, **kwargs)
        raise RuntimeError("stop after capturing service Popen arguments")

    monkeypatch.setattr(manage.subprocess, "Popen", service_process)
    with pytest.raises(RuntimeError, match="capturing service Popen"):
        manage.start(
            SimpleNamespace(
                profile=str(profile_path),
                policy=mode,
                run_id="sampler-test",
                run_dir=str(tmp_path / "run"),
                start_timeout=1,
            )
        )
    assert captured["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert captured["env"]["VLLM_PE_POLICY"] == mode
    assert captured["start_new_session"] is True
    assert manage.os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


@pytest.mark.parametrize("raw", [None, "1", "false"])
def test_direct_runtime_requires_explicit_sampler_disable(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", raw)
    with pytest.raises(ValueError, match="VLLM_USE_FLASHINFER_SAMPLER=0"):
        load_runtime()._configuration(SimpleNamespace(), "progressive", "/unused")


def test_runtime_checks_resolved_sampler_and_records_boolean(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.29.0"
    vllm.envs = SimpleNamespace(VLLM_USE_FLASHINFER_SAMPLER=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    runtime = load_runtime()
    with pytest.raises(ValueError, match="resolved enabled"):
        runtime._configuration(SimpleNamespace(), "progressive", str(tmp_path))
    spec = type("FullAttentionSpec", (), {"block_size": 16})()
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(
                async_scheduling=False,
                max_num_seqs=8,
                max_num_batched_tokens=1024,
                max_num_scheduled_tokens=1024,
                enable_chunked_prefill=True,
            ),
            cache_config=SimpleNamespace(
                enable_prefix_caching=True,
                cache_dtype="auto",
                gpu_memory_utilization=0.8,
            ),
            model_config=SimpleNamespace(
                enforce_eager=True,
                model="local-model",
                max_model_len=2048,
                dtype="float16",
                seed=17,
            ),
            attention_config=SimpleNamespace(backend="FLASH_ATTN"),
            max_concurrent_batches=1,
            kv_transfer_config=None,
            ec_transfer_config=None,
            speculative_config=None,
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
            num_blocks=200,
        ),
        hash_block_size=16,
        is_encoder_decoder=False,
        is_mm_encoder_only=False,
    )
    vllm.envs.VLLM_USE_FLASHINFER_SAMPLER = False
    config = runtime._configuration(scheduler, "native", str(tmp_path))
    assert config["use_flashinfer_sampler"] is False


@pytest.mark.parametrize("problem", ["missing", "session_count", "input_budget"])
def test_paired_rejects_manifest_before_any_gpu_start(
    manage, monkeypatch, tmp_path, problem
):
    sys.path.insert(0, str(manage.ROOT / "experiment"))
    import replay

    monkeypatch.setattr(manage, "profile", lambda _: {"max_model_len": 2048})
    monkeypatch.setattr(manage, "start", lambda *_: pytest.fail("GPU must not start"))
    manifest = tmp_path / "manifest_50.json"
    if problem != "missing":
        count = 8 if problem == "session_count" else 50
        manifest.write_text(
            json.dumps(
                {
                    "schema": replay.SCHEMA,
                    "output_tokens": 128,
                    "selected_sessions": count,
                    "sessions": [{} for _ in range(count)],
                    "max_model_len": 4096,
                }
            )
        )
    args = SimpleNamespace(
        profile="unused",
        manifest=str(manifest),
        concurrency=4,
        out=str(tmp_path / "paired"),
        seeds=[17, 29, 43],
        start_timeout=1,
    )
    with pytest.raises((FileNotFoundError, ValueError)) as exc:
        manage.paired(args)
    if problem == "missing":
        assert "experiment/replay.py prepare" in str(exc.value)
        assert "--sessions 50" in str(exc.value)
        assert "--max-model-len 2048" in str(exc.value)
    elif problem == "session_count":
        assert "expected 50" in str(exc.value)
    else:
        assert "max_model_len differs" in str(exc.value)
    assert not Path(args.out).exists()
