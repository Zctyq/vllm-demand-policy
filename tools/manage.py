#!/usr/bin/env python
"""Isolated deployment helpers. Never connects to SSH or changes installed vLLM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
CORE = [
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/request.py",
    "vllm/v1/engine/input_processor.py",
    "vllm/entrypoints/openai/chat_completion/serving.py",
    "vllm/entrypoints/serve/engine/serving.py",
]
CHANGED = CORE[:3] + [
    "vllm/v1/core/progressive_policy.py",
    "vllm/v1/core/progressive_runtime.py",
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def audit(args):
    source = Path(args.source).resolve()
    reference = ROOT / "reference/vllm-v0.29.0"
    rows = []
    for rel in CORE:
        original, ref = source / rel, reference / rel
        rows.append(
            {
                "path": rel,
                "sha256": sha(original) if original.exists() else None,
                "official_tag_sha256": sha(ref) if ref.exists() else None,
                "matches_reference": original.exists()
                and ref.exists()
                and original.read_bytes() == ref.read_bytes(),
            }
        )
    packages = {}
    for name in ("vllm", "torch", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader",
            ]
        )
    result = {
        "recorded_date_client_timezone": "2026-10-03 Asia/Shanghai",
        "host_system": platform.system(),
        "machine": platform.machine(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages_in_this_interpreter": packages,
        "local_source": str(source),
        "local_source_has_git": (source / ".git").exists(),
        "reference_tag": "v0.29.0",
        "reference_commit": "98dff2a81d747d1dba01a47f939f48c3526d4206",
        "core_files": rows,
        "gpu": gpu,
        "source_match_is_not_installed_build_verification": True,
        "remote_server_contacted_for_this_local_phase": False,
        "outside_this_environment_audit": [
            "installed GPU vLLM build and extension ABI",
            "running EngineCore import paths",
            "running GPU model path/configuration",
            "data/tokenizer checks are recorded separately in export_audit/audit.json",
            "GPU pool block count and deployed launch arguments",
        ],
    }
    dump(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare_overlay(args):
    """Run with the existing GPU environment's Python, without pip install."""
    import vllm

    if vllm.__version__ != "0.29.0":
        raise RuntimeError(f"Expected installed 0.29.0, found {vllm.__version__}")
    installed = Path(vllm.__file__).resolve().parent
    base = json.loads((ROOT / "results/local_environment.json").read_text())
    failures = []
    hashes = {}
    for row in base["core_files"]:
        rel = row["path"]
        path = installed.parent / rel
        actual = sha(path) if path.exists() else None
        hashes[rel] = actual
        if actual != row["sha256"] or not row["matches_reference"]:
            failures.append(rel)
    if failures:
        raise RuntimeError(
            "Installed Python source differs; refuse overlay: " + ", ".join(failures)
        )
    dest = Path(args.dest).resolve()
    if dest.exists():
        raise FileExistsError(f"Choose a new empty overlay destination: {dest}")
    site = dest / "site-packages"
    site.mkdir(parents=True)
    shutil.copytree(installed, site / "vllm", symlinks=False)
    if (installed.parent / "vllm.libs").exists():
        shutil.copytree(installed.parent / "vllm.libs", site / "vllm.libs")
    for rel in CHANGED:
        shutil.copy2(ROOT / "worktree" / rel, site / rel)
    binaries = {}
    for path in (site / "vllm").rglob("*.so"):
        rel = path.relative_to(site)
        original = installed.parent / rel
        binaries[str(rel)] = sha(path)
        if not original.exists() or sha(original) != binaries[str(rel)]:
            raise RuntimeError(f"Compiled extension copy mismatch: {rel}")
    if not binaries:
        raise RuntimeError(
            "No installed compiled .so extensions found; refuse GPU overlay"
        )
    manifest = {
        "python": sys.executable,
        "version": vllm.__version__,
        "original_package": str(installed),
        "site_packages": str(site),
        "base_source_sha256": hashes,
        "patched_sha256": {rel: sha(site / rel) for rel in CHANGED},
        "compiled_extensions_unchanged": binaries,
    }
    env = dict(os.environ, PYTHONPATH=str(site))
    probe = command(
        [
            sys.executable,
            "-c",
            (
                "import json,vllm; from vllm.v1.core import block_pool, "
                "kv_cache_manager,progressive_runtime; "
                "from vllm.v1.core.sched import scheduler; "
                "print(json.dumps({'vllm':vllm.__file__,'scheduler':scheduler.__file__,"
                "'block_pool':block_pool.__file__,"
                "'kv_cache_manager':kv_cache_manager.__file__,"
                "'runtime':progressive_runtime.__file__}))"
            ),
        ],
        env=env,
    )
    imported = json.loads(probe.splitlines()[-1])
    if not all(Path(value).is_relative_to(site) for value in imported.values()):
        raise RuntimeError("Full-package overlay import proof failed")
    manifest["import_probe"] = imported
    dump(dest / "overlay_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


PROFILE_FIELDS = (
    "python",
    "overlay_site",
    "model",
    "served_model_name",
    "gpu",
    "port",
    "dtype",
    "max_model_len",
    "block_size",
    "num_gpu_blocks",
    "max_num_batched_tokens",
    "max_num_seqs",
    "enable_chunked_prefill",
    "attention_backend",
    "seed",
    "gpu_memory_utilization",
)


def profile(path):
    value = json.loads(Path(path).read_text())
    missing = [key for key in PROFILE_FIELDS if key not in value or value[key] is None]
    if missing:
        raise ValueError(
            "Fill verified existing deployment values: " + ", ".join(missing)
        )
    if "," in str(value["gpu"]):
        raise ValueError("Select exactly one RTX 5090 GPU")
    if "FLASHINFER" in value["attention_backend"].upper():
        raise ValueError("FlashInfer is disabled for this experiment")
    if value["max_num_seqs"] < 2:
        raise ValueError("Do not serialize the engine with max_num_seqs=1")
    if type(value["port"]) is not int or not 1024 <= value["port"] <= 65535:
        raise ValueError("Use a dedicated unprivileged port in [1024,65535]")
    for key in (
        "block_size",
        "num_gpu_blocks",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < value["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0,1]")
    if type(value["enable_chunked_prefill"]) is not bool:
        raise ValueError("enable_chunked_prefill must be boolean")
    for key in ("python", "model", "overlay_site"):
        if not Path(value[key]).exists():
            raise FileNotFoundError(f"{key}: {value[key]}")
    manifest_path = Path(value["overlay_site"]).parent / "overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    site = Path(value["overlay_site"]).resolve()
    if manifest.get("version") != "0.29.0" or (
        Path(manifest.get("site_packages", "")).resolve() != site
    ):
        raise ValueError("Overlay manifest does not describe this v0.29.0 package")
    if os.path.abspath(value["python"]) != os.path.abspath(manifest["python"]):
        raise ValueError("Use the same environment Python that prepared the overlay")
    if set(manifest.get("patched_sha256", {})) != set(CHANGED):
        raise ValueError("Overlay manifest is missing required patched source files")
    for rel, digest in manifest["patched_sha256"].items():
        if sha(site / rel) != digest:
            raise ValueError(f"Overlay source changed after preparation: {rel}")
    binaries = manifest.get("compiled_extensions_unchanged", {})
    if not binaries:
        raise ValueError("Overlay manifest has no compiled extension fingerprints")
    for rel, digest in binaries.items():
        if sha(site / rel) != digest:
            raise ValueError(f"Overlay compiled extension changed: {rel}")
    return value


def verify_engine_profile(p, evidence):
    """Validate the child EngineCore, not just the API parent or CLI flags."""
    if evidence.get("use_flashinfer_sampler") is not False:
        raise RuntimeError("EngineCore FlashInfer sampler must be explicitly disabled")
    for name in (
        "num_gpu_blocks",
        "block_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "seed",
    ):
        if evidence.get(name) != p[name]:
            raise RuntimeError(f"EngineCore actual {name} differs from profile")
    if evidence.get("version") != "0.29.0":
        raise RuntimeError("EngineCore version differs from checked source")
    if os.path.abspath(evidence.get("python", "")) != os.path.abspath(p["python"]):
        raise RuntimeError("EngineCore Python differs from the overlay environment")
    if Path(evidence.get("model", "")).resolve() != Path(p["model"]).resolve():
        raise RuntimeError("EngineCore model differs from profile")
    if evidence.get("gpu_memory_utilization") != p["gpu_memory_utilization"]:
        raise RuntimeError("EngineCore GPU memory configuration differs from profile")
    if evidence.get("cuda_visible_devices") != str(p["gpu"]):
        raise RuntimeError("EngineCore GPU selection differs from profile")
    if not (
        evidence.get("prefix_caching") is True
        and evidence.get("enforce_eager") is True
        and evidence.get("async_scheduling") is False
        and evidence.get("max_concurrent_batches") == 1
    ):
        raise RuntimeError("EngineCore did not use the required experiment settings")
    aliases = {"half": "float16", "float": "float32"}
    requested_dtype = aliases.get(p["dtype"], p["dtype"])
    if evidence.get("dtype", "").removeprefix("torch.") != requested_dtype:
        raise RuntimeError("Profile must specify the verified resolved dtype")
    backend = str(evidence.get("attention_backend", "")).split(".")[-1]
    if backend != p["attention_backend"]:
        raise RuntimeError("EngineCore attention backend differs from profile")
    site = Path(p["overlay_site"]).resolve()
    expected_names = {
        "sched.scheduler",
        "kv_cache_manager",
        "block_pool",
        "progressive_policy",
        "progressive_runtime",
    }
    if set(evidence.get("core_paths", {})) != expected_names:
        raise RuntimeError("Incomplete EngineCore import proof")
    for name, imported in evidence.get("core_paths", {}).items():
        expected = site / "vllm/v1/core" / (name.replace(".", "/") + ".py")
        if not imported or Path(imported).resolve() != expected:
            raise RuntimeError(f"EngineCore did not import selected overlay: {name}")
        if evidence.get("core_sha256", {}).get(name) != sha(expected):
            raise RuntimeError(f"EngineCore source hash mismatch: {name}")


def server_command(p, launch_alias=None):
    cmd = [
        p["python"],
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        p["model"],
        "--served-model-name",
        p["served_model_name"],
        "--host",
        "127.0.0.1",
        "--port",
        str(p["port"]),
        "--dtype",
        p["dtype"],
        "--max-model-len",
        str(p["max_model_len"]),
        "--block-size",
        str(p["block_size"]),
        "--num-gpu-blocks-override",
        str(p["num_gpu_blocks"]),
        "--max-num-batched-tokens",
        str(p["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(p["max_num_seqs"]),
        "--attention-backend",
        p["attention_backend"],
        "--gpu-memory-utilization",
        str(p["gpu_memory_utilization"]),
        "--seed",
        str(p["seed"]),
        "--enable-prefix-caching",
        "--enforce-eager",
        "--no-async-scheduling",
        "--enable-chunked-prefill"
        if p["enable_chunked_prefill"]
        else "--no-enable-chunked-prefill",
    ]
    if launch_alias:
        cmd.insert(cmd.index("--host"), launch_alias)
    return cmd


def start(args):
    p = profile(args.profile)
    if platform.system() != "Linux" or not shutil.which("nvidia-smi"):
        raise RuntimeError(
            "GPU launch requires local Linux NVIDIA runtime; "
            "no remote connection is attempted"
        )
    gpu_name = command(
        ["nvidia-smi", "-i", str(p["gpu"]), "--query-gpu=name", "--format=csv,noheader"]
    )
    if "5090" not in gpu_name or "\n" in gpu_name:
        raise RuntimeError(f"Selected device is not one RTX 5090: {gpu_name}")
    busy = command(
        [
            "nvidia-smi",
            "-i",
            str(p["gpu"]),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ]
    )
    if busy:
        raise RuntimeError(
            "Selected GPU has an existing compute process; it was not stopped"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", p["port"]))
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    engine_logs = run / "engine"
    launch_token = secrets.token_hex(16)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VLLM_PE_"):
            del env[key]
    env.update(
        PYTHONPATH=str(Path(p["overlay_site"]).resolve()),
        CUDA_VISIBLE_DEVICES=str(p["gpu"]),
        VLLM_PE_POLICY=args.policy,
        VLLM_PE_RUN_ID=args.run_id,
        VLLM_PE_LOG_DIR=str(engine_logs),
        VLLM_PE_LAUNCH_TOKEN=launch_token,
        VLLM_SERVER_DEV_MODE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
    )
    launch_alias = "pe-instance-" + launch_token
    cmd = server_command(p, launch_alias)
    with (run / "server.log").open("wb") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    record = {
        "pid": proc.pid,
        "launch_token": launch_token,
        "run_dir": str(run),
        "run_id": args.run_id,
        "policy": args.policy,
        "launch_alias": launch_alias,
        "command": cmd,
        "profile": p,
        "engine_log_dir": str(engine_logs),
    }
    dump(run / "process.json", record)
    deadline = time.monotonic() + args.start_timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Experiment server exited {proc.returncode}; "
                    f"inspect {run / 'server.log'}"
                )
            try:
                with LOCAL_HTTP.open(
                    f"http://127.0.0.1:{p['port']}/health", timeout=2
                ) as response:
                    ready = response.status == 200
                if ready and (engine_logs / "run_config.json").exists():
                    with LOCAL_HTTP.open(
                        f"http://127.0.0.1:{p['port']}/v1/models", timeout=2
                    ) as response:
                        model_ids = {
                            item["id"] for item in json.load(response).get("data", [])
                        }
                    if launch_alias not in model_ids:
                        raise RuntimeError("Port is served by a different API process")
                    if proc.poll() is not None:
                        raise RuntimeError(
                            "Experiment API process exited during startup"
                        )
                    evidence = json.loads((engine_logs / "run_config.json").read_text())
                    if (
                        evidence["num_gpu_blocks"] != p["num_gpu_blocks"]
                        or evidence["block_size"] != p["block_size"]
                    ):
                        raise RuntimeError("Actual pool differs from pinned profile")
                    if evidence["mode"] != args.policy:
                        raise RuntimeError("EngineCore imported wrong policy")
                    if evidence.get("run_id") != args.run_id:
                        raise RuntimeError("EngineCore run ID differs from this launch")
                    verify_engine_profile(p, evidence)
                    print(json.dumps(record, indent=2))
                    return record
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(0.5)
        raise TimeoutError("Experiment server did not become ready; inspect server.log")
    except BaseException:
        stop_record(record)
        raise


def owned_group_members(record):
    """Find live experiment members even after the API group leader exits."""
    pid = record["pid"]
    if type(pid) is not int or pid <= 1 or not record.get("launch_token"):
        raise ValueError("Invalid experiment process identity")
    marker = ("VLLM_PE_LAUNCH_TOKEN=" + record["launch_token"]).encode()
    members = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdecimal():
            continue
        member = int(proc_dir.name)
        try:
            if os.getpgid(member) != pid:
                continue
            state = (proc_dir / "stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state == "Z":
                continue
            environment = (proc_dir / "environ").read_bytes().split(b"\0")
            if marker not in environment:
                raise RuntimeError(
                    "Process group identity changed; refusing to signal another process"
                )
            members.append(member)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop_record(record):
    pid = record["pid"]
    if not owned_group_members(record):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not owned_group_members(record):
            return
        time.sleep(0.2)
    if owned_group_members(record):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def paired(args):
    sys.path.insert(0, str(ROOT / "experiment"))
    from replay import validate_manifest
    from summarize import plan

    p = profile(args.profile)
    validate_manifest(
        args.manifest,
        expected_sessions=50,
        concurrency=args.concurrency,
        expected_max_model_len=p["max_model_len"],
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    schedule = plan(args.seeds)
    dump(out / "plan.json", schedule)
    runs = []
    for item in schedule:
        run_id = f"seed{item['seed']}-{item['policy']}"
        run = out / run_id
        start_args = argparse.Namespace(
            profile=args.profile,
            policy=item["policy"],
            run_id=run_id,
            run_dir=str(run),
            start_timeout=args.start_timeout,
        )
        record = start(start_args)
        try:
            cmd = [
                p["python"],
                str(ROOT / "experiment/replay.py"),
                "run",
                "--manifest",
                str(Path(args.manifest).resolve()),
                "--base-url",
                f"http://127.0.0.1:{p['port']}",
                "--model",
                record["launch_alias"],
                "--out",
                str(run / "client"),
                "--engine-log-dir",
                record["engine_log_dir"],
                "--policy",
                item["policy"],
                "--seed",
                str(item["seed"]),
                "--concurrency",
                str(args.concurrency),
                "--expected-block-size",
                str(p["block_size"]),
                "--expected-num-gpu-blocks",
                str(p["num_gpu_blocks"]),
                "--run-id",
                run_id,
            ]
            with (run / "client.log").open("w") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            runs.append(str(run / "client"))
        finally:
            stop_record(record)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiment/summarize.py"),
            "summarize",
            *runs,
            "--out",
            str(out / "paired_results.json"),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--source", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("overlay")
    a.add_argument("--dest", required=True)
    a = sub.add_parser("start")
    a.add_argument("--profile", required=True)
    a.add_argument("--policy", choices=("native", "progressive"), required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--run-dir", required=True)
    a.add_argument("--start-timeout", type=float, default=900)
    a = sub.add_parser("stop")
    a.add_argument("--record", required=True)
    a = sub.add_parser("paired")
    a.add_argument("--profile", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--concurrency", type=int, default=4)
    a.add_argument("--seeds", nargs=3, type=int, default=[17, 29, 43])
    a.add_argument("--start-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args)
    elif args.action == "overlay":
        prepare_overlay(args)
    elif args.action == "start":
        start(args)
    elif args.action == "stop":
        stop_record(json.loads(Path(args.record).read_text()))
    else:
        if args.concurrency not in (4, 8, 16):
            parser.error("Choose calibrated session concurrency 4, 8, or 16")
        paired(args)


if __name__ == "__main__":
    main()
