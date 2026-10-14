"""Two optimized runs, retaining every previously measured run."""

import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import run_comparison as runner

args = SimpleNamespace(
    manifest=str(
        runner.PROJECT / "results/stratified10-seed17-i28ico0v/manifest_10.json"
    ),
    concurrency=4,
)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    args.port = sock.getsockname()[1]
config = json.loads(
    (runner.PROJECT / "configs/server.gpu1.calibration.json").read_text()
)
manifest = runner.replay.validate_manifest(
    args.manifest, expected_sessions=10, concurrency=args.concurrency
)
deployment = json.loads((runner.ROOT / "deployment.json").read_text())
processes = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        str(config["gpu"]),
        "--query-compute-apps=process_name",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
if any(
    Path(name.strip()).name != "nvidia-cuda-mps-server"
    for name in processes.splitlines()
):
    raise RuntimeError("selected GPU has another compute application")
idle = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        str(config["gpu"]),
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip()
memory, utilization = [int(x.strip()) for x in idle.split(",")]
if memory > 128 or utilization != 0:
    raise RuntimeError("selected GPU is not idle")
runner.write(
    runner.ROOT / "optimized-preflight.json",
    {
        "processes": processes,
        "memory_mib": memory,
        "utilization_percent": utilization,
        "mps_clients_verified": False,
        "existing_mps_left_running": True,
        "port": args.port,
    },
)
for repeat in [1, 2]:
    runner.run("optimized", repeat, args, config, manifest, deployment)
print("OPTIMIZED_RUNS_COMPLETE", flush=True)
