"""Read-only installed source comparison; does not import the vLLM engine."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path,
                    help="directory containing vllm/, optional for installed package")
args = parser.parse_args()
if args.source_root:
    source = args.source_root.resolve()
else:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm not found; nothing modified")
    source = Path(spec.origin).resolve().parent.parent
try:
    version = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    version = "not installed in this interpreter"
expected = json.loads((ROOT / "results/demand/source-versions.json").read_text())
matches, differences = [], {}
for label, files in expected.items():
    differences[label] = []
    for rel, digest in files.items():
        path = source / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != digest:
            differences[label].append({"path": rel, "actual_sha256": actual,
                                       "expected_sha256": digest})
    if not differences[label]:
        matches.append(label)
print(json.dumps({"python": sys.executable, "installed_version": version,
                  "source_root": str(source), "matches": matches,
                  "differences": differences,
                  "switches": {key: os.getenv(key) for key in (
                      "VLLM_WAITING_HIT_ASC", "VLLM_DEMAND_EVICTION",
                      "VLLM_DEMAND_TELEMETRY")}}, indent=2))
raise SystemExit(0 if matches else 1)
