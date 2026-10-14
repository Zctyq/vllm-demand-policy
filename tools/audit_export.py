#!/usr/bin/env python3
"""Audit a local trace export against its original tokenizer and manifest.

No model weights, dataset downloads, model execution, or HTTP requests are used.
Only the four reviewed pure tokenization helpers are loaded from the old script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def helpers(path):
    names = {"clean_messages", "template_messages", "encode", "hash_ids"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("original script lacks the four reviewed pure helpers")
    module = ast.Module(body=functions, type_ignores=[])
    scope = {"json": json, "hashlib": hashlib}
    exec(compile(module, str(path), "exec"), scope)
    return scope


def common_prefix(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def audit(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    start = time.monotonic()
    root = Path(args.export_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "requests.jsonl",
        "original_manifest.json",
        "export_info.json",
        "replay_concurrent.py",
    ]
    sources = [root / name for name in names] + sorted((root / "tokenizer").glob("*"))
    fingerprints = {
        str(p.relative_to(root)): sha_file(p) for p in sources if p.is_file()
    }
    manifest = json.loads((root / "original_manifest.json").read_text())
    info = json.loads((root / "export_info.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    baseline = helpers(root / "replay_concurrent.py")
    new_source = Path(__file__).resolve().parents[1] / "experiment" / "replay.py"
    spec = importlib.util.spec_from_file_location("pe_export_replay", new_source)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    tokenizer = AutoTokenizer.from_pretrained(
        str(root / "tokenizer"), local_files_only=True
    )
    expected = {}
    for session in manifest["sessions"]:
        for call in session["calls"]:
            key = (session["session_id"], call["call"])
            if key in expected:
                raise ValueError("duplicate session/call in original manifest")
            expected[key] = call
    actual_keys = [(row["session_id"], row["call"]) for row in rows]
    if len(set(actual_keys)) != len(rows) or set(actual_keys) != set(expected):
        raise ValueError(
            "export session/call membership differs from original manifest"
        )
    counts = Counter(row["session_id"] for row in rows)
    if not (
        len(rows) == manifest["selected_calls"] == info["requests"]
        and len(counts) == manifest["selected_sessions"] == info["sessions"]
    ):
        raise ValueError("export/manifest/info counts disagree")
    previous, per_request, errors = {}, [], []
    for index, row in enumerate(rows):
        sid, call = row["session_id"], row["call"]
        reference = expected[sid, call]
        try:
            prev_call, prev_ids = previous.get(sid, (0, []))
            if call != prev_call + 1:
                raise ValueError("export rows are not in contiguous session call order")
            old_ids = baseline["encode"](
                tokenizer, baseline["clean_messages"](row["input"])
            )
            new_ids = replay.encode(tokenizer, replay.clean_messages(row["input"]))
            digest = replay.hash_ids(new_ids)
            raw_gap = float(row["pre_gap"])
            if (
                isinstance(row["pre_gap"], bool)
                or not math.isfinite(raw_gap)
                or raw_gap < 0
            ):
                raise ValueError("invalid original pre_gap")
            gap = 0.0 if call == 1 else raw_gap
            lcp = common_prefix(prev_ids, new_ids)
            raw_output = row["output_length"]
            checks = {
                "manifest_dataset_row_match": row["original_dataset_row"]
                == reference["dataset_row"],
                "old_new_token_ids": old_ids == new_ids,
                "original_hash_algorithm": baseline["hash_ids"](old_ids) == digest,
                "manifest_token_hash": digest == reference["prompt_sha256"],
                "manifest_token_count": len(new_ids) == reference["prompt_tokens"],
                "manifest_adjacent_lcp": lcp == reference["lcp_tokens"],
                "manifest_pre_gap": gap == reference["pre_gap"],
                "manifest_original_output": (
                    not isinstance(raw_output, bool)
                    and int(raw_output) == raw_output
                    and raw_output >= 0
                    and max(1, int(raw_output)) == reference["output_budget"]
                ),
                "fixed_128_context_fits": len(new_ids) + 128
                <= manifest["max_model_len"],
            }
            record = {
                "export_row": index,
                "session_id": sid,
                "call": call,
                "original_dataset_row": row["original_dataset_row"],
                "input_tokens": len(new_ids),
                "prompt_sha256": digest,
                "lcp_tokens": lcp,
                "pre_gap_seconds_in_original_replay": gap,
                "original_output_budget": reference["output_budget"],
                **checks,
            }
            per_request.append(record)
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append({"session_id": sid, "call": call, "failed": failed})
            previous[sid] = (call, new_ids)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"session_id": sid, "call": call, "error": str(exc)})
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(
                f"Checked {index + 1}/{len(rows)} calls; mismatches={len(errors)}",
                flush=True,
            )
    for name, digest in fingerprints.items():
        if sha_file(root / name) != digest:
            raise ValueError(f"input changed during audit: {name}")
    if per_request:
        with (out / "token_checks.csv").open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(dest, fieldnames=list(per_request[0]))
            writer.writeheader()
            writer.writerows(per_request)
    lengths = [row["input_tokens"] for row in per_request]
    packages = {
        name: importlib.metadata.version(name)
        for name in ("transformers", "tokenizers", "jinja2", "huggingface-hub")
    }
    report = {
        "status": "passed"
        if not errors and len(per_request) == len(rows)
        else "failed",
        "scope": (
            "local CPU tokenization and export consistency; "
            "not EngineCore or GPU inference"
        ),
        "export_dir": str(root),
        "source_sha256": fingerprints,
        "replay_source_sha256": sha_file(new_source),
        "audit_script_sha256": sha_file(Path(__file__)),
        "python": sys.executable,
        "python_version": sys.version,
        "local_packages": packages,
        "export_reported_environment": info["export_environment"],
        "tokenizer_versions_match_export": all(
            packages[name] == info["export_environment"][name]
            for name in ("transformers", "tokenizers")
        ),
        "sessions": len(counts),
        "requests": len(rows),
        "verified_requests": len(per_request),
        "calls_per_session_distribution": dict(
            sorted(Counter(counts.values()).items())
        ),
        "eligible_sessions_10_to_12_calls": sum(
            10 <= count <= 12 for count in counts.values()
        ),
        "mean_calls_per_session": statistics.mean(counts.values()),
        "input_token_range": [min(lengths), max(lengths)] if lengths else None,
        "total_input_tokens": sum(lengths),
        "max_model_len_from_original_manifest": manifest["max_model_len"],
        "output_tokens_per_request_for_new_experiment": 128,
        "planned_total_output_tokens": len(rows) * 128,
        "original_total_output_budget": sum(
            c["output_budget"] for c in expected.values()
        ),
        "gap_unit_used_by_original_replay": "seconds",
        "gap_semantics_used_by_original_replay": (
            "previous response completes, then wait pre_gap; first call immediate"
        ),
        "source_gap_unit_independently_verified": False,
        "gap_evidence_limitation": (
            "export has no upstream timing schema/timestamps; "
            "original replay code is the available semantic evidence"
        ),
        "gap_sum_excluding_first_calls": sum(
            r["pre_gap_seconds_in_original_replay"] for r in per_request
        ),
        "mismatches": errors,
        "gpu_executed": False,
        "http_requests_sent": 0,
        "elapsed_seconds": time.monotonic() - start,
    }
    replay.dump(out / "audit.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "sessions",
                    "requests",
                    "verified_requests",
                    "input_token_range",
                    "tokenizer_versions_match_export",
                    "eligible_sessions_10_to_12_calls",
                    "elapsed_seconds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--out", required=True)
    raise SystemExit(audit(parser.parse_args()))
