#!/usr/bin/env python3
"""Fixed-workload, causal Agent replay for the progressive-eviction experiment.

CPU prepare uses the deployed tokenizer without truncation. Runtime HTTP calls are
made only by the explicit `run` command. Adapted from the user's existing
replay_concurrent_test_50tasks.py: message normalization, token hash validation,
dataset grouping and stdlib HTTP are retained. No global prefix-hit metric is used
as the historical reuse denominator; H and K must come from the instrumented
engine's request records. No generated model text feeds into later trace inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import shlex
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

SCHEMA = "progressive-replay-1"
OUTPUT_TOKENS = 128
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_ids(ids):
    return sha_bytes(json.dumps(ids, separators=(",", ":")).encode())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_concurrency(concurrency, session_count):
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("active-session concurrency must be a positive integer")
    if concurrency > session_count:
        raise ValueError(
            f"active-session cap C={concurrency} exceeds {session_count} sessions; "
            f"use at least {concurrency} complete sessions. For C=16, run: "
            '"$PE_PYTHON" experiment/replay.py subset '
            "--manifest results/manifest_50.json --sessions 16 "
            "--seed 20261001 --out results/calibration_16.json"
        )


def manifest_preparation_hint(path, expected_sessions=None, max_model_len=None):
    path = Path(path)
    match = re.fullmatch(r"calibration_(8|16)", path.stem)
    count = expected_sessions or (int(match[1]) if match else 50)
    parent = path.parent / "manifest_50.json" if count < 50 else path
    length = str(max_model_len) if max_model_len else "32768"
    prepare_command = (
        '"$PE_PYTHON" experiment/replay.py prepare '
        '--data "$PE_EXPORT/requests.jsonl" --tokenizer "$PE_EXPORT/tokenizer" '
        '--reference-manifest "$PE_EXPORT/original_manifest.json" '
        f"--manifest {shlex.quote(str(parent))} --max-model-len {length} "
        "--gap-unit s --gap-evidence docs/pre-gap-evidence.txt "
        "--order-field call --sessions 50 --min-calls 5 --max-calls 50"
    )
    if count == 50:
        return prepare_command
    subset_command = (
        '"$PE_PYTHON" experiment/replay.py subset '
        f"--manifest {shlex.quote(str(parent))} --sessions {count} "
    )
    commands = (
        "Prepare the parent first if it is missing:\n"
        + prepare_command
        + "\nFor a rebuilt parent, create a new selection record:\n"
        + subset_command
        + f"--seed 20261001 --out {shlex.quote(str(path))}"
    )
    if count == 8:
        commands += (
            "\nAlternatively, only if the original parent's SHA matches the "
            "existing selection record, reproduce its IDs with:\n"
            + subset_command
            + "--selection "
            + shlex.quote(str(path.parent / "calibration_selection.json"))
            + f" --out {shlex.quote(str(path))}"
        )
    return commands


def manifest_order_sha256(manifest):
    """Hash ordered request identities and token digests without a tokenizer."""
    ordered = [
        [
            session["session_id"],
            [
                [
                    call["call_index"],
                    call["dataset_row"],
                    call.get("original_dataset_row"),
                    call.get("source_call"),
                    call["input_tokens"],
                    call["prompt_sha256"],
                    call["pre_gap_seconds"],
                ]
                for call in session["calls"]
            ],
        ]
        for session in manifest["sessions"]
    ]
    return sha_bytes(
        json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()
    )


def validate_manifest(
    path, *, expected_sessions=None, concurrency=None, expected_max_model_len=None
):
    """Validate a local prepared snapshot before GPU startup or data packaging.

    This checks stored tokens/hashes and call order; it does not retokenize text or
    contact a service. Old prepared snapshots without an order digest remain valid;
    their per-request token digests and explicit call indexes are always checked.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared manifest is missing: {path}\n"
            "Set PE_PYTHON to the intended Python and PE_EXPORT to your local "
            "raw export directory (not included in the source repository).\n"
            + manifest_preparation_hint(path, expected_sessions, expected_max_model_len)
        )
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"invalid prepared manifest schema: {path}; rerun prepare")
    if manifest.get("output_tokens") != OUTPUT_TOKENS:
        raise ValueError("manifest output budget must be exactly 128")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("manifest sessions must be a nonempty list")
    if type(manifest.get("selected_sessions")) is not int or manifest[
        "selected_sessions"
    ] != len(sessions):
        raise ValueError("manifest selected_sessions disagrees with actual task count")
    if expected_sessions is not None and len(sessions) != expected_sessions:
        raise ValueError(
            f"manifest has {len(sessions)} sessions; expected {expected_sessions}"
        )
    if concurrency is not None:
        validate_concurrency(concurrency, len(sessions))
    max_len = manifest.get("max_model_len")
    if type(max_len) is not int or max_len <= OUTPUT_TOKENS:
        raise ValueError("invalid manifest max_model_len")
    if expected_max_model_len is not None and max_len != expected_max_model_len:
        raise ValueError("manifest max_model_len differs from the launch profile")
    seen_sessions, call_count = set(), 0
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("manifest session must be an object")
        sid = session.get("session_id")
        if not isinstance(sid, str) or not sid or sid in seen_sessions:
            raise ValueError("manifest requires unique nonempty session IDs")
        seen_sessions.add(sid)
        calls = session.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"manifest session {sid} has no complete call sequence")
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ValueError("manifest call must be an object")
            if type(call.get("call_index")) is not int or call["call_index"] != i:
                raise ValueError("manifest call_index must be consecutive from zero")
            if "source_call" in call and call["source_call"] != i + 1:
                raise ValueError("manifest source_call disagrees with call sequence")
            if type(call.get("dataset_row")) is not int or call["dataset_row"] < 0:
                raise ValueError("invalid manifest dataset_row")
            ids = call.get("prompt_token_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(token) is not int or token < 0 for token in ids)
                or type(call.get("input_tokens")) is not int
                or call["input_tokens"] != len(ids)
                or hash_ids(ids) != call.get("prompt_sha256")
            ):
                raise ValueError("corrupt manifest prompt token length/hash")
            if len(ids) + OUTPUT_TOKENS > max_len:
                raise ValueError("manifest input + 128 exceeds configured context")
            clean_messages(call.get("messages"))
            gap = gap_seconds(call.get("pre_gap_seconds"), "s")
            if i == 0 and gap != 0:
                raise ValueError("first call must have zero replay gap")
            call_count += 1
    if (
        type(manifest.get("selected_calls")) is not int
        or manifest["selected_calls"] != call_count
    ):
        raise ValueError("manifest selected_calls disagrees with actual call count")
    if "request_order_sha256" in manifest and manifest[
        "request_order_sha256"
    ] != manifest_order_sha256(manifest):
        raise ValueError("manifest request order digest does not match its content")
    return manifest


def _selection_details(parent, parent_sha, selection, expected_sessions=None):
    count = selection.get("selected_sessions")
    ids = selection.get("session_ids")
    seed = selection.get("calibration_selection_seed")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(ids, list)
        or any(not isinstance(sid, str) for sid in ids)
        or len(ids) != count
        or len(set(ids)) != count
        or type(seed) is not int
    ):
        raise ValueError("invalid calibration selection count/IDs/seed")
    if expected_sessions is not None and count != expected_sessions:
        raise ValueError("selection session count differs from --sessions")
    if selection.get("calibration_parent_manifest_sha256") != parent_sha:
        raise ValueError(
            "selection parent SHA256 differs from this manifest; do not ignore it. "
            "Use the matching existing parent or create a new selection without "
            "--selection, preserving the recorded seed and a new output path."
        )
    all_ids = [s["session_id"] for s in parent["sessions"]]
    if count > len(all_ids) or not set(ids).issubset(all_ids):
        raise ValueError("selection contains missing or excessive parent sessions")
    sampled = set(random.Random(seed).sample(sorted(all_ids), count))
    if set(ids) != sampled:
        raise ValueError("selection IDs do not reproduce its recorded selection seed")
    selected = [s for s in parent["sessions"] if s["session_id"] in sampled]
    if [s["session_id"] for s in selected] != ids:
        raise ValueError("selection IDs are not in original parent manifest order")
    calls = [c for s in selected for c in s["calls"]]
    expected = {
        "selected_calls": len(calls),
        "max_input_tokens": max(c["input_tokens"] for c in calls),
        "fixed_output_tokens_total": len(calls) * OUTPUT_TOKENS,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"selection {key} differs from complete parent sessions")
    return selected


def validate_subset_selection(parent_path, subset_path, selection_path):
    """Validate full-session equality and the recorded selection, without GPU I/O."""
    parent = validate_manifest(parent_path, expected_sessions=50)
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Calibration selection record is missing: {selection_path}. "
            "Generate a new subset and its selection record with: "
            '"$PE_PYTHON" experiment/replay.py subset '
            f"--manifest {shlex.quote(str(parent_path))} --sessions 8 "
            "--seed 20261001 --out results/calibration_8_new.json"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    parent_sha = file_sha256(parent_path)
    selected = _selection_details(parent, parent_sha, selection)
    child = validate_manifest(
        subset_path,
        expected_sessions=len(selected),
        expected_max_model_len=parent["max_model_len"],
    )
    if child["sessions"] != selected:
        raise ValueError("calibration subset changes parent order or complete calls")
    for field in ("calibration_parent_manifest_sha256", "calibration_selection_seed"):
        if child.get(field) != selection[field]:
            raise ValueError(f"calibration subset {field} differs from selection")
    child_sha = file_sha256(subset_path)
    order_sha = manifest_order_sha256(child)
    if selection.get("manifest_sha256", child_sha) != child_sha:
        raise ValueError("calibration manifest SHA256 differs from selection")
    if selection.get("request_order_sha256", order_sha) != order_sha:
        raise ValueError("calibration order digest differs from selection")
    return {
        "calibration_parent_manifest_sha256": parent_sha,
        "manifest_sha256": child_sha,
        "request_order_sha256": order_sha,
        "selected_sessions": len(selected),
        "selected_calls": child["selected_calls"],
        "session_ids": selection["session_ids"],
        "calibration_selection_seed": selection["calibration_selection_seed"],
    }


def subset(args):
    """Select complete sessions from a fixed 50-session snapshot; no tokenizer."""
    out = Path(args.out)
    selection_out = out.with_suffix(".selection.json")
    if out.exists() or selection_out.exists():
        raise FileExistsError("subset output/selection already exists; use new paths")
    parent = validate_manifest(args.manifest, expected_sessions=50)
    if type(args.sessions) is not int or not 1 <= args.sessions <= 50:
        raise ValueError("subset --sessions must be between 1 and 50")
    parent_sha = file_sha256(args.manifest)
    if args.selection:
        selection_path = Path(args.selection)
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Selection record is missing: {selection_path}. "
                "Use an existing record or explicitly create a new selection "
                "by omitting --selection and supplying --seed 20261001."
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = _selection_details(
            parent, parent_sha, selection, expected_sessions=args.sessions
        )
        if (
            args.seed is not None
            and args.seed != selection["calibration_selection_seed"]
        ):
            raise ValueError("--seed differs from the selection record")
        seed = selection["calibration_selection_seed"]
    else:
        seed = 20261001 if args.seed is None else args.seed
        ids = sorted(s["session_id"] for s in parent["sessions"])
        chosen = set(random.Random(seed).sample(ids, args.sessions))
        selected = [s for s in parent["sessions"] if s["session_id"] in chosen]
    child = dict(parent)
    child.update(
        selected_sessions=len(selected),
        selected_calls=sum(len(s["calls"]) for s in selected),
        sessions=selected,
        calibration_parent_manifest_sha256=parent_sha,
        calibration_selection=(
            f"Uniform sample of {len(selected)} complete sessions from sorted IDs; "
            "retain original manifest order."
        ),
        calibration_selection_seed=seed,
    )
    if "request_order_sha256" in child:
        child["request_order_sha256"] = manifest_order_sha256(child)
    dump(out, child)
    selection = {
        "selected_sessions": len(selected),
        "selected_calls": child["selected_calls"],
        "calibration_parent_manifest_sha256": parent_sha,
        "calibration_selection": child["calibration_selection"],
        "calibration_selection_seed": seed,
        "session_ids": [s["session_id"] for s in selected],
        "max_input_tokens": max(
            c["input_tokens"] for s in selected for c in s["calls"]
        ),
        "fixed_output_tokens_total": child["selected_calls"] * OUTPUT_TOKENS,
        "gpu_executed": False,
        "manifest_sha256": file_sha256(out),
        "request_order_sha256": manifest_order_sha256(child),
    }
    if args.selection:
        selection["selection_source_path"] = str(Path(args.selection).resolve())
        selection["selection_source_sha256"] = file_sha256(args.selection)
    dump(selection_out, selection)
    print(
        json.dumps({"manifest": str(out), "selection": str(selection_out), **selection})
    )


def clean_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("input must be a nonempty cumulative message list")
    if not all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages
    ):
        raise ValueError("input contains invalid messages")
    return [
        {k: v for k, v in msg.items() if v is not None or k == "content"}
        for msg in messages
    ]


def template_messages(messages):
    result = json.loads(json.dumps(messages))
    for msg in result:
        for call in msg.get("tool_calls") or []:
            fn = call.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    return result


def encode(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        template_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        truncation=False,
    )["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("unexpected token batch")
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(type(x) is int for x in ids):
        raise ValueError("invalid token IDs")
    return ids


def gap_seconds(value, unit):
    if isinstance(value, bool):
        raise ValueError("pre_gap cannot be boolean")
    gap = float(value)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("pre_gap must be finite and nonnegative")
    if unit not in ("s", "ms", "us"):
        raise ValueError("explicit supported pre_gap unit required")
    return gap * {"s": 1.0, "ms": 0.001, "us": 0.000001}[unit]


def request_marker(run_id, session_id, call_index):
    if (
        not run_id
        or not isinstance(session_id, str)
        or type(call_index) is not int
        or call_index < 0
    ):
        raise ValueError("invalid request metadata")
    raw = json.dumps(
        {"r": run_id, "s": session_id, "c": call_index},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return "pe1." + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def metadata(request_id):
    """Accept the marker itself and vLLM's chatcmpl- / engine suffix wrapper.

    JSON's closing `}` provides an unambiguous boundary even if an engine suffix
    happens to consist of base64url characters. Decode success alone is not enough:
    re-encode and compare the exact marker before accepting a candidate.
    """
    if not isinstance(request_id, str):
        return None
    match = re.search(r"pe1\.([A-Za-z0-9_-]+)", request_id)
    if not match:
        return None
    encoded = match.group(1)
    # Engine IDs can append random suffixes. Only accept a canonical JSON marker.
    for length in range(len(encoded), max(0, len(encoded) - 128), -1):
        part = encoded[:length]
        try:
            decoded = base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
            obj = json.loads(decoded)
            if (
                set(obj) == {"r", "s", "c"}
                and isinstance(obj["r"], str)
                and isinstance(obj["s"], str)
                and type(obj["c"]) is int
                and obj["c"] >= 0
                and request_marker(obj["r"], obj["s"], obj["c"]) == "pe1." + part
            ):
                return obj
        except (ValueError, TypeError, UnicodeError):
            pass
    return None


class ReferenceMismatch(ValueError):
    """A fixed export differs from its original manifest; never resample it away."""


def index_reference(manifest):
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("reference manifest must contain a nonempty sessions list")
    order, records, original_rows = [], {}, set()
    for session in sessions:
        sid = str(session["session_id"])
        if sid in order:
            raise ValueError("duplicate session in reference manifest")
        order.append(sid)
        calls = session.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("empty reference session")
        for number, call in enumerate(calls, 1):
            row = call.get("dataset_row")
            if (
                type(call.get("call")) is not int
                or call["call"] != number
                or type(row) is not int
                or row < 0
                or row in original_rows
            ):
                raise ValueError(
                    "reference calls must have unique original rows "
                    "and consecutive 1-based call numbers"
                )
            if (
                type(call.get("prompt_tokens")) is not int
                or call["prompt_tokens"] <= 0
                or not isinstance(call.get("prompt_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", call["prompt_sha256"]) is None
            ):
                raise ValueError("invalid reference prompt token length/hash")
            gap_seconds(
                call.get("pre_gap"), "s"
            )  # Numeric validation; no unit inference.
            records[(sid, number, row)] = call
            original_rows.add(row)
    return order, records


def prepare_rows(
    rows,
    tokenizer,
    *,
    gap_unit,
    max_model_len,
    sessions=50,
    min_calls=10,
    max_calls=12,
    selection_seed=20261001,
    order_field=None,
    reference_manifest=None,
):
    reference_order, reference_records = (
        index_reference(reference_manifest)
        if reference_manifest is not None
        else (None, None)
    )
    grouped = defaultdict(list)
    seen_reference_keys = set()
    for i, row in enumerate(rows):
        if not isinstance(row.get("session_id"), (str, int)):
            raise ValueError("session_id must be string or integer")
        sid = str(row["session_id"])
        if reference_records is not None:
            if (
                type(row.get("original_dataset_row")) is not int
                or type(row.get("call")) is not int
            ):
                raise ReferenceMismatch(
                    "reference export requires integer original_dataset_row "
                    "and 1-based call"
                )
            key = (sid, row["call"], row["original_dataset_row"])
            if key not in reference_records or key in seen_reference_keys:
                raise ReferenceMismatch(
                    f"unknown/duplicate reference request at exported row {i}"
                )
            if row.get("pre_gap") != reference_records[key]["pre_gap"]:
                raise ReferenceMismatch(
                    f"pre_gap differs from reference at exported row {i}"
                )
            seen_reference_keys.add(key)
        grouped[sid].append((i, row))
    if reference_records is not None and seen_reference_keys != set(reference_records):
        raise ReferenceMismatch(
            "export is missing requests from the fixed reference manifest"
        )
    eligible, report = [], {}
    for sid, items in sorted(grouped.items()):
        if order_field:
            try:
                items = sorted(items, key=lambda pair: pair[1][order_field])
                keys = [row[order_field] for _, row in items]
                if len(set(keys)) != len(keys):
                    raise ValueError("duplicate call order values")
            except (KeyError, TypeError, ValueError) as exc:
                report[sid] = {
                    "calls": len(items),
                    "eligible": False,
                    "errors": [str(exc)],
                }
                continue
        calls, errors = [], []
        if not min_calls <= len(items) <= max_calls:
            errors.append(f"{len(items)} LLM calls outside [{min_calls}, {max_calls}]")
        if not errors:
            for call_index, (index, row) in enumerate(items):
                try:
                    messages = clean_messages(row["input"])
                    ids = encode(tokenizer, messages)
                    prompt_sha = hash_ids(ids)
                    if reference_records is not None:
                        expected = reference_records[
                            (sid, row["call"], row["original_dataset_row"])
                        ]
                        if row["call"] != call_index + 1:
                            raise ReferenceMismatch(
                                "reference call order differs; supply "
                                "--order-field call for unordered exports"
                            )
                        if (
                            len(ids) != expected["prompt_tokens"]
                            or prompt_sha != expected["prompt_sha256"]
                        ):
                            raise ReferenceMismatch(
                                "prompt token length/hash differs from reference "
                                f"for session {sid}, call {row['call']}"
                            )
                    # First call has no preceding local response. Do not interpret
                    # its trace gap as a delay from the batch's t0.
                    gap = (
                        0.0
                        if call_index == 0
                        else gap_seconds(row["pre_gap"], gap_unit)
                    )
                    if len(ids) + OUTPUT_TOKENS > max_model_len:
                        raise ValueError(
                            f"input {len(ids)} + output {OUTPUT_TOKENS} "
                            f"exceeds max_model_len {max_model_len}"
                        )
                    record = {
                        "call_index": call_index,
                        "dataset_row": index,
                        "input_tokens": len(ids),
                        "prompt_sha256": prompt_sha,
                        "prompt_token_ids": ids,
                        "messages": messages,
                        "pre_gap_seconds": gap,
                    }
                    if "original_dataset_row" in row:
                        record["original_dataset_row"] = row["original_dataset_row"]
                    if "call" in row:
                        record["source_call"] = row["call"]
                    calls.append(record)
                except ReferenceMismatch:
                    raise
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append({"dataset_row": index, "error": str(exc)})
        report[sid] = {"calls": len(items), "eligible": not errors, "errors": errors}
        if not errors:
            eligible.append({"session_id": sid, "calls": calls})
    if len(eligible) < sessions:
        return None, report
    selected = random.Random(selection_seed).sample(eligible, sessions)
    if reference_order is not None:
        # Selecting all 50 preserves the original fixed manifest's exact order.
        # A requested calibration subset retains relative order in that manifest.
        rank = {sid: i for i, sid in enumerate(reference_order)}
        selected.sort(key=lambda session: rank[session["session_id"]])
    return selected, report


def load_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return jsonl(path), sha_bytes(path.read_bytes())
    from datasets import load_from_disk

    rows = load_from_disk(str(path))
    if not hasattr(rows, "column_names") or isinstance(rows.column_names, dict):
        raise ValueError(
            "expected one Dataset, not a DatasetDict; supply the actual split directory"
        )
    return rows, getattr(rows, "_fingerprint", None)


def inspect_data(args):
    rows, fingerprint = load_rows(args.data)
    fields, sessions, gaps, message_counts = Counter(), Counter(), [], []
    invalid_gaps = 0
    for row in rows:
        fields.update(row.keys())
        sessions[str(row.get("session_id"))] += 1
        if isinstance(row.get("input"), list):
            message_counts.append(len(row["input"]))
        try:
            gap = float(row["pre_gap"])
            if not math.isfinite(gap) or gap < 0:
                raise ValueError("invalid gap")
            gaps.append(gap)
        except (ValueError, TypeError, KeyError):
            invalid_gaps += 1
    report = {
        "data": str(Path(args.data).resolve()),
        "fingerprint": fingerprint,
        "rows_llm_calls": len(rows),
        "sessions": len(sessions),
        "field_presence_counts": dict(fields),
        "calls_per_session_distribution": dict(
            sorted(Counter(sessions.values()).items())
        ),
        "messages_per_input_range": [min(message_counts), max(message_counts)]
        if message_counts
        else None,
        "pre_gap_raw_min": min(gaps) if gaps else None,
        "pre_gap_raw_max": max(gaps) if gaps else None,
        "pre_gap_raw_mean": statistics.mean(gaps) if gaps else None,
        "pre_gap_invalid_or_missing_rows": invalid_gaps,
        "pre_gap_unit": (
            "UNVERIFIED: numeric values alone do not identify unit or gap semantics"
        ),
        "within_session_order": (
            "UNVERIFIED: check dataset provenance / explicit call-order field"
        ),
    }
    dump(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare(args):
    from transformers import AutoTokenizer

    dest = Path(args.manifest)
    if dest.exists():
        raise ValueError("manifest already exists; use a new filename")
    evidence = Path(args.gap_evidence)
    if not evidence.is_file() or not evidence.read_text(encoding="utf-8").strip():
        raise ValueError(
            "--gap-evidence must be a nonempty local note/source "
            "documenting pre_gap unit and semantics"
        )
    data = Path(args.data)
    rows, fingerprint = load_rows(data)
    reference_arg = getattr(args, "reference_manifest", None)
    reference_path = Path(reference_arg).resolve() if reference_arg else None
    reference_bytes = reference_path.read_bytes() if reference_path else None
    reference = json.loads(reference_bytes) if reference_bytes is not None else None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    selected, report = prepare_rows(
        rows,
        tokenizer,
        gap_unit=args.gap_unit,
        max_model_len=args.max_model_len,
        sessions=args.sessions,
        min_calls=args.min_calls,
        max_calls=args.max_calls,
        selection_seed=args.selection_seed,
        order_field=args.order_field,
        reference_manifest=reference,
    )
    dump(dest.with_suffix(".preflight.json"), report)
    if selected is None:
        raise ValueError(
            f"only {sum(x['eligible'] for x in report.values())} eligible sessions; "
            f"need {args.sessions}; no partial-session inclusion or silent truncation"
        )
    manifest = {
        "schema": SCHEMA,
        "data": str(data.resolve()),
        "dataset_fingerprint": fingerprint,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_model_len": args.max_model_len,
        "output_tokens": OUTPUT_TOKENS,
        "gap_unit": args.gap_unit,
        "gap_evidence_path": str(evidence.resolve()),
        "gap_evidence_sha256": sha_bytes(evidence.read_bytes()),
        "gap_semantics": (
            "previous response completes, then wait this call's pre_gap; "
            "first call immediate"
        ),
        "call_order": args.order_field
        or (
            "dataset row order within session "
            "(must be verified against dataset provenance)"
        ),
        "min_calls": args.min_calls,
        "max_calls": args.max_calls,
        "reference_manifest_path": str(reference_path) if reference_path else None,
        "reference_manifest_sha256": sha_bytes(reference_bytes)
        if reference_bytes is not None
        else None,
        "reference_token_hashes_verified": reference is not None,
        "reference_token_hash_scope": "selected sessions"
        if reference is not None
        else None,
        "session_order": "original reference manifest order"
        if reference is not None
        else "seeded session sample order",
        "selection_seed": args.selection_seed,
        "selected_sessions": len(selected),
        "selected_calls": sum(len(s["calls"]) for s in selected),
        "sessions": selected,
        "preprocessing": (
            "omit null optional fields; parse tool arguments on tokenizer copy; "
            "preserve roles and full cumulative inputs"
        ),
    }
    dump(dest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(dest),
                "sessions": len(selected),
                "calls": manifest["selected_calls"],
                "calls_distribution": dict(Counter(len(s["calls"]) for s in selected)),
                "max_input_tokens": max(
                    c["input_tokens"] for s in selected for c in s["calls"]
                ),
            },
            ensure_ascii=False,
        )
    )


def http(base, route, payload=None, *, method=None, marker=None, timeout=30):
    parts = urllib.parse.urlsplit(base)
    if (
        parts.scheme not in ("http", "https")
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "base URL must be plain http(s) origin "
            "without credentials, query or fragment"
        )
    headers = {"Content-Type": "application/json"}
    if marker:
        headers["X-Request-Id"] = marker
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    req = urllib.request.Request(
        base.rstrip("/") + route,
        method=method,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode(),
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as exc:
        # Never print reflected prompt bodies or authentication headers.
        raise RuntimeError(f"HTTP {exc.code} for {route.split('?')[0]}") from None


def metric_values(text, name):
    pattern = r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([^\s]+)"
    return [float(m.group(1)) for m in re.finditer(pattern, text, re.M)]


def server_cache_config(text):
    lines = [
        line for line in text.splitlines() if line.startswith("vllm:cache_config_info{")
    ]
    if len(lines) != 1:
        raise ValueError("need exactly one engine's cache_config_info metric")
    return dict(re.findall(r'(\w+)="([^"]*)"', lines[0]))


def assert_idle(text):
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        vals = metric_values(text, name)
        if not vals or any(not math.isfinite(v) or v != 0 for v in vals):
            raise ValueError(f"server must be idle; {name} not confirmed zero")


class MetricsMonitor(threading.Thread):
    """Same one-second diagnostics in both groups, bounded in-memory state."""

    def __init__(self, base_url, output_path):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.output_path = Path(output_path)
        self.stop_event = threading.Event()
        self.errors = 0

    def run(self):
        start = time.monotonic()
        with self.output_path.open("w", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                row = {"monitor_elapsed_seconds": time.monotonic() - start}
                try:
                    snapshot = http(self.base_url, "/metrics", timeout=5)
                    for name in (
                        "num_requests_running",
                        "num_requests_waiting",
                        "kv_cache_usage_perc",
                        "num_preemptions_total",
                        "prefix_cache_queries_total",
                        "prefix_cache_hits_total",
                    ):
                        vals = metric_values(snapshot, "vllm:" + name)
                        row[name] = sum(vals) if vals else None
                except Exception as exc:
                    self.errors += 1
                    row["error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.stop_event.wait(1)

    def close(self):
        self.stop_event.set()
        self.join(timeout=6)


def validate_cache(config, block_size, num_gpu_blocks):
    if config.get("enable_prefix_caching", "").lower() != "true":
        raise ValueError("prefix caching must be enabled")
    if (
        int(config.get("block_size", -1)) != block_size
        or int(config.get("num_gpu_blocks", -1)) != num_gpu_blocks
    ):
        raise ValueError(
            "actual KV block size/count differs from fixed expected capacity"
        )
    if config.get("prefix_caching_hash_block_size") not in (
        None,
        "None",
        str(block_size),
    ):
        raise ValueError("prototype requires hash block size = physical block size")


def engine_evidence(
    log_dir, *, policy, block_size, num_gpu_blocks, max_model_len, requested_run_id=None
):
    path = Path(log_dir) / "run_config.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    mode = evidence.get("mode", evidence.get("policy"))
    if mode != policy:
        raise ValueError("actual EngineCore policy differs from requested policy")
    for field, expected in (
        ("block_size", block_size),
        ("num_gpu_blocks", num_gpu_blocks),
        ("max_model_len", max_model_len),
    ):
        if evidence.get(field) != expected:
            raise ValueError(
                f"actual EngineCore {field} differs from fixed experiment value"
            )
    if not str(evidence.get("version", "")).startswith("0.29.0"):
        raise ValueError("this adapter is source-checked only for vLLM 0.29.0")
    if evidence.get("use_flashinfer_sampler") is not False:
        raise ValueError(
            "EngineCore use_flashinfer_sampler must be boolean False; "
            "start the experiment with VLLM_USE_FLASHINFER_SAMPLER=0"
        )
    if not evidence.get("run_id") or (
        requested_run_id and evidence["run_id"] != requested_run_id
    ):
        raise ValueError("client --run-id and actual EngineCore run_id must match")
    paths, hashes = evidence.get("core_paths"), evidence.get("core_sha256")
    if (
        not isinstance(paths, dict)
        or len(paths) < 4
        or not isinstance(hashes, dict)
        or set(paths) != set(hashes)
    ):
        raise ValueError("missing actual EngineCore module import paths/source hashes")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("invalid EngineCore source hashes")
    # Compare actual runtime configuration and code content; filesystem locations
    # and policy switch may differ without changing the controlled experiment.
    selected = {
        k: v
        for k, v in evidence.items()
        if k
        not in {
            "run_id",
            "mode",
            "policy",
            "log_dir",
            "python",
            "pid",
            "timestamp",
            "core_paths",
        }
    }
    for required in (
        "model",
        "dtype",
        "async_scheduling",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "attention_backend",
        "use_flashinfer_sampler",
        "enforce_eager",
        "check_interval",
        "first_cold_ratio",
        "later_cold_ratio",
        "private_suffix_only",
    ):
        if required not in selected:
            raise ValueError(f"engine runtime evidence missing {required}")
    return evidence, sha_bytes(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    )


def confirmed_policy_reset(log_dir, previous_rows, run_id):
    new = jsonl(Path(log_dir) / "reset.jsonl")[previous_rows:]
    return next(
        (
            r
            for r in reversed(new)
            if r.get("run_id") == run_id
            and r.get("success") is True
            and r.get("completed") == 0
        ),
        None,
    )


def engine_metrics(log_dir, *, run_id, checks_offset, policy):
    root = Path(log_dir)
    totals = json.loads((root / "eviction_summary.json").read_text(encoding="utf-8"))
    for name in (
        "active_logical",
        "active_physical",
        "native_logical",
        "native_physical",
    ):
        if type(totals.get(name)) is not int or totals[name] < 0:
            raise ValueError(f"missing/invalid actual engine eviction counter {name}")
    checks = jsonl(root / "policy_checks.jsonl")[checks_offset:]
    for check in checks:
        if check.get("run_id") != run_id:
            raise ValueError("mixed run IDs in policy checks")
        if check.get("free_before") != check.get("free_after"):
            raise ValueError("pure invalidation/reordering changed free block capacity")
        if (
            not isinstance(check.get("duration_ms"), (int, float))
            or not math.isfinite(check["duration_ms"])
            or check["duration_ms"] < 0
        ):
            raise ValueError("invalid policy check duration")
    if sum(c["evicted_logical_blocks"] for c in checks) != totals["active_logical"]:
        raise ValueError(
            "checkpoint active logical evictions disagree with reset-scoped totals"
        )
    if sum(c["evicted_physical_blocks"] for c in checks) != totals["active_physical"]:
        raise ValueError(
            "checkpoint physical evictions disagree with reset-scoped totals"
        )
    if policy == "native" and (totals["active_logical"] or totals["active_physical"]):
        raise ValueError("native mode unexpectedly performed proactive hard eviction")
    durations = [c["duration_ms"] for c in checks]
    result = {
        "active_evicted_logical_blocks": totals["active_logical"],
        "active_evicted_physical_blocks": totals["active_physical"],
        "native_evicted_observed_logical_blocks": totals["native_logical"],
        "native_evicted_physical_blocks": totals["native_physical"],
        "policy_check_count": len(checks),
        "policy_check_total_ms": sum(durations),
        "policy_check_mean_ms": statistics.mean(durations) if durations else None,
        "policy_check_max_ms": max(durations) if durations else None,
        "policy_free_count_invariant_verified": True,
        "native_logical_counter_scope": (
            "last resident copy removed for policy-observed logical nodes; "
            "physical counter includes all valid hashed copies "
            "overwritten by native allocation"
        ),
    }
    return result, checks, totals


def workload_qualification(summary, policy, requests):
    """Keep zero/negative-benefit runs valid; qualify the mechanism exercised.

    Waiting requests or a high utilization sample alone cannot prove KV pressure:
    batching/token budgets can cause queuing too. Use observed native replacement
    of valid cached blocks and actual per-request engine preemption counts.
    """
    preemptions = [
        r["engine_preemptions"]
        for r in requests
        if type(r.get("engine_preemptions")) is int and r["engine_preemptions"] >= 0
    ]
    native_overwrites = summary["native_evicted_physical_blocks"]
    active = summary["active_evicted_logical_blocks"]
    pressure = native_overwrites > 0 or sum(preemptions) > 0
    notes = []
    if policy == "progressive" and active == 0:
        notes.append(
            "Progressive mode performed zero proactive hard evictions. "
            "Timing results remain valid, but this run did not exercise "
            "the proactive eviction mechanism."
        )
    if not pressure:
        notes.append(
            "No native overwrite of valid cached blocks or request preemption "
            "was observed. This run does not establish cache competition; "
            "a paired baseline may provide separate pressure evidence."
        )
    return {
        "proactive_eviction_exercised": active > 0,
        "cache_pressure_observed": pressure,
        "cache_pressure_evidence": {
            "native_valid_cached_physical_overwrites": native_overwrites,
            "request_preemptions": sum(preemptions),
            "requests_with_preemption_observation": len(preemptions),
            "requests_joined": len(requests),
        },
        "qualification_notes": notes,
    }


def payload_for(call, model):
    return {
        "model": model,
        "messages": call["messages"],
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stop": [],
        "stop_token_ids": [],
        "n": 1,
        "stream": False,
        "return_token_ids": True,
        "add_generation_prompt": True,
    }


def validate_response(response, call, marker):
    if metadata(response.get("id")) != metadata(marker):
        raise ValueError("response request ID does not preserve supplied metadata")
    ids = response.get("prompt_token_ids")
    if not isinstance(ids, list) or not ids or not all(type(x) is int for x in ids):
        raise ValueError(
            "missing actual prompt_token_ids; cannot verify server tokenization"
        )
    usage = response.get("usage") or {}
    if (
        usage.get("prompt_tokens") != len(ids)
        or len(ids) != call["input_tokens"]
        or hash_ids(ids) != call["prompt_sha256"]
    ):
        raise ValueError(
            "server token IDs differ from CPU preflight; no truncation is allowed"
        )
    if usage.get("completion_tokens") != OUTPUT_TOKENS:
        raise ValueError(
            f"actual completion length differs from fixed {OUTPUT_TOKENS} tokens"
        )
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "length":
        raise ValueError("response did not finish at the required output length")
    return {
        "input_tokens": len(ids),
        "output_tokens": usage["completion_tokens"],
        "response_id": response["id"],
        "finish_reason": choices[0]["finish_reason"],
        "prompt_sha256": hash_ids(ids),
        "response_cached_tokens_diagnostic": (
            usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens"),
    }


async def replay_sessions(
    sessions,
    *,
    concurrency,
    seed,
    run_id,
    send,
    clock=time.monotonic,
    sleep=asyncio.sleep,
    emit=None,
):
    """All tasks arrive at t0. Exactly C session workers hold admission slots.

    On a failed request, its session ends; other sessions finish and all outcomes
    are logged. Such a run is marked invalid for equal-workload comparisons.
    `send` returns (response, begin_monotonic, end_monotonic).
    """
    validate_concurrency(concurrency, len(sessions))
    ordered = list(sessions)
    random.Random(seed).shuffle(ordered)
    queue = asyncio.Queue()
    for session in ordered:
        queue.put_nowait(session)
    start = clock()
    tasks, requests = [], []
    inflight_http_requests = 0
    max_inflight_http_requests = 0
    emit = emit or (lambda kind, record: None)
    for session in ordered:
        emit("arrival", {"session_id": session["session_id"], "arrival_seconds": 0.0})

    async def worker():
        nonlocal inflight_http_requests, max_inflight_http_requests
        while not queue.empty():
            try:
                session = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            sid = session["session_id"]
            task = {
                "session_id": sid,
                "arrival_seconds": 0.0,
                "admission_seconds": clock() - start,
                "success": True,
                "completed_calls": 0,
            }
            previous_end = None
            for call in session["calls"]:
                if previous_end is not None:
                    await sleep(
                        max(0, previous_end + call["pre_gap_seconds"] - clock())
                    )
                marker = request_marker(run_id, sid, call["call_index"])
                record = {
                    "run_id": run_id,
                    "request_id": marker,
                    "session_id": sid,
                    "call_index": call["call_index"],
                    "dataset_row": call["dataset_row"],
                    "success": False,
                    "pre_gap_seconds": call["pre_gap_seconds"],
                }
                try:
                    inflight_http_requests += 1
                    max_inflight_http_requests = max(
                        max_inflight_http_requests, inflight_http_requests
                    )
                    try:
                        response, begin, end = await send(call, marker)
                    finally:
                        inflight_http_requests -= 1
                    record.update(
                        started_seconds=begin - start,
                        completed_seconds=end - start,
                        actual_gap_seconds=None
                        if previous_end is None
                        else begin - previous_end,
                    )
                    record["observed_output_tokens"] = (
                        response.get("usage") or {}
                    ).get("completion_tokens")
                    record.update(validate_response(response, call, marker))
                    record["success"] = True
                    previous_end = end
                    task["completed_calls"] += 1
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    task["success"] = False
                requests.append(record)
                emit("request", record)
                if not task["success"]:
                    break
            task["completion_seconds"] = (
                previous_end - start if task["success"] else clock() - start
            )
            tasks.append(task)
            emit("task", task)
            queue.task_done()

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return {
        "tasks": tasks,
        "requests": requests,
        "task_order": [s["session_id"] for s in ordered],
        "measurement_seconds": max(
            (t["completion_seconds"] for t in tasks), default=0.0
        ),
        "configured_active_session_limit": concurrency,
        "effective_worker_count": concurrency,
        "observed_max_inflight_http_requests": max_inflight_http_requests,
    }


def client_summary(result):
    tasks, requests = result["tasks"], result["requests"]
    success = sum(t["success"] for t in tasks)
    complete = (
        bool(tasks) and success == len(tasks) and all(r["success"] for r in requests)
    )
    duration = result["measurement_seconds"]
    output = sum(r.get("output_tokens", 0) for r in requests if r["success"])
    return {
        "valid_complete_run": complete,
        "sessions_planned": len(tasks),
        "sessions_completed": success,
        "sessions_failed": len(tasks) - success,
        "requests_successful": sum(r["success"] for r in requests),
        "requests_failed": sum(not r["success"] for r in requests),
        "elapsed_seconds": duration,
        "configured_active_session_limit": result.get(
            "configured_active_session_limit"
        ),
        "effective_worker_count": result.get("effective_worker_count"),
        "observed_max_inflight_http_requests": result.get(
            "observed_max_inflight_http_requests"
        ),
        "concurrency_scope": (
            "session workers include tool-gap waits; observed HTTP concurrency "
            "counts client send calls awaiting completion, not GPU batch size"
        ),
        "output_tokens": output,
        "observed_output_tokens_including_invalid_responses": sum(
            r.get("observed_output_tokens", r.get("output_tokens", 0))
            for r in requests
            if type(r.get("observed_output_tokens", r.get("output_tokens", 0))) is int
        ),
        "task_throughput_per_second": success / duration
        if complete and duration > 0
        else None,
        "mean_task_completion_seconds": statistics.mean(
            t["completion_seconds"] - t["arrival_seconds"] for t in tasks
        )
        if complete
        else None,
        "output_token_throughput_per_second": output / duration
        if complete and duration > 0
        else None,
    }


def join_engine_records(client_records, engine_records, run_id, block_size=None):
    indexed = {}
    for row in engine_records:
        meta = metadata(row.get("request_id"))
        if not meta or meta["r"] != run_id:
            continue
        key = (meta["s"], meta["c"])
        if key in indexed:
            raise ValueError(
                "duplicate engine completion record for one logical request"
            )
        if (
            str(row.get("session_id")) != meta["s"]
            or row.get("call_index") != meta["c"]
        ):
            raise ValueError("engine session metadata disagrees with request ID")
        indexed[key] = row
    joined, h_total, lost = [], 0, 0
    for client in client_records:
        if not client["success"]:
            continue
        key = (client["session_id"], client["call_index"])
        if key not in indexed:
            raise ValueError(
                f"missing EngineCore record for session/call {key}; "
                "E2E metadata/metrics not verified"
            )
        engine = indexed[key]
        if not engine.get("success"):
            raise ValueError("client/engine success status disagreement")
        for name in ("input_tokens", "output_tokens"):
            if engine.get(name) != client[name]:
                raise ValueError(f"client/engine {name} disagreement")
        h, k = engine.get("history_reusable_tokens"), engine.get("adopted_hit_tokens")
        if (
            type(h) is not int
            or type(k) is not int
            or not 0 <= k <= h < client["input_tokens"]
        ):
            raise ValueError("engine H/K missing or out of range")
        if block_size and (h % block_size or k % block_size):
            raise ValueError("history/hit tokens must obey full block boundaries")
        if client["call_index"] == 0 and h != 0:
            raise ValueError(
                "first session call cannot have locally computed session history"
            )
        loss = max(0, h - k)
        h_total += h
        lost += loss
        joined.append(
            dict(
                client,
                history_reusable_tokens=h,
                adopted_hit_tokens=k,
                lost_history_tokens=loss,
                engine_preemptions=engine.get("preemptions"),
            )
        )
    if len(indexed) != len(joined):
        raise ValueError("unmatched engine records for this run")
    return joined, {
        "history_reusable_tokens": h_total,
        "lost_history_tokens": lost,
        "historical_prefix_loss_rate": lost / h_total if h_total else None,
        "engine_request_metadata_verified": True,
    }


def run(args):
    manifest_path = Path(args.manifest)
    manifest = validate_manifest(manifest_path, concurrency=args.concurrency)
    sessions = manifest["sessions"]
    evidence, signature = engine_evidence(
        args.engine_log_dir,
        policy=args.policy,
        block_size=args.expected_block_size,
        num_gpu_blocks=args.expected_num_gpu_blocks,
        max_model_len=manifest["max_model_len"],
        requested_run_id=args.run_id,
    )
    run_id = evidence["run_id"]
    if any(
        (metadata(row.get("request_id")) or {}).get("r") == run_id
        for row in jsonl(Path(args.engine_log_dir) / "requests.jsonl")
    ):
        raise ValueError(
            "EngineCore log already contains this measured run; restart dedicated "
            "experiment process with new run_id/log directory"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "schema": SCHEMA,
        "run_id": run_id,
        "policy": args.policy,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "configured_active_session_limit": args.concurrency,
        "effective_worker_count": args.concurrency,
        "model": evidence["model"],
        "request_model_alias": args.model,
        "manifest_sha256": file_sha256(manifest_path),
        "num_gpu_blocks": args.expected_num_gpu_blocks,
        "block_size": args.expected_block_size,
        "max_model_len": manifest["max_model_len"],
        "output_tokens": OUTPUT_TOKENS,
        "sessions": len(sessions),
        "clock": "client time.monotonic; all arrivals=t0",
        "engine_log_dir": str(Path(args.engine_log_dir).resolve()),
        "engine_comparison_signature": signature,
        "status": "preparing",
    }
    dump(out / "run_config.json", config)
    dump(out / "engine_startup.json", evidence)
    monitor = None
    try:
        models = json.loads(http(args.base_url, "/v1/models"))
        matching = [m for m in models.get("data", []) if m.get("id") == args.model]
        if (
            len(matching) != 1
            or matching[0].get("max_model_len") != manifest["max_model_len"]
        ):
            raise ValueError(
                "server model name or max_model_len differs from preflight"
            )
        before = http(args.base_url, "/metrics")
        assert_idle(before)
        cache = server_cache_config(before)
        validate_cache(cache, args.expected_block_size, args.expected_num_gpu_blocks)
        (out / "metrics_before.txt").write_text(before)
        config["cache_config"] = cache
        # Development reset endpoint is version checked; never substitute cache
        # salts, and never reset once per task/request. Warmup is outside t0.
        call = sessions[0]["calls"][0]
        if any(s["session_id"] == "__pe_warmup__" for s in sessions):
            raise ValueError("trace uses reserved warmup session name")
        warmup_id = request_marker(run_id, "__pe_warmup__", 0)
        response = json.loads(
            http(
                args.base_url,
                "/v1/chat/completions",
                payload_for(call, args.model),
                marker=warmup_id,
                timeout=args.timeout,
            )
        )
        validate_response(response, call, warmup_id)
        assert_idle(http(args.base_url, "/metrics"))
        reset_rows = len(jsonl(Path(args.engine_log_dir) / "reset.jsonl"))
        reply = json.loads(
            http(
                args.base_url,
                "/reset_prefix_cache?reset_running_requests=false&reset_external=false",
                method="POST",
            )
        )
        if reply != {"success": True}:
            raise ValueError(
                "reset endpoint did not explicitly confirm empty prefix cache"
            )
        deadline = time.monotonic() + args.log_wait_seconds
        reset_proof = None
        while time.monotonic() < deadline:
            with suppress(json.JSONDecodeError):
                reset_proof = confirmed_policy_reset(
                    args.engine_log_dir, reset_rows, run_id
                )
            if reset_proof:
                break
            time.sleep(0.1)
        if not reset_proof:
            raise ValueError(
                "prefix cache reset lacks EngineCore policy/ghost/completion "
                "reset proof; formal t0 not started"
            )
        after_reset = http(args.base_url, "/metrics")
        assert_idle(after_reset)
        validate_cache(
            server_cache_config(after_reset),
            args.expected_block_size,
            args.expected_num_gpu_blocks,
        )
        (out / "metrics_after_reset.txt").write_text(after_reset)
        config["cache_start"] = "warmup_then_confirmed_reset"
        config["policy_reset_proof"] = reset_proof
        config["engine_requests_log_offset"] = len(
            jsonl(Path(args.engine_log_dir) / "requests.jsonl")
        )
        config["engine_checks_log_offset"] = len(
            jsonl(Path(args.engine_log_dir) / "policy_checks.jsonl")
        )
        config["status"] = "running"
        dump(out / "run_config.json", config)
        monitor = MetricsMonitor(args.base_url, out / "metrics.jsonl")
        monitor.start()
        with (
            (out / "client_requests.jsonl").open("w", encoding="utf-8") as rf,
            (out / "client_tasks.jsonl").open("w", encoding="utf-8") as tf,
            (out / "client_arrivals.jsonl").open("w", encoding="utf-8") as af,
        ):

            def emit(kind, row):
                file = {"request": rf, "task": tf, "arrival": af}[kind]
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()

            def send_sync(call, marker):
                begin = time.monotonic()
                response = json.loads(
                    http(
                        args.base_url,
                        "/v1/chat/completions",
                        payload_for(call, args.model),
                        marker=marker,
                        timeout=args.timeout,
                    )
                )
                return response, begin, time.monotonic()

            async def run_async():
                asyncio.get_running_loop().set_default_executor(
                    ThreadPoolExecutor(max_workers=args.concurrency)
                )

                async def send(call, marker):
                    return await asyncio.to_thread(send_sync, call, marker)

                return await replay_sessions(
                    sessions,
                    concurrency=args.concurrency,
                    seed=args.seed,
                    run_id=run_id,
                    send=send,
                    emit=emit,
                )

            result = asyncio.run(run_async())
        dump(out / "task_order.json", result["task_order"])
        monitor.close()
        summary = client_summary(result)
        summary["metrics_poll_errors"] = monitor.errors
        dump(out / "client_summary.json", summary)
        # Policy logger must flush one completion record before/at final output.
        # Local file retries tolerate its bounded flush interval; no model retry.
        engine_file = Path(args.engine_log_dir) / "requests.jsonl"
        deadline = time.monotonic() + args.log_wait_seconds
        while True:
            try:
                joined, reuse = join_engine_records(
                    result["requests"],
                    jsonl(engine_file)[config["engine_requests_log_offset"] :],
                    run_id,
                    args.expected_block_size,
                )
                break
            except (ValueError, json.JSONDecodeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
            for row in joined:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary.update(reuse)
        eviction, checks, totals = engine_metrics(
            args.engine_log_dir,
            run_id=run_id,
            checks_offset=config["engine_checks_log_offset"],
            policy=args.policy,
        )
        summary.update(eviction)
        summary.update(workload_qualification(summary, args.policy, joined))
        dump(out / "eviction_summary.json", totals)
        with (out / "policy_checks.jsonl").open("w", encoding="utf-8") as f:
            for row in checks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary.update(run_id=run_id, policy=args.policy, seed=args.seed)
        dump(out / "summary.json", summary)
        (out / "metrics_after.txt").write_text(http(args.base_url, "/metrics"))
        config["status"] = "completed" if summary["valid_complete_run"] else "failed"
        dump(out / "run_config.json", config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["valid_complete_run"]:
            raise RuntimeError(
                "failed requests/tasks: run cannot be used "
                "for equal-workload comparison"
            )
    except BaseException as exc:
        config["status"] = "failed"
        dump(out / "run_config.json", config)
        dump(
            out / "error.json",
            {"error": f"{type(exc).__name__}: {exc}", "valid_complete_run": False},
        )
        raise
    finally:
        if monitor is not None and monitor.is_alive():
            monitor.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect-data",
        help="local-only schema/count/gap diagnostics; never guesses time unit",
    )
    inspect.add_argument("--data", required=True)
    inspect.add_argument("--out", required=True)
    a = sub.add_parser(
        "prepare", help="CPU tokenizer preflight; selects whole sessions"
    )
    a.add_argument(
        "--data", required=True, help="datasets.save_to_disk directory or JSONL"
    )
    a.add_argument(
        "--tokenizer",
        required=True,
        help="existing local deployed model/tokenizer path",
    )
    a.add_argument("--manifest", required=True)
    a.add_argument(
        "--reference-manifest",
        help=(
            "original fixed manifest; requires the complete matching JSONL export "
            "with original_dataset_row/call and verifies token length/hash/gaps"
        ),
    )
    a.add_argument("--gap-unit", required=True, choices=("s", "ms", "us"))
    a.add_argument(
        "--gap-evidence",
        required=True,
        help=(
            "local evidence note documenting actual field unit/semantics "
            "and within-session row ordering"
        ),
    )
    a.add_argument(
        "--order-field",
        help="explicit call ordering column; default verified dataset row order",
    )
    a.add_argument("--max-model-len", required=True, type=int)
    a.add_argument("--sessions", type=int, default=50)
    a.add_argument("--min-calls", type=int, default=10)
    a.add_argument("--max-calls", type=int, default=12)
    a.add_argument("--selection-seed", type=int, default=20261001)
    subset_parser = sub.add_parser(
        "subset", help="derive complete calibration sessions without a tokenizer"
    )
    subset_parser.add_argument(
        "--manifest", required=True, help="fixed 50-session parent"
    )
    subset_parser.add_argument("--sessions", type=int, required=True)
    subset_parser.add_argument("--selection", help="existing ID/seed/parent-SHA record")
    subset_parser.add_argument(
        "--seed", type=int, help="default 20261001 for a new selection"
    )
    subset_parser.add_argument(
        "--out", required=True, help="new subset filename; never overwritten"
    )
    validate_parser = sub.add_parser(
        "validate", help="CPU-only prepared-manifest checks before server startup"
    )
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--expected-sessions", type=int)
    validate_parser.add_argument("--concurrency", type=int)
    validate_parser.add_argument("--expected-max-model-len", type=int)
    validate_parser.add_argument("--out", help="optional small validation report")
    b = sub.add_parser(
        "run",
        help=(
            "causal HTTP replay against an already started dedicated experiment server"
        ),
    )
    b.add_argument("--manifest", required=True)
    b.add_argument("--base-url", default="http://127.0.0.1:8001")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True, help="new run directory; must not exist")
    b.add_argument("--engine-log-dir", required=True)
    b.add_argument("--policy", required=True, choices=("native", "progressive"))
    b.add_argument("--seed", required=True, type=int)
    b.add_argument("--concurrency", type=int, default=4)
    b.add_argument("--expected-block-size", type=int, required=True)
    b.add_argument("--expected-num-gpu-blocks", type=int, required=True)
    b.add_argument("--timeout", type=float, default=1800)
    b.add_argument("--log-wait-seconds", type=float, default=5)
    b.add_argument("--run-id")
    args = p.parse_args()
    if args.command == "inspect-data":
        inspect_data(args)
    elif args.command == "prepare":
        if (
            args.sessions < 1
            or args.min_calls < 1
            or args.max_calls < args.min_calls
            or args.max_model_len <= OUTPUT_TOKENS
        ):
            p.error("invalid session/call/model-length bounds")
        prepare(args)
    elif args.command == "subset":
        subset(args)
    elif args.command == "validate":
        manifest = validate_manifest(
            args.manifest,
            expected_sessions=args.expected_sessions,
            concurrency=args.concurrency,
            expected_max_model_len=args.expected_max_model_len,
        )
        result = {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "selected_sessions": manifest["selected_sessions"],
            "selected_calls": manifest["selected_calls"],
            "request_order_sha256": manifest_order_sha256(manifest),
            "max_model_len": manifest["max_model_len"],
            "output_tokens": manifest["output_tokens"],
            "configured_active_session_limit": args.concurrency,
            "effective_worker_count": args.concurrency,
            "stored_token_hashes_verified": True,
            "tokens_retokenized": False,
            "gpu_executed": False,
            "http_requests": 0,
        }
        if args.out:
            dump(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if (
            args.concurrency < 1
            or args.expected_block_size < 1
            or args.expected_num_gpu_blocks < 1
            or any(
                not math.isfinite(v) or v <= 0
                for v in (args.timeout, args.log_wait_seconds)
            )
        ):
            p.error("concurrency/capacity/timeouts must be positive")
        run(args)


if __name__ == "__main__":
    main()
