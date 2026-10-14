"""CPU-only tests of concrete replay/measurement risks; no vLLM/GPU claim."""

import asyncio
import copy
import io
import json
import random
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from replay import (
    OUTPUT_TOKENS,
    SCHEMA,
    client_summary,
    confirmed_policy_reset,
    engine_evidence,
    engine_metrics,
    file_sha256,
    gap_seconds,
    hash_ids,
    join_engine_records,
    manifest_order_sha256,
    metadata,
    payload_for,
    prepare_rows,
    replay_sessions,
    request_marker,
    run,
    subset,
    validate_cache,
    validate_manifest,
    validate_response,
    validate_subset_selection,
    workload_qualification,
)
from summarize import paired_results, plan


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [ord(c) for m in messages for c in m["content"]]}


def make_call(i=0, gap=0.0, ids=None):
    ids = ids or list(range(160))
    return {
        "call_index": i,
        "dataset_row": i,
        "input_tokens": len(ids),
        "prompt_sha256": hash_ids(ids),
        "prompt_token_ids": ids,
        "messages": [{"role": "user", "content": f"fixed trace input {i}"}],
        "pre_gap_seconds": gap,
    }


def response(call, marker):
    return {
        "id": "chatcmpl-" + marker,
        "prompt_token_ids": call["prompt_token_ids"],
        "usage": {"prompt_tokens": call["input_tokens"], "completion_tokens": 128},
        "choices": [
            {"finish_reason": "length", "message": {"content": "do not append me"}}
        ],
    }


def prepared_fixture(count=50):
    sessions = []
    for i in range(count):
        call = make_call()
        call["dataset_row"] = i
        sessions.append({"session_id": f"s{i:02}", "calls": [call]})
    return {
        "schema": SCHEMA,
        "selected_sessions": count,
        "selected_calls": count,
        "output_tokens": 128,
        "max_model_len": 1024,
        "sessions": sessions,
    }


class ReplayTests(unittest.TestCase):
    def test_metadata_survives_engine_wrappers_unicode_and_greedy_suffix(self):
        marker = request_marker("runA", "任务-1/a", 10)
        expected = {"r": "runA", "s": "任务-1/a", "c": 10}
        self.assertEqual(metadata(marker), expected)
        self.assertEqual(metadata("chatcmpl-" + marker + "-aBcD1234"), expected)
        self.assertIsNone(metadata("unrelated"))

    def test_gap_units_reject_invalid_and_first_gap_not_applied(self):
        self.assertEqual(gap_seconds(250, "ms"), 0.25)
        for gap in (-1, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                gap_seconds(gap, "s")
        rows = [
            {
                "session_id": "s",
                "input": [{"role": "user", "content": "abc"}],
                "pre_gap": None,
            },
            {
                "session_id": "s",
                "input": [{"role": "user", "content": "abcdef"}],
                "pre_gap": 250,
            },
        ]
        selected, _ = prepare_rows(
            rows,
            FakeTokenizer(),
            gap_unit="ms",
            max_model_len=256,
            sessions=1,
            min_calls=2,
            max_calls=2,
        )
        self.assertEqual(
            [c["pre_gap_seconds"] for c in selected[0]["calls"]], [0, 0.25]
        )

    def test_whole_sessions_calls_not_message_count_and_no_truncation(self):
        rows = [
            {
                "session_id": "a",
                "input": [{"role": "user", "content": "x"}] * 11,
                "pre_gap": 0,
            },
            {
                "session_id": "b",
                "input": [{"role": "user", "content": "x" * 100}],
                "pre_gap": 0,
            },
            {
                "session_id": "b",
                "input": [{"role": "user", "content": "x" * 200}],
                "pre_gap": 0,
            },
        ]
        chosen, report = prepare_rows(
            rows,
            FakeTokenizer(),
            gap_unit="s",
            max_model_len=300,
            sessions=1,
            min_calls=2,
            max_calls=2,
        )
        self.assertIsNone(chosen)
        self.assertFalse(report["a"]["eligible"])
        self.assertFalse(report["b"]["eligible"])

    @staticmethod
    def reference_fixture():
        rows, reference = [], {"sessions": []}
        for sid, original_start in (("a", 10), ("b", 20)):
            calls = []
            for call in (1, 2):
                content = sid + str(call)
                ids = [ord(c) for c in content]
                gap = 0.0 if call == 1 else 0.125
                original_row = original_start + call - 1
                rows.append(
                    {
                        "session_id": sid,
                        "call": call,
                        "original_dataset_row": original_row,
                        "input": [{"role": "user", "content": content}],
                        "pre_gap": gap,
                    }
                )
                calls.append(
                    {
                        "call": call,
                        "dataset_row": original_row,
                        "prompt_tokens": len(ids),
                        "prompt_sha256": hash_ids(ids),
                        "pre_gap": gap,
                    }
                )
            reference["sessions"].append({"session_id": sid, "calls": calls})
        reference["sessions"].reverse()
        return rows, reference

    def test_reference_preserves_fixed_order_and_original_rows(self):
        rows, reference = self.reference_fixture()
        chosen, report = prepare_rows(
            rows,
            FakeTokenizer(),
            gap_unit="s",
            max_model_len=256,
            sessions=2,
            min_calls=2,
            max_calls=2,
            selection_seed=1,
            order_field="call",
            reference_manifest=reference,
        )
        self.assertTrue(all(s["eligible"] for s in report.values()))
        self.assertEqual([s["session_id"] for s in chosen], ["b", "a"])
        first = chosen[0]["calls"][0]
        self.assertEqual(
            (
                first["call_index"],
                first["source_call"],
                first["dataset_row"],
                first["original_dataset_row"],
            ),
            (0, 1, 2, 20),
        )

    def test_reference_hash_or_gap_mismatch_fails_without_resampling(self):
        for field, wrong in (("prompt_sha256", "0" * 64), ("pre_gap", 9.0)):
            with self.subTest(field=field):
                rows, reference = self.reference_fixture()
                reference["sessions"][0]["calls"][0][field] = wrong
                with self.assertRaisesRegex(ValueError, "differs from reference"):
                    prepare_rows(
                        rows,
                        FakeTokenizer(),
                        gap_unit="s",
                        max_model_len=256,
                        sessions=1,
                        min_calls=2,
                        max_calls=2,
                        selection_seed=1,
                        order_field="call",
                        reference_manifest=reference,
                    )

    def test_fixed_payload_no_cache_salt_and_no_prompt_identity(self):
        call = make_call()
        p = payload_for(call, "qwen")
        self.assertEqual(p["messages"], call["messages"])
        self.assertNotIn("cache_salt", p)
        self.assertEqual(
            (p["min_tokens"], p["max_tokens"], p["ignore_eos"]), (128, 128, True)
        )

    def test_real_lengths_and_response_token_ids_required(self):
        call, marker = make_call(), request_marker("r", "s", 0)
        good = response(call, marker)
        validate_response(good, call, marker)
        for mutate in (
            lambda r: r["usage"].update(completion_tokens=127),
            lambda r: r["prompt_token_ids"].__setitem__(0, 999),
            lambda r: r.update(id="chatcmpl-lost"),
        ):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_response(bad, call, marker)

    def test_causal_cap_all_arrive_t0_and_fixed_followup_inputs(self):
        sessions = [
            {"session_id": str(i), "calls": [make_call(0), make_call(1, 0.02)]}
            for i in range(6)
        ]
        original = copy.deepcopy(sessions)
        active, max_active, sent = set(), [0], []

        async def send(call, marker):
            sid = metadata(marker)["s"]
            begin = asyncio.get_running_loop().time()
            if call["call_index"] == 0:
                active.add(sid)
                max_active[0] = max(max_active[0], len(active))
            await asyncio.sleep(0.002)
            end = asyncio.get_running_loop().time()
            sent.append((sid, call["call_index"], begin, end))
            if call["call_index"] == 1:
                active.remove(sid)
            return response(call, marker), begin, end

        result = asyncio.run(
            replay_sessions(sessions, concurrency=2, seed=17, run_id="r", send=send)
        )
        self.assertEqual(max_active[0], 2)
        self.assertEqual(sessions, original)
        self.assertEqual(len(result["tasks"]), 6)
        self.assertTrue(all(t["arrival_seconds"] == 0 for t in result["tasks"]))
        self.assertTrue(any(t["admission_seconds"] > 0.02 for t in result["tasks"]))
        for sid in (str(i) for i in range(6)):
            pair = [x for x in sent if x[0] == sid]
            self.assertGreaterEqual(pair[1][2] - pair[0][3], 0.019)
        summary = client_summary(result)
        self.assertAlmostEqual(
            summary["mean_task_completion_seconds"],
            sum(t["completion_seconds"] for t in result["tasks"]) / 6,
        )
        self.assertEqual(summary["output_tokens"], 6 * 2 * OUTPUT_TOKENS)
        self.assertEqual(summary["configured_active_session_limit"], 2)
        self.assertEqual(summary["effective_worker_count"], 2)
        self.assertEqual(summary["observed_max_inflight_http_requests"], 2)

    def test_eight_sessions_reject_c16_and_http_concurrency_can_be_below_cap(self):
        sessions = prepared_fixture(8)["sessions"]
        sends = []

        async def send(call, marker):
            sends.append(marker)
            now = asyncio.get_running_loop().time()
            return response(call, marker), now, now

        with self.assertRaisesRegex(ValueError, "C=16 exceeds 8"):
            asyncio.run(
                replay_sessions(sessions, concurrency=16, seed=1, run_id="r", send=send)
            )
        self.assertEqual(sends, [])
        for cap in (4, 8):
            result = asyncio.run(
                replay_sessions(
                    sessions, concurrency=cap, seed=1, run_id="r", send=send
                )
            )
            self.assertEqual(result["configured_active_session_limit"], cap)
            self.assertEqual(result["effective_worker_count"], cap)
            self.assertEqual(result["observed_max_inflight_http_requests"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_8.json"
            path.write_text(json.dumps(prepared_fixture(8)))
            with self.assertRaisesRegex(ValueError, "C=16 exceeds 8"):
                # Fails before reading EngineCore config or invoking any HTTP.
                run(Namespace(manifest=str(path), concurrency=16))

    def test_manifest_validation_rejects_bad_budget_order_hash_and_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_8.json"
            with self.assertRaises(FileNotFoundError) as error:
                validate_manifest(path, expected_sessions=8, concurrency=4)
            self.assertIn("replay.py prepare", str(error.exception))
            self.assertIn("replay.py subset", str(error.exception))
            base = prepared_fixture(8)
            base["request_order_sha256"] = manifest_order_sha256(base)
            path.write_text(json.dumps(base))
            validate_manifest(
                path, expected_sessions=8, concurrency=8, expected_max_model_len=1024
            )
            for mutate in (
                lambda m: m.update(max_model_len=256),
                lambda m: m["sessions"].reverse(),
                lambda m: m["sessions"][0]["calls"][0].update(prompt_sha256="0" * 64),
                lambda m: m.update(selected_calls=9),
            ):
                changed = copy.deepcopy(base)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    validate_manifest(path, expected_sessions=8, concurrency=4)

    def test_subset_reuses_selection_seed_parent_hash_and_complete_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path, selection_path = (
                root / "manifest_50.json",
                root / "calibration_selection.json",
            )
            parent = prepared_fixture()
            parent_path.write_text(json.dumps(parent))
            parent_sha = file_sha256(parent_path)
            chosen = set(
                random.Random(20261001).sample(
                    sorted(s["session_id"] for s in parent["sessions"]), 8
                )
            )
            selected = [s for s in parent["sessions"] if s["session_id"] in chosen]
            selection = {
                "selected_sessions": 8,
                "selected_calls": 8,
                "calibration_parent_manifest_sha256": parent_sha,
                "calibration_selection_seed": 20261001,
                "session_ids": [s["session_id"] for s in selected],
                "max_input_tokens": 160,
                "fixed_output_tokens_total": 8 * 128,
            }
            selection_path.write_text(json.dumps(selection))
            out = root / "calibration_8.json"
            args = Namespace(
                manifest=str(parent_path),
                sessions=8,
                selection=str(selection_path),
                seed=None,
                out=str(out),
            )
            with redirect_stdout(io.StringIO()):
                subset(args)
            stats = validate_subset_selection(parent_path, out, selection_path)
            self.assertEqual(stats["session_ids"], selection["session_ids"])
            self.assertEqual(json.loads(out.read_text())["sessions"], selected)
            self.assertEqual(file_sha256(parent_path), parent_sha)
            with self.assertRaises(FileExistsError):
                subset(args)
            selection["calibration_parent_manifest_sha256"] = "0" * 64
            selection_path.write_text(json.dumps(selection))
            args.out = str(root / "invalid.json")
            with self.assertRaisesRegex(ValueError, "parent SHA256"):
                subset(args)
            self.assertFalse(Path(args.out).exists())
            selection["calibration_parent_manifest_sha256"] = parent_sha
            selection_path.write_text(json.dumps(selection))
            altered = json.loads(out.read_text())
            altered["sessions"][0]["calls"][0]["messages"][0]["content"] = "changed"
            out.write_text(json.dumps(altered))
            with self.assertRaisesRegex(
                ValueError, "changes parent order or complete calls"
            ):
                validate_subset_selection(parent_path, out, selection_path)

    def test_failure_invalidates_equal_work_comparison(self):
        result = {
            "tasks": [
                {"success": True, "arrival_seconds": 0, "completion_seconds": 5},
                {"success": False, "arrival_seconds": 0, "completion_seconds": 7},
            ],
            "requests": [{"success": True, "output_tokens": 128}, {"success": False}],
            "measurement_seconds": 7,
        }
        summary = client_summary(result)
        self.assertFalse(summary["valid_complete_run"])
        self.assertIsNone(summary["task_throughput_per_second"])
        self.assertIsNone(summary["mean_task_completion_seconds"])

    def test_history_joins_engine_adopted_hits_no_global_or_adjacent_lcp_proxy(self):
        marker = request_marker("r", "s", 1)
        client = [
            {
                "request_id": marker,
                "session_id": "s",
                "call_index": 1,
                "success": True,
                "input_tokens": 160,
                "output_tokens": 128,
                "response_cached_tokens_diagnostic": 128,
            }
        ]
        engine = [
            {
                "request_id": "chatcmpl-" + marker + "-12345678",
                "session_id": "s",
                "call_index": 1,
                "success": True,
                "input_tokens": 160,
                "output_tokens": 128,
                "history_reusable_tokens": 112,
                "adopted_hit_tokens": 64,
            }
        ]
        _, summary = join_engine_records(client, engine, "r")
        self.assertAlmostEqual(summary["historical_prefix_loss_rate"], 48 / 112)
        with self.assertRaises(ValueError):
            join_engine_records(client, [], "r")
        with self.assertRaises(ValueError):
            join_engine_records(client, engine * 2, "r")
        engine[0]["history_reusable_tokens"] = 0
        engine[0]["adopted_hit_tokens"] = 0
        _, summary = join_engine_records(client, engine, "r")
        self.assertIsNone(summary["historical_prefix_loss_rate"])

    def test_exact_capacity_guard(self):
        c = {
            "enable_prefix_caching": "True",
            "num_gpu_blocks": "100",
            "block_size": "16",
        }
        validate_cache(c, 16, 100)
        with self.assertRaises(ValueError):
            validate_cache(c, 16, 101)

    def test_actual_engine_paths_build_configuration_and_fresh_reset_required(self):
        evidence = {
            "version": "0.29.0",
            "mode": "native",
            "run_id": "r",
            "num_gpu_blocks": 100,
            "block_size": 16,
            "max_model_len": 1024,
            "core_paths": {str(i): f"/actual/import/{i}.py" for i in range(4)},
            "core_sha256": {str(i): "a" * 64 for i in range(4)},
            "model": "/actual/model",
            "dtype": "bfloat16",
            "async_scheduling": False,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "enable_chunked_prefill": True,
            "check_interval": 10,
            "first_cold_ratio": 0.2,
            "later_cold_ratio": 0.4,
            "private_suffix_only": True,
            "attention_backend": "TRITON_ATTN",
            "use_flashinfer_sampler": False,
            "enforce_eager": True,
        }
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "run_config.json").write_text(json.dumps(evidence))
            _, signature = engine_evidence(
                path,
                policy="native",
                block_size=16,
                num_gpu_blocks=100,
                max_model_len=1024,
            )
            self.assertEqual(len(signature), 64)
            evidence["first_cold_ratio"] = 0.1
            (root / "run_config.json").write_text(json.dumps(evidence))
            _, changed = engine_evidence(
                path,
                policy="native",
                block_size=16,
                num_gpu_blocks=100,
                max_model_len=1024,
            )
            self.assertNotEqual(signature, changed)
            for unsafe in (True, "0", 0, None):
                evidence["use_flashinfer_sampler"] = unsafe
                (root / "run_config.json").write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "use_flashinfer_sampler"):
                    engine_evidence(
                        path,
                        policy="native",
                        block_size=16,
                        num_gpu_blocks=100,
                        max_model_len=1024,
                    )
            evidence["use_flashinfer_sampler"] = False
            (root / "run_config.json").write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                engine_evidence(
                    path,
                    policy="progressive",
                    block_size=16,
                    num_gpu_blocks=100,
                    max_model_len=1024,
                )
            (root / "reset.jsonl").write_text(
                json.dumps({"run_id": "r", "success": True, "completed": 0}) + "\n"
            )
            self.assertIsNone(confirmed_policy_reset(path, 1, "r"))
            with (root / "reset.jsonl").open("a") as f:
                f.write(
                    json.dumps({"run_id": "r", "success": True, "completed": 0}) + "\n"
                )
            self.assertIsNotNone(confirmed_policy_reset(path, 1, "r"))

    def test_eviction_summary_validates_free_count_and_real_counts(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "eviction_summary.json").write_text(
                json.dumps(
                    {
                        "active_logical": 20,
                        "active_physical": 21,
                        "native_logical": 2,
                        "native_physical": 3,
                    }
                )
            )
            check = {
                "run_id": "r",
                "free_before": 100,
                "free_after": 100,
                "duration_ms": 0.1,
                "evicted_logical_blocks": 20,
                "evicted_physical_blocks": 21,
            }
            (root / "policy_checks.jsonl").write_text(json.dumps(check) + "\n")
            summary, _, _ = engine_metrics(
                path, run_id="r", checks_offset=0, policy="progressive"
            )
            self.assertEqual(summary["active_evicted_physical_blocks"], 21)
            with self.assertRaises(ValueError):
                engine_metrics(path, run_id="r", checks_offset=0, policy="native")
            check["free_after"] = 101
            (root / "policy_checks.jsonl").write_text(json.dumps(check) + "\n")
            with self.assertRaises(ValueError):
                engine_metrics(path, run_id="r", checks_offset=0, policy="progressive")

    def test_zero_eviction_and_absent_pressure_qualify_without_fabricating_success(
        self,
    ):
        summary = {
            "active_evicted_logical_blocks": 0,
            "native_evicted_physical_blocks": 0,
        }
        result = workload_qualification(
            summary, "progressive", [{"engine_preemptions": 0}]
        )
        self.assertFalse(result["proactive_eviction_exercised"])
        self.assertFalse(result["cache_pressure_observed"])
        self.assertEqual(len(result["qualification_notes"]), 2)
        result = workload_qualification(summary, "native", [{"engine_preemptions": 1}])
        self.assertTrue(result["cache_pressure_observed"])
        self.assertEqual(len(result["qualification_notes"]), 0)
        summary.update(
            active_evicted_logical_blocks=20, native_evicted_physical_blocks=10
        )
        result = workload_qualification(summary, "progressive", [])
        self.assertTrue(result["proactive_eviction_exercised"])
        self.assertTrue(result["cache_pressure_observed"])

    def test_pair_plan_alternates_and_rejects_changed_workload(self):
        runs = plan([17, 29, 43])
        self.assertEqual(
            [r["policy"] for r in runs],
            ["native", "progressive", "progressive", "native", "native", "progressive"],
        )
        config = {
            "seed": 17,
            "policy": "native",
            "model": "q",
            "request_model_alias": "launch-native-9dcb",
            "manifest_sha256": "x",
            "status": "completed",
            "num_gpu_blocks": 100,
            "block_size": 16,
            "max_model_len": 32000,
            "output_tokens": 128,
            "sessions": 50,
            "concurrency": 4,
            "engine_comparison_signature": "same-build-and-settings",
            "cache_start": "warmup_then_confirmed_reset",
        }
        summary = {
            "valid_complete_run": True,
            "engine_request_metadata_verified": True,
            "policy_free_count_invariant_verified": True,
            "output_tokens": 64000,
            "task_throughput_per_second": 1,
            "historical_prefix_loss_rate": 0.1,
            "mean_task_completion_seconds": 40,
        }
        a = {"config": config, "summary": summary, "task_order": ["a", "b"]}
        b = copy.deepcopy(a)
        b["config"]["policy"] = "progressive"
        b["config"]["request_model_alias"] = "launch-progressive-62ae"
        b["summary"]["historical_prefix_loss_rate"] = 0.2
        result = paired_results([a, b])[0]
        self.assertTrue(result["valid_pair"])
        self.assertFalse(result["competition_and_eviction_tested"])
        self.assertEqual(len(result["qualification_notes"]), 2)
        self.assertGreater(
            result["historical_prefix_loss_rate"]["progressive_minus_native"], 0
        )
        b["config"]["model"] = "a-different-real-model"
        self.assertFalse(paired_results([a, b])[0]["valid_pair"])
        b["config"]["model"] = "q"
        b["config"]["num_gpu_blocks"] = 101
        self.assertFalse(paired_results([a, b])[0]["valid_pair"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
