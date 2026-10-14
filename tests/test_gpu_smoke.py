"""Validate sampler log admission without sending HTTP or loading a GPU model."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/gpu_smoke.py"
SPEC = importlib.util.spec_from_file_location("pe_smoke_test", SOURCE)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


@pytest.mark.parametrize(
    "line,accepted",
    [
        ("run_config use_flashinfer_sampler=false\n", False),
        (
            (
                "(EngineCore_DP0 pid=123) FlashInfer top-p/top-k sampling disabled "
                "via VLLM_USE_FLASHINFER_SAMPLER=0.\n"
            ),
            True,
        ),
        (
            (
                "FlashInfer top-p/top-k sampling disabled via "
                "VLLM_USE_FLASHINFER_SAMPLER=0.\n"
                "Using FlashInfer for top-p & top-k sampling.\n"
            ),
            False,
        ),
    ],
)
def test_sampler_requires_initialization_log_not_only_environment(
    tmp_path, line, accepted
):
    path = tmp_path / "server.log"
    path.write_text(line)
    if accepted:
        evidence = SMOKE.sampler_log_evidence(path)
        assert evidence["snapshot_bytes"] == len(line.encode())
        assert len(evidence["disabled_initialization_lines"]) == 1
    else:
        with pytest.raises(ValueError):
            SMOKE.sampler_log_evidence(path)
