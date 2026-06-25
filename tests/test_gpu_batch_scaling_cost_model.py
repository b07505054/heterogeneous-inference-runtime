import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.llm_runtime_decision import (
    CostModel,
    MemoryPlanner,
    Request,
    RuntimeScheduler,
    load_gpu_batch_scaling_artifact,
)

ARTIFACT_PATH = ROOT / "results" / "llm_runtime_artifacts" / "gpu_decode_batch_scaling_gtx1650maxq.json"


def test_loader_parses_committed_measured_artifact():
    artifact = load_gpu_batch_scaling_artifact(str(ARTIFACT_PATH))

    assert artifact is not None
    assert len(artifact.decode_rows) > 0
    assert len(artifact.prefill_rows) > 0


def test_exact_decode_lookup_returns_measured_mean_ms():
    artifact = load_gpu_batch_scaling_artifact(str(ARTIFACT_PATH))

    measured = artifact.lookup_decode_ms(batch_size=2, context_tokens=128)

    assert measured == 16.879433


def test_off_grid_decode_lookup_chooses_nearest_row():
    artifact = load_gpu_batch_scaling_artifact(str(ARTIFACT_PATH))

    # batch_size=3 is equidistant between measured 2 and 4; sorted nearest
    # selection picks the smaller (2). context_tokens=300 is nearest to the
    # measured 128 within that batch-size bucket.
    measured = artifact.lookup_decode_ms(batch_size=3, context_tokens=300)
    expected = artifact.lookup_decode_ms(batch_size=2, context_tokens=128)

    assert measured == expected


def test_cost_model_without_artifact_preserves_current_formula():
    model = CostModel()

    prefill = model.predict_prefill_ms(512)
    decode = model.predict_decode_step_ms(4, 512)

    assert prefill == (model.prefill_base_ms + 512 * model.prefill_ms_per_token)
    assert decode == (
        model.decode_base_ms
        + 3 * model.decode_ms_per_batch_item
        + (512 / 1000.0) * model.decode_ms_per_1k_context
    )


def test_missing_path_returns_none_and_formula_fallback_works():
    artifact = load_gpu_batch_scaling_artifact(str(ROOT / "does" / "not" / "exist.json"))

    assert artifact is None

    model = CostModel(gpu_batch_scaling=artifact)
    assert model.gpu_batch_scaling is None
    assert model.predict_decode_step_ms(2, 128) == (
        model.decode_base_ms
        + 1 * model.decode_ms_per_batch_item
        + (128 / 1000.0) * model.decode_ms_per_1k_context
    )


def test_malformed_artifact_returns_none(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json")

    assert load_gpu_batch_scaling_artifact(str(bad_path)) is None

    wrong_type_path = tmp_path / "wrong_type.json"
    wrong_type_path.write_text('{"artifact_type": "something_else", "status": "measured"}')

    assert load_gpu_batch_scaling_artifact(str(wrong_type_path)) is None


def test_scheduler_runs_with_calibrated_cost_model():
    artifact = load_gpu_batch_scaling_artifact(str(ARTIFACT_PATH))
    cost_model = CostModel(gpu_batch_scaling=artifact)

    scheduler = RuntimeScheduler(
        policy="inflight_paged_kv_continuous_batching",
        cost_model=cost_model,
        memory=MemoryPlanner(
            total_blocks=128,
            block_size_tokens=16,
            kv_mb_per_block=3.125,
        ),
        rng=random.Random(7),
        max_decode_batch_size=8,
    )

    result = scheduler.run(
        [
            Request("req-1", prompt_tokens=128, output_tokens=8, arrival_ms=0.0),
            Request("req-2", prompt_tokens=512, output_tokens=8, arrival_ms=1.0),
        ]
    )

    assert result.completed_requests + result.rejected_requests == 2


def test_calibrated_decode_prediction_differs_from_uncalibrated():
    artifact = load_gpu_batch_scaling_artifact(str(ARTIFACT_PATH))

    formula_model = CostModel()
    calibrated_model = CostModel(gpu_batch_scaling=artifact)

    formula_decode = formula_model.predict_decode_step_ms(2, 128)
    calibrated_decode = calibrated_model.predict_decode_step_ms(2, 128)

    assert calibrated_decode == 16.879433
    assert formula_decode != calibrated_decode
