from array import array
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.paged_kv_runtime import build_paged_kv_runtime
from scripts import benchmark_paged_vs_contiguous_pi as bench
from scripts import profile_contiguous_vs_page_major as diff_profile
from scripts import profile_integrated_attention_overhead as integrated_profile


ROOT = Path(__file__).resolve().parents[1]
PAGED_PLAN = (
    ROOT
    / "artifacts"
    / "kv_selection_evaluation"
    / "raspberry_pi"
    / "compiler_plans"
    / "paged8_page_major.json"
)


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("paged_vs_contiguous")
    native_dir = root / "native"
    native_dir.mkdir()
    so = native_dir / "libattention_fp32.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(ROOT / "native/cpu_kernels/attention_fp32.cpp"),
            "-o",
            str(so),
        ],
        check=True,
    )
    return root, "native/libattention_fp32.so", hashlib.sha256(so.read_bytes()).hexdigest()


def compiler_payload(native_artifact):
    _, ref, digest = native_artifact
    payload = json.loads(PAGED_PLAN.read_text())
    for function in payload["function_plans"]:
        for decision in function["per_op_decisions"]:
            if "paged_kv_execution" in decision:
                decision["paged_kv_execution"]["pool_artifact_ref"] = ref
                decision["paged_kv_execution"]["pool_artifact_sha256"] = digest
            if "attention_execution" in decision:
                decision["attention_execution"]["artifact_ref"] = ref
                decision["attention_execution"]["artifact_sha256"] = digest
    return payload


def test_benchmark_configuration_shapes_are_equivalent(native_artifact):
    artifact_root, ref, digest = native_artifact
    plan = parse_execution_plan(compiler_payload(native_artifact))
    context = build_paged_kv_runtime(plan, artifact_root)
    contract = context.contract
    kv, att = bench.contiguous_contract(contract, ref, digest, prompt=7)

    assert kv["batch"] == contract.batch == 1
    assert kv["num_kv_heads"] == contract.num_kv_heads
    assert kv["head_dim"] == contract.head_dim
    assert kv["capacity_tokens"] == contract.maximum_logical_tokens
    assert kv["bytes_per_token"] == contract.bytes_per_token
    assert att["decode"]["entry_point"] == "hir_cpu_attention_decode_contiguous_kv_fp32"
    assert contract.paged_attention_entry_point == "hir_cpu_attention_decode_paged_kv_page_major_fp32"


def test_contiguous_and_paged_inputs_are_identical_and_use_one_reference(native_artifact):
    artifact_root, ref, digest = native_artifact
    plan = parse_execution_plan(compiler_payload(native_artifact))
    context = build_paged_kv_runtime(plan, artifact_root)

    result = bench.correctness_case(context.contract, artifact_root, ref, digest, context, 9, 123)

    assert result["same_inputs"] is True
    assert result["contiguous_vs_reference"]["passed"] is True
    assert result["paged_vs_reference"]["passed"] is True
    assert result["paged_vs_contiguous"]["passed"] is True
    assert result["block_table"] == [0, 1]
    context.page_manager.validate_invariants()


def test_invalid_correctness_results_are_excluded_before_benchmark(native_artifact):
    artifact_root, ref, digest = native_artifact
    plan = parse_execution_plan(compiler_payload(native_artifact))
    context = build_paged_kv_runtime(plan, artifact_root)
    result = bench.correctness_case(context.contract, artifact_root, ref, digest, context, 7, 5)

    result["paged_vs_reference"]["passed"] = False
    assert not all(
        x["passed"]
        for x in (
            result["contiguous_vs_reference"],
            result["paged_vs_reference"],
            result["paged_vs_contiguous"],
        )
    )


def test_memory_and_fragmentation_formulas_are_stable():
    payload = bench.memory_payload(
        tokens=9,
        bytes_per_token=128,
        capacity_tokens=64,
        bytes_per_page=1024,
        page_tokens=8,
    )

    assert payload["useful_kv_bytes"] == 1152
    assert payload["paged"]["allocated_pages"] == 2
    assert payload["paged"]["allocated_kv_bytes"] == 2048
    assert payload["paged"]["fragmentation_bytes"] == 896
    assert payload["paged"]["utilization_percent"] == pytest.approx(56.25)
    assert payload["contiguous"]["allocated_kv_bytes"] == 8192
    assert payload["contiguous"]["fragmentation_bytes"] == 7040
    assert payload["contiguous"]["utilization_percent"] == pytest.approx(14.0625)


def test_speedup_and_overhead_labels_are_mathematically_correct():
    faster = bench.label_ratio(2.0, 1.0)
    slower = bench.label_ratio(1.0, 1.5)
    same = bench.label_ratio(1.0, 1.02)

    assert faster["label"] == "speedup"
    assert faster["speedup"] == pytest.approx(2.0)
    assert faster["overhead_percent"] == pytest.approx(-50.0)
    assert slower["label"] == "overhead"
    assert slower["overhead_percent"] == pytest.approx(50.0)
    assert same["label"] == "statistically_indistinguishable"


def test_empty_or_insufficient_samples_are_rejected():
    with pytest.raises(ValueError, match="insufficient samples"):
        bench.stat([], warmups=1)


def test_stability_statistics_include_mad_and_rule():
    summary = bench.stat([1.0, 1.1, 0.9, 1.0, 1.05], warmups=2)

    assert "mad_ms" in summary
    assert "stability_rule" in summary
    assert summary["samples"] == 5


def test_timer_overhead_is_recorded():
    overhead = bench.timer_overhead(samples=5)

    assert overhead["summary"]["samples"] == 5
    assert len(overhead["raw_ms"]) == 5


def test_stage_breakdown_math_includes_remainder():
    result = bench.stage_breakdown(10.0, {"setup": 1.0, "qk": 3.0, "softmax": 2.0})

    assert result["stages"]["setup"]["percent_of_total"] == pytest.approx(10.0)
    assert result["stages"]["qk"]["percent_of_total"] == pytest.approx(30.0)
    assert result["stages"]["softmax"]["percent_of_total"] == pytest.approx(20.0)
    assert result["stages"]["unclassified_remainder"]["ms"] == pytest.approx(4.0)
    assert result["percent_sum"] == pytest.approx(100.0)


def test_softmax_stage_contract_names_are_attributable():
    stages = {
        "softmax_max_reduction": 1.0,
        "softmax_exp_sum": 2.0,
        "softmax_reciprocal": 0.25,
        "softmax_normalization_writeback": 0.0,
        "v_accumulation_fused_normalization": 3.0,
    }
    result = bench.stage_breakdown(10.0, stages)

    assert result["stages"]["softmax_normalization_writeback"]["ms"] == 0.0
    assert result["stages"]["softmax_exp_sum"]["percent_of_total"] == pytest.approx(20.0)


def test_differential_profiler_stage_names_are_symmetric():
    assert "qk_score_generation_plus_max" in diff_profile.STAGES
    assert "softmax_exp_sum" in diff_profile.STAGES
    assert "v_accumulation" in diff_profile.STAGES
    assert "block_table_validation_cache" in diff_profile.STAGES


def test_differential_profiler_delta_and_gap_accounting():
    payload = {
        "stage_rows": [{
            "valid_tokens": 64,
            "contiguous": {
                "validation_setup_ms": [1.0],
                "qk_score_generation_plus_max_ms": [2.0],
                "direct_exported_kernel_ms": [10.0],
                "modeled_total_ms": [3.0],
            },
            "page_major": {
                "validation_setup_ms": [1.5],
                "qk_score_generation_plus_max_ms": [4.0],
                "block_table_validation_cache_ms": [0.5],
                "direct_exported_kernel_ms": [14.0],
                "modeled_total_ms": [6.0],
            },
        }],
        "repetitions": 1,
        "inner_iterations": 1,
        "timer_overhead_ms": 0.0,
    }

    enriched = diff_profile.enrich(payload)
    row = enriched["stage_rows"][0]
    by_stage = {item["stage"]: item for item in row["stage_differential"]}

    assert row["direct_total_gap_ms"] == pytest.approx(4.0)
    assert by_stage["qk_score_generation_plus_max"]["delta_ms"] == pytest.approx(2.0)
    assert by_stage["qk_score_generation_plus_max"]["percent_of_direct_total_gap"] == pytest.approx(50.0)
    assert by_stage["block_table_validation_cache"]["contiguous_ms"] == 0.0
    assert row["largest_positive_delta_stage"]["stage"] == "qk_score_generation_plus_max"
    assert row["unclassified_differential_remainder_ms"] == pytest.approx(1.0)


def test_integrated_profiler_stage_delta_accounting():
    rows = integrated_profile.stage_delta_rows(
        {"ctypes_ffi_argument_preparation": 1.0, "output_handling_conversion": 2.0},
        {"ctypes_ffi_argument_preparation": 5.0, "output_handling_conversion": 3.0, "block_table_lookup": 2.0},
        10.0,
    )
    by_stage = {row["stage"]: row for row in rows}

    assert by_stage["ctypes_ffi_argument_preparation"]["delta_us"] == pytest.approx(4.0)
    assert by_stage["ctypes_ffi_argument_preparation"]["percent_of_integrated_gap"] == pytest.approx(40.0)
    assert by_stage["block_table_lookup"]["contiguous_us"] == 0.0
    assert by_stage["block_table_lookup"]["ratio"] is None


def test_integrated_profiler_allocation_audit_names_validation_hoist():
    audit = integrated_profile.allocation_audit()

    assert audit["paged_before_optimization_per_decode"]["block_table_array_allocated"] == 1
    assert audit["paged_before_optimization_per_decode"]["global_page_manager_invariant_scan"] == 1
    assert audit["paged_after_optimization_steady_decode"]["global_page_manager_invariant_scan"] == 0
    assert audit["paged_after_optimization_steady_decode"]["per_request_block_table_validation"] == 1


def test_integrated_profiler_timer_overhead_units_are_microseconds():
    overhead = integrated_profile.timer_overhead(samples=5)

    assert overhead["summary"]["samples"] == 5
    assert len(overhead["raw_us"]) == 5
    assert "median_us" in overhead["summary"]


def test_stage_breakdown_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="total_ms"):
        bench.stage_breakdown(0.0, {"setup": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        bench.stage_breakdown(1.0, {"setup": -1.0})


def test_aggregate_unstable_rows_counts_only_timing_rows():
    rows = [
        {"timing": {"summary": {"unstable": True}}},
        {"timing": {"summary": {"unstable": False}}},
        {"other": "metadata"},
    ]

    assert bench.aggregate_unstable_rows(rows) == 1


def test_json_artifact_schema_keys_are_stable(native_artifact, tmp_path):
    artifact_root, ref, digest = native_artifact
    plan = parse_execution_plan(compiler_payload(native_artifact))
    context = build_paged_kv_runtime(plan, artifact_root)
    row = bench.benchmark_token_count(context.contract, artifact_root, ref, digest, plan, 1, 1, ("contiguous",))[0]

    assert {"path", "operation", "token_count", "round", "timing", "memory", "allocator_telemetry"} <= set(row)
    assert {"summary", "raw_ms", "inner_iterations"} <= set(row["timing"])
    assert row["timing"]["summary"]["samples"] == bench.MICRO_SAMPLES


def test_plan_and_native_hashes_are_captured(native_artifact):
    artifact_root, _, digest = native_artifact
    so = artifact_root / "native/libattention_fp32.so"

    assert bench.sha256(so) == digest
    assert len(bench.sha256(PAGED_PLAN)) == 64


def test_no_runtime_selector_called_for_paged_no_redecision(native_artifact, monkeypatch):
    import deployment.attention_runtime as attention_runtime

    def fail(*args, **kwargs):
        raise AssertionError("selector called")

    monkeypatch.setattr(attention_runtime, "select_attention_plan", fail)
    artifact_root, _, _ = native_artifact
    plan = parse_execution_plan(compiler_payload(native_artifact))
    context = build_paged_kv_runtime(plan, artifact_root)

    assert context.contract.runtime_no_kernel_redecision is True


def test_compare_rejects_large_errors():
    got = array("f", [1.0, 2.0])
    ref = [1.0, 4.0]

    assert bench.compare(got, ref)["passed"] is False
