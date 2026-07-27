from perf_model.e2e3_test_helpers import synthetic_raw, synthetic_server_info
from perf_model.interference_model import (
    decode_interference_ms, decode_interference_ms_chunked_prefill_independent, UNCALIBRATED_INTERFERENCE,
)
from scripts.analyze_perf_model_results_e2e3 import (
    check_full_adherence, stall_alignment, row_summary, pair_off_on, evaluate_hypotheses, ranking_analysis,
)
from scripts.analyze_perf_model_results import build_model_features, build_hardware_features
from perf_model import phase_model


# --- 1. chunked-prefill state recorded in runtime configuration ---
def test_fixed_configuration_records_chunked_prefill_state():
    raw_off = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                             req0_gaps_s=[0.01], rest_submit_times=[0.06])
    raw_on = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=True, arrival_mode="burst",
                            req0_gaps_s=[0.01], rest_submit_times=[0.06])
    assert raw_off["fixed_configuration"]["enable_chunked_prefill"] is False
    assert raw_on["fixed_configuration"]["enable_chunked_prefill"] is True


# --- 2. requested-vs-resolved adherence for chunked prefill ---
def test_adherence_matches_when_resolved_equals_requested():
    server_info = synthetic_server_info(enable_chunked_prefill=False, max_num_seqs=2)
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                         req0_gaps_s=[0.01], rest_submit_times=[0.06], server_info=server_info)
    result = check_full_adherence(raw)
    assert result["chunked_prefill_match"] is True
    assert result["derived_config_adherent"] is True


def test_adherence_rejects_when_resolved_differs_from_requested():
    server_info = synthetic_server_info(enable_chunked_prefill=True, max_num_seqs=2)  # vLLM ignored the request
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                         req0_gaps_s=[0.01], rest_submit_times=[0.06], server_info=server_info)
    result = check_full_adherence(raw)
    assert result["chunked_prefill_match"] is False


# --- 11. fail-closed when resolved chunked-prefill state is missing ---
def test_adherence_fails_closed_when_server_info_missing():
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                         req0_gaps_s=[0.01], rest_submit_times=[0.06], server_info=None)
    result = check_full_adherence(raw)
    assert result["derived_config_adherent"] is False
    assert result["chunked_prefill_match"] is False
    assert "server_info_unavailable" in result["mismatches"]


# --- 3. staggered-arrival causal timing methodology (deterministic given a fixed timeline) ---
def test_stall_alignment_detects_stall_coinciding_with_new_request_submission():
    # req0 decodes smoothly (10ms gaps) then hits one big 400ms stall right when
    # the rest of the requests are submitted.
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="staggered",
                         req0_gaps_s=[0.010, 0.011, 0.400, 0.012], rest_submit_times=[0.05 + 0.010 + 0.011 + 0.05])
    findings = stall_alignment(raw)
    assert len(findings) == 1
    assert findings[0]["aligned_with_new_request_submission"] is True
    assert findings[0]["ratio_to_median"] > 3


def test_stall_alignment_rejects_unaligned_stall():
    # same big stall, but the rest requests are submitted long AFTER it -- not causally linked.
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="staggered",
                         req0_gaps_s=[0.010, 0.011, 0.400, 0.012], rest_submit_times=[5.0])
    findings = stall_alignment(raw)
    assert findings[0]["aligned_with_new_request_submission"] is False


# --- 6. off-vs-on result pairing ---
def test_pair_off_on_computes_percent_difference():
    model, hardware = build_model_features(), build_hardware_features()
    throughput = phase_model.UNCALIBRATED
    server_info_off = synthetic_server_info(enable_chunked_prefill=False, max_num_seqs=2)
    server_info_on = synthetic_server_info(enable_chunked_prefill=True, max_num_seqs=2)
    raw_off = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                             req0_gaps_s=[0.010, 0.170, 0.170], rest_submit_times=[0.06], server_info=server_info_off)
    raw_on = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=True, arrival_mode="burst",
                            req0_gaps_s=[0.010, 0.012, 0.011], rest_submit_times=[0.06], server_info=server_info_on)
    summaries = [row_summary(raw_off, model, hardware, throughput), row_summary(raw_on, model, hardware, throughput)]
    pairs = pair_off_on(summaries)
    assert len(pairs) == 1
    diff = pairs[0]["diff"]["request0_tpot_ms"]
    assert diff["percent"] < 0  # "on" should show lower mean tpot than "off" in this synthetic case
    assert pairs[0]["correctness_both_pass"] is True


# --- 7. joint candidate ranking ---
def test_ranking_analysis_picks_measured_global_best():
    model, hardware = build_model_features(), build_hardware_features()
    throughput = phase_model.UNCALIBRATED
    rows = []
    for cand, cp, gaps in ((1, False, [0.010]), (2, False, [0.170]), (2, True, [0.012])):
        si = synthetic_server_info(enable_chunked_prefill=cp, max_num_seqs=cand)
        raw = synthetic_raw(workload_id="C", max_num_seqs=cand, chunked_prefill=cp, arrival_mode="burst",
                             req0_gaps_s=gaps, rest_submit_times=[0.06], server_info=si)
        rows.append(row_summary(raw, model, hardware, throughput))
    ranking = ranking_analysis(rows)
    assert "C" in ranking
    assert ranking["C"]["measured_global_best"] in {"1_False", "2_True", "2_False"}


# --- 8. hypothesis-result generation ---
def test_evaluate_hypotheses_h1_supported_with_aligned_large_stalls():
    model, hardware = build_model_features(), build_hardware_features()
    throughput = phase_model.UNCALIBRATED
    si = synthetic_server_info(enable_chunked_prefill=False, max_num_seqs=2)
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="staggered",
                         req0_gaps_s=[0.010, 0.011, 0.400, 0.012], rest_submit_times=[0.081], server_info=si)
    summaries = [row_summary(raw, model, hardware, throughput)]
    result = evaluate_hypotheses(summaries, pair_off_on(summaries))
    assert result["H1"]["result"] == "SUPPORTED"


def test_evaluate_hypotheses_h1_inconclusive_with_no_staggered_data():
    model, hardware = build_model_features(), build_hardware_features()
    throughput = phase_model.UNCALIBRATED
    si = synthetic_server_info(enable_chunked_prefill=False, max_num_seqs=2)
    raw = synthetic_raw(workload_id="C", max_num_seqs=2, chunked_prefill=False, arrival_mode="burst",
                         req0_gaps_s=[0.010, 0.011], rest_submit_times=[0.06], server_info=si)
    summaries = [row_summary(raw, model, hardware, throughput)]
    result = evaluate_hypotheses(summaries, pair_off_on(summaries))
    assert result["H1"]["result"] == "INCONCLUSIVE"


# --- 9. interference-model term ---
def test_decode_interference_ms_gated_by_chunked_prefill():
    off = decode_interference_ms(chunked_prefill_enabled=False, active_decode_sequences=1,
                                  admitted_prefill_tokens=128, calibrated_cost_per_prefill_token_ms=3.0)
    on = decode_interference_ms(chunked_prefill_enabled=True, active_decode_sequences=1,
                                 admitted_prefill_tokens=128, calibrated_cost_per_prefill_token_ms=3.0)
    assert off == 384.0
    assert on == 0.0


def test_decode_interference_ms_zero_without_active_decode_or_prefill_work():
    assert decode_interference_ms(chunked_prefill_enabled=False, active_decode_sequences=0,
                                   admitted_prefill_tokens=128, calibrated_cost_per_prefill_token_ms=3.0) == 0.0
    assert decode_interference_ms(chunked_prefill_enabled=False, active_decode_sequences=1,
                                   admitted_prefill_tokens=0, calibrated_cost_per_prefill_token_ms=3.0) == 0.0


def test_chunked_prefill_independent_form_ignores_the_flag():
    val = decode_interference_ms_chunked_prefill_independent(
        active_decode_sequences=1, admitted_prefill_tokens=128, calibrated_cost_per_prefill_token_ms=3.0,
    )
    assert val == 384.0


def test_uncalibrated_interference_sentinel():
    assert UNCALIBRATED_INTERFERENCE.form == "unavailable"
    assert UNCALIBRATED_INTERFERENCE.calibrated_cost_per_prefill_token_ms is None


# --- 12. handling when metrics text is unavailable ---
def test_row_summary_handles_missing_metrics_text_gracefully():
    model, hardware = build_model_features(), build_hardware_features()
    throughput = phase_model.UNCALIBRATED
    si = synthetic_server_info(enable_chunked_prefill=False, max_num_seqs=1)
    raw = synthetic_raw(workload_id="A", max_num_seqs=1, chunked_prefill=False, arrival_mode="burst",
                         req0_gaps_s=[0.010], rest_submit_times=[], server_info=si)
    raw["post_warmup_metrics_text"] = None
    raw["final_metrics_text"] = None
    summary = row_summary(raw, model, hardware, throughput)
    assert summary["server_prefill_ms"] is None
    assert summary["server_decode_token_ms"] is None
    assert summary["server_queue_ms"] is None
