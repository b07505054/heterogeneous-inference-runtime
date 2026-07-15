import pytest
from deployment.kv_selection_evaluation import OBJECTIVES, admission_count, regret, select_measured

def rows():
    return [
      {"target_id":"pi","workload_id":"W1","candidate_id":"cpu_contiguous_kv_fp32_v1","page_tokens":None,"correctness_passed":True,"append_decode_p95_ms":1.0,"request_owned_bytes":100,"internal_fragmentation_ratio":0.0},
      {"target_id":"pi","workload_id":"W1","candidate_id":"cpu_paged_kv_fp32_v1","page_tokens":8,"correctness_passed":True,"append_decode_p95_ms":1.1,"request_owned_bytes":40,"internal_fragmentation_ratio":.1},
    ]
def test_objective_weights_are_explicit():
    assert set(OBJECTIVES)=={"latency","memory_efficiency","balanced"}
    assert all("max_latency_regression" in x for x in OBJECTIVES.values())
def test_exact_lookup_and_unseen_fallback():
    assert select_measured(rows(),target_id="pi",workload_id="W1",objective="latency")["candidate_id"].endswith("contiguous_kv_fp32_v1")
    assert select_measured(rows(),target_id="host",workload_id="W1",objective="latency")["selection_reason"].startswith("unseen_exact")
def test_memory_and_balanced_use_declared_scores():
    assert select_measured(rows(),target_id="pi",workload_id="W1",objective="memory_efficiency")["page_tokens"]==8
    assert "weights" in select_measured(rows(),target_id="pi",workload_id="W1",objective="balanced")
def test_wrong_target_or_failed_correctness_is_not_legal():
    bad=rows();bad[0]["correctness_passed"]=False
    assert select_measured(bad,target_id="pi",workload_id="W1",objective="latency")["page_tokens"]==8
def test_regret_and_admission_formulas():
    a,b=rows();a["objective_score"]=2;b["objective_score"]=1
    assert regret(a,b)["relative_latency_regret"]==pytest.approx(1/1.1-1)
    assert admission_count(1024,[(8,1.0)],bytes_per_token=8,contiguous_capacity=16,page_tokens=8)=={"contiguous":8,"paged_formula":16}
