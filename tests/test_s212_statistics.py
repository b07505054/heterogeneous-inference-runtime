import json
from pathlib import Path
from scripts.analyze_serving_distributed_level2_12 import boot_mean,ci,q

ROOT=Path(__file__).resolve().parents[1]

def test_cluster_bootstrap_is_deterministic():
 assert boot_mean([.1,.2,.3],100)==boot_mean([.1,.2,.3],100)

def test_quantile_is_bounded_by_observations():
 assert q([3,1,2],.95)==3

def test_positive_interval_requires_positive_lower_bound():
 assert ci([1,1,1])[0]>0
 assert ci([-1,1])[0]<=0

def test_preregistration_locks_ten_repetitions_and_tail_tolerance():
 p=json.loads((ROOT/"results/runtime_paths/serving_distributed_level2_12/s2_12_preregistration.json").read_text())
 assert p["measured_repetitions_per_policy_trace_cell"]==10
 assert p["tail_noninferiority_tolerance"]==.005
 assert p["locked"] is True

def test_s212_registry_contains_unique_new_request_hashes():
 p=json.loads((ROOT/"results/runtime_paths/serving_distributed_level2_12/s2_12_trace_registry.json").read_text())
 hashes=[x["logical_request_hash"] for x in p["traces"]]
 assert len(hashes)==12==len(set(hashes))
