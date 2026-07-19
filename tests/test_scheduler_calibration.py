import copy
import math

import pytest

from deployment.scheduler_calibration import (
    DatasetSplit, ObjectiveProfile, SchedulerPolicyPlan, SchedulerSelectorV1,
    evaluate_policy, make_objective, objective_profiles, request_objective,
    scheduler_features, terminal_cost)
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState,
    SchedulerCompiler, SchedulerProfile, run_scheduler)


def state():
    profile = SchedulerProfile(max_num_seqs=4, max_num_batched_tokens=8,
                               max_prefill_chunk_tokens=4,
                               balanced_decode_reservation=2)
    result = ReplicaSchedulerState("replica-0", profile)
    decode = RequestExecutionState("d", "s-d", "replica-0", 0, 8, 8, 4)
    prefill = RequestExecutionState("p", "s-p", "replica-0", 0, 20, 0, 2)
    result.ingest(decode); result.ingest(prefill)
    return result


@pytest.mark.parametrize("name", list(objective_profiles()))
def test_objective_profiles_are_valid(name):
    objective = make_objective(name)
    objective.validate()
    assert objective.objective_version == "request_objective_v1"


def test_invalid_objective_and_normalization_and_weight():
    with pytest.raises(ServingPlanError):
        make_objective("missing")
    base = make_objective()
    with pytest.raises(ServingPlanError):
        ObjectiveProfile(base.profile_id, 0, 1, 1, 1, base.weights).validate()
    weights = dict(base.weights); weights["ttft"] = -1
    with pytest.raises(ServingPlanError):
        ObjectiveProfile(base.profile_id, 1, 1, 1, 1, weights).validate()
    weights["ttft"] = math.nan
    with pytest.raises(ServingPlanError):
        ObjectiveProfile(base.profile_id, 1, 1, 1, 1, weights).validate()


def test_dataset_splits_reject_overlap_and_seed_reuse():
    good = DatasetSplit(("cal",), ("dev",), ("held",), ("stress",),
                        ("qwen",), {"cal":1,"dev":2,"held":3,"stress":4,"qwen":5})
    good.validate()
    with pytest.raises(ServingPlanError):
        DatasetSplit(("x",), (), ("x",), (), (), {}).validate()
    with pytest.raises(ServingPlanError):
        DatasetSplit(("x",), (), ("y",), (), (), {"x":1,"y":1}).validate()


def test_features_include_age_gap_remaining_work_topology():
    s = state(); s.clock_ms = 5
    features = scheduler_features(s, replica_core_budget=4,
                                  prefix_hit_tokens=8, kv_pressure=.2)
    assert features["decode_ready_count"] == 1
    assert features["total_remaining_prefill_tokens"] == 20
    assert features["replica_core_budget"] == 4
    assert features["prefix_hit_token_total"] == 8


def test_horizon_evaluator_does_not_mutate_live_state():
    base = state()
    s = ReplicaSchedulerState("replica-0", SchedulerProfile(
        max_num_seqs=1, max_num_batched_tokens=8,
        max_prefill_chunk_tokens=4, balanced_decode_reservation=1))
    for request in base.requests.values():
        s.ingest(copy.deepcopy(request))
    before = copy.deepcopy(s)
    result = evaluate_policy(s, "decode_first", make_objective(), horizon=2)
    assert s == before
    assert result["evaluation_level"] == "fixed_horizon_predicted_outcome"
    assert result["simulated_steps"] == 2
    assert result["excluded_request_delay"]


def test_horizons_and_invalid_continuation():
    s = state()
    values = [evaluate_policy(s, "decode_first", make_objective(),
                              horizon=h)["value"] for h in (1,2,4,8)]
    assert len(values) == 4 and all(math.isfinite(x) for x in values)
    with pytest.raises(ServingPlanError):
        evaluate_policy(s, "decode_first", make_objective(), horizon=0)
    with pytest.raises(ServingPlanError):
        evaluate_policy(s, "decode_first", make_objective(), horizon=1,
                        continuation_policy="unknown")


def test_terminal_cost_accounts_for_remaining_work():
    s = state()
    with_terminal = evaluate_policy(s, "decode_first", make_objective(),
                                    horizon=1, terminal=True)
    without = evaluate_policy(s, "decode_first", make_objective(),
                              horizon=1, terminal=False)
    assert with_terminal["terminal"]["normalized_terminal_cost"] > 0
    assert with_terminal["value"] > without["value"]


def test_full_trace_request_objective_and_oracle_legality():
    s = state()
    rows = {p:evaluate_policy(s,p,make_objective(),horizon=None)
            for p in ("decode_first","prefill_first","chunked_balanced","slo_aware")}
    assert all(x["evaluation_level"] == "full_trace_request_level"
               for x in rows.values())
    assert all(x["terminal"]["normalized_terminal_cost"] == 0
               for x in rows.values())


def test_v1_policy_plan_and_roundtrip_fields():
    s = state()
    selector = SchedulerSelectorV1(make_objective())
    plan = selector.select(s, valid_for_steps=4, trigger="trace_start")
    assert plan.policy_id in ("decode_first","prefill_first",
                              "chunked_balanced","slo_aware")
    assert plan.selector_version == "scheduler_selector_v1_request_level"
    assert '"plan_kind": "scheduler_policy_epoch"' in plan.serialize()
    assert selector.records[0]["evaluations"]


def test_policy_plan_rejects_unknown_candidate():
    with pytest.raises(ServingPlanError):
        SchedulerPolicyPlan("p","replica-0","missing","balanced_interactive",
                            0,1,"trace_start","v1","v1",1)


def test_selector_v1_preserves_plan_only_runtime():
    s = state(); selector = SchedulerSelectorV1(make_objective())
    policy = selector.select(s).policy_id
    rt = PlanOnlySchedulerRuntime()
    run_scheduler(s, SchedulerCompiler(), rt, policy=policy)
    assert all(v == 0 for v in rt.counters().values())
    assert all(r.finished for r in s.requests.values())


def test_v0_is_not_modified_by_v1():
    s = state()
    before = SchedulerCompiler().compile(s).candidate_id
    SchedulerSelectorV1(make_objective()).select(s)
    after = SchedulerCompiler().compile(s).candidate_id
    assert before == after
