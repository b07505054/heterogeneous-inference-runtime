import copy
import math

import pytest

from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState,
    ScheduleItem, ScheduleStepPlan, SchedulerCompiler, SchedulerProfile,
    deserialize_schedule_plan, run_scheduler,
)


def request(name="r", prompt=20, matched=0, output=3, arrival=0, replica="replica-0"):
    return RequestExecutionState(name, f"serving-{name}", replica, arrival,
                                 prompt, matched, output)


def state(profile=None, requests=()):
    s = ReplicaSchedulerState("replica-0", profile or SchedulerProfile(
        max_num_seqs=4, max_num_batched_tokens=8,
        max_prefill_chunk_tokens=4, balanced_decode_reservation=2))
    for r in requests:
        s.ingest(r)
    return s


def test_request_state_invariants_and_full_hit():
    r = request(prompt=16, matched=16)
    r.make_ready()
    assert r.phase == "DECODE" and r.prefill_remaining_tokens == 0
    bad = request(prompt=4)
    bad.decode_completed_tokens = 1
    with pytest.raises(ServingPlanError, match="decode began"):
        bad.validate()


@pytest.mark.parametrize("prompt,expected", [(3, 1), (8, 2), (9, 3), (33, 9)])
def test_chunked_prefill_exact_coverage(prompt, expected):
    s = state(requests=[request(prompt=prompt, output=1)])
    run_scheduler(s, SchedulerCompiler(), PlanOnlySchedulerRuntime(),
                  policy="chunked_balanced")
    r = s.requests["r"]
    assert len(r.prefill_chunks) == expected
    assert r.prefill_chunks[0][0] == 0
    assert r.prefill_chunks[-1][1] == prompt
    assert all(a[1] == b[0] for a, b in zip(r.prefill_chunks, r.prefill_chunks[1:]))


def test_partial_prefix_chunks_only_uncached_suffix():
    s = state(requests=[request(prompt=20, matched=8, output=1)])
    run_scheduler(s, SchedulerCompiler(), PlanOnlySchedulerRuntime(),
                  policy="chunked_balanced")
    assert s.requests["r"].prefill_chunks == [(0, 4), (4, 8), (8, 12)]


def test_decode_and_prefill_coexist_in_step():
    a = request("a", prompt=0, output=3)
    a.phase = "DECODE"
    b = request("b", prompt=20, output=1)
    s = state(requests=[a, b])
    plan = SchedulerCompiler().compile(s, forced_policy="chunked_balanced")
    assert {x.phase for x in plan.items} == {"decode", "prefill"}
    assert plan.scheduled_tokens <= 8


def test_dynamic_arrival_and_batch_membership():
    a = request("a", prompt=0, output=4)
    a.phase = "DECODE"
    b = request("b", prompt=4, output=2, arrival=.2)
    s = state(requests=[a, b])
    rt = PlanOnlySchedulerRuntime()
    run_scheduler(s, SchedulerCompiler(), rt, policy="chunked_balanced")
    memberships = [set(i["request_id"] for i in x["items"]) for x in rt.events]
    assert memberships[0] == {"a"}
    assert any("a" in x and "b" in x for x in memberships)
    assert any("a" not in x and "b" in x for x in memberships)


@pytest.mark.parametrize("policy", [
    "decode_first", "prefill_first", "chunked_balanced", "slo_aware"])
def test_candidates_deterministic_legal_roundtrip(policy):
    a = request("a", prompt=0, output=2)
    a.phase = "DECODE"
    s = state(requests=[a, request("b", prompt=12, output=1)])
    compiler = SchedulerCompiler()
    p = compiler.compile(s, forced_policy=policy)
    loaded = deserialize_schedule_plan(p.serialize(), s)
    assert p.to_dict() == loaded.to_dict()
    p2 = SchedulerCompiler().compile(s, forced_policy=policy)
    assert [x.to_dict() for x in p.items] == [x.to_dict() for x in p2.items]


def test_runtime_exact_plan_zero_override_counters():
    s = state(requests=[request(prompt=8, output=1)])
    p = SchedulerCompiler().compile(s, forced_policy="prefill_first")
    rt = PlanOnlySchedulerRuntime()
    rt.execute(s, deserialize_schedule_plan(p.serialize(), s))
    assert all(v == 0 for v in rt.counters().values())
    assert rt.events[0]["scheduled_tokens"] == p.scheduled_tokens


def test_budget_one_token_prefill():
    p = SchedulerProfile(max_num_seqs=2, max_num_batched_tokens=1,
                         max_prefill_chunk_tokens=1,
                         balanced_decode_reservation=0)
    s = state(p, [request(prompt=2, output=1)])
    plan = SchedulerCompiler().compile(s, forced_policy="chunked_balanced")
    assert plan.items[0].token_count == 1


def mutate_plan(plan, **changes):
    value = plan.to_dict()
    value.update(changes)
    return value


def test_negative_duplicate_budget_sequence_and_schema():
    s = state(requests=[request(prompt=8, output=1)])
    p = SchedulerCompiler().compile(s, forced_policy="prefill_first")
    cases = [
        {"items": [p.items[0].to_dict(), p.items[0].to_dict()],
         "scheduled_sequences": 2, "scheduled_tokens": p.scheduled_tokens * 2,
         "unused_tokens": p.maximum_tokens - p.scheduled_tokens * 2},
        {"scheduled_tokens": 99, "unused_tokens": 0},
        {"maximum_sequences": 0},
        {"schema_version": 9},
        {"step_id": 3},
        {"predicted_cost": {"total_score": math.nan}},
    ]
    for change in cases:
        with pytest.raises(ServingPlanError):
            ScheduleStepPlan.from_dict(mutate_plan(p, **change), s)


def test_negative_offsets_chunks_decode_and_foreign_replica():
    s = state(requests=[request(prompt=8, output=1)])
    p = SchedulerCompiler().compile(s, forced_policy="prefill_first")
    base = p.to_dict()
    for item in (
        {"request_id": "r", "phase": "prefill", "token_start": 1, "token_count": 4},
        {"request_id": "r", "phase": "prefill", "token_start": 0, "token_count": 5},
        {"request_id": "r", "phase": "prefill", "token_start": 0, "token_count": 0},
        {"request_id": "r", "phase": "decode", "token_start": 0, "token_count": 1},
        {"request_id": "logical-worker-0", "phase": "prefill",
         "token_start": 0, "token_count": 1},
    ):
        value = dict(base)
        value["items"] = [item]
        value["scheduled_tokens"] = item["token_count"]
        value["unused_tokens"] = value["maximum_tokens"] - item["token_count"]
        with pytest.raises(ServingPlanError):
            ScheduleStepPlan.from_dict(value, s)
    value = dict(base)
    value["replica_id"] = "replica-1"
    with pytest.raises(ServingPlanError):
        ScheduleStepPlan.from_dict(value, s)


def test_finished_request_cannot_be_rescheduled():
    r = request(prompt=0, output=0)
    r.phase = "FINISHED"
    s = state(requests=[r])
    value = ScheduleStepPlan(
        "p", "replica-0", s.step_id, s.version, "decode_first",
        "decode_first_v1", 8, 1, 7, 4, 1,
        (ScheduleItem("r", "decode", 0, 1),), {"total_score": 1}).to_dict()
    with pytest.raises(ServingPlanError):
        ScheduleStepPlan.from_dict(value, s)


def test_compiler_selected_and_provenance_chain():
    a = request("a", prompt=0, output=1)
    a.phase = "DECODE"
    s = state(requests=[a, request("b", prompt=6, output=1)])
    compiler, rt = SchedulerCompiler(), PlanOnlySchedulerRuntime()
    plan = compiler.compile(s)
    seen = []
    rt.execute(s, deserialize_schedule_plan(plan.serialize(), s),
               lambda req, item, schedule: {
                   "model_invocation_id": f"invoke-{req.request_id}",
                   "operator_plan_id": f"operator-{req.request_id}",
                   "operator_provenance": [{"logical_worker_id": 0}]})
    for event in rt.events[0]["items"]:
        seen.append((event["request_id"], event["model_invocation_id"],
                     event["operator_plan_id"]))
    assert plan.selection_mode == "compiler_selected"
    assert seen and compiler.traces[0]["selected_candidate_id"] == plan.candidate_id


def test_1000_requests_and_more_than_10000_steps():
    profile = SchedulerProfile(max_num_seqs=4, max_num_batched_tokens=2,
                               max_prefill_chunk_tokens=1,
                               balanced_decode_reservation=1)
    requests = [request(f"r{i}", prompt=10, output=11, arrival=i * .001)
                for i in range(1000)]
    s = state(profile, requests)
    rt = PlanOnlySchedulerRuntime()
    run_scheduler(s, SchedulerCompiler(), rt, policy="chunked_balanced",
                  max_steps=30000)
    assert len(rt.events) >= 10000
    assert len(s.finished_ids) == len(set(s.finished_ids)) == 1000
    assert all(v == 0 for v in rt.counters().values())
    assert all(r.finished for r in s.requests.values())
