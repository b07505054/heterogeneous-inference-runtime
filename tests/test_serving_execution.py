import json

import pytest

from deployment.serving_execution import (
    CPUReplicaProfile, FunctionalClusterProfile, PlanOnlyServingRuntime,
    ReplicaPrefixCache, ServingDistributedCompiler, ServingExecutionPlan,
    ServingPlanError, ServingRequest, deserialize_serving_plan,
    deterministic_trace,
)


def cluster(n=4, cores=8):
    return FunctionalClusterProfile.local(n, total_logical_cores=cores,
                                          total_kv_capacity_bytes=1024 * 1024)


def test_default_topology_and_replica_isolation():
    c = cluster()
    assert [r.logical_core_budget for r in c.replicas] == [2, 2, 2, 2]
    rt = PlanOnlyServingRuntime(c, block_size=4, bytes_per_token=4)
    tokens = tuple(range(8))
    rt.replicas["replica-0"].cache.insert(tokens)
    assert rt.replicas["replica-0"].cache.lookup(tokens).matched_tokens == 8
    assert rt.replicas["replica-1"].cache.lookup(tokens).matched_tokens == 0
    rt.replicas["replica-2"].queue.append("busy")
    assert len(rt.replicas["replica-3"].queue) == 0


def test_block_prefix_lineage_partial_and_order():
    cache = ReplicaPrefixCache(4096, block_size=4, bytes_per_token=4)
    cache.insert((1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert cache.lookup((1, 2, 3, 4, 5, 6, 7, 8, 99)).matched_tokens == 8
    assert cache.lookup((1, 2, 3, 9, 5, 6, 7, 8)).matched_tokens == 0
    assert cache.lookup((1, 2, 3)).matched_tokens == 0


def test_cache_eviction_is_local():
    a = ReplicaPrefixCache(16, block_size=4, bytes_per_token=1)
    b = ReplicaPrefixCache(16, block_size=4, bytes_per_token=1)
    for i in range(6):
        a.insert(tuple(range(i * 100, i * 100 + 4)))
    assert a.evictions == 2
    assert b.evictions == 0


@pytest.mark.parametrize("policy", [
    "round_robin", "least_queue", "max_prefix_hit", "prefix_queue_cost"])
def test_routing_candidates_roundtrip_and_exact_dispatch(policy):
    c = cluster(2)
    rt = PlanOnlyServingRuntime(c, block_size=4, bytes_per_token=4)
    compiler = ServingDistributedCompiler(c)
    req = ServingRequest("r", tuple(range(9)), 2)
    selected = compiler.plan(req, rt.replicas, policy=policy,
                             operator_plan_id="operator-plan-1")
    loaded = deserialize_serving_plan(selected.serialize(), c)
    event = rt.execute(req, loaded)
    assert selected.selected_replica_id == loaded.selected_replica_id
    assert event["planned_replica_id"] == event["executed_replica_id"]
    assert all(v == 0 for v in rt.counters().values())


def test_prefix_queue_tradeoff():
    c = cluster(2)
    rt = PlanOnlyServingRuntime(c, block_size=4, bytes_per_token=4)
    comp = ServingDistributedCompiler(c)
    req = ServingRequest("x", tuple(range(64)), 4)
    rt.replicas["replica-0"].cache.insert(req.token_ids)
    rt.replicas["replica-0"].available_at_ms = 100
    prefix = comp.plan(req, rt.replicas, policy="max_prefix_hit")
    combined = comp.plan(req, rt.replicas, policy="prefix_queue_cost")
    assert prefix.selected_replica_id == "replica-0"
    assert combined.selected_replica_id == "replica-1"


def test_least_queue_ignores_expensive_recompute():
    c = cluster(2)
    rt = PlanOnlyServingRuntime(c, block_size=4, bytes_per_token=4)
    comp = ServingDistributedCompiler(c)
    req = ServingRequest("x", tuple(range(128)), 2)
    rt.replicas["replica-1"].cache.insert(req.token_ids)
    rt.replicas["replica-1"].available_at_ms = 1
    least = comp.plan(req, rt.replicas, policy="least_queue")
    combined = comp.plan(req, rt.replicas, policy="prefix_queue_cost")
    assert least.selected_replica_id == "replica-0"
    assert combined.selected_replica_id == "replica-1"


def test_nested_operator_provenance_keeps_namespaces_separate():
    c = cluster(2)
    rt = PlanOnlyServingRuntime(c)
    comp = ServingDistributedCompiler(c)
    req = ServingRequest("nested", tuple(range(32)), 1)
    plan = deserialize_serving_plan(comp.plan(
        req, rt.replicas, operator_plan_id="op-plan").serialize(), c)
    event = rt.execute(req, plan, lambda *_: {
        "operator_provenance": [{"operator_plan_id": "op-plan",
                                 "logical_worker_id": 0,
                                 "candidate_id": "native_avx2"}]})
    assert event["executed_replica_id"].startswith("replica-")
    assert event["operator_provenance"][0]["logical_worker_id"] == 0


def test_stress_1000_deterministic_events():
    def run():
        c = cluster()
        rt = PlanOnlyServingRuntime(c, block_size=16, bytes_per_token=32)
        comp = ServingDistributedCompiler(c)
        for req in deterministic_trace("hot_prefix", 1000):
            plan = deserialize_serving_plan(comp.plan(
                req, rt.replicas, policy="prefix_queue_cost").serialize(), c)
            rt.execute(req, plan)
        return [(x["request_id"], x["executed_replica_id"]) for x in rt.events], rt
    first, rt = run()
    second, _ = run()
    assert first == second
    assert len(first) == len(set(r for r, _ in first)) == 1000
    assert sum(r.completed for r in rt.replicas.values()) == 1000
    assert all(v == 0 for v in rt.counters().values())


def test_negative_plan_validation():
    c = cluster(2)
    rt = PlanOnlyServingRuntime(c)
    req = ServingRequest("r", tuple(range(16)), 1)
    compiler = ServingDistributedCompiler(c)
    payload = compiler.plan(req, rt.replicas).to_dict()
    cases = [
        ("selected_replica_id", "missing"),
        ("routing_policy", "unknown"),
        ("schema_version", 99),
        ("matched_tokens", 99),
    ]
    for field, bad in cases:
        changed = dict(payload)
        changed[field] = bad
        with pytest.raises(ServingPlanError):
            ServingExecutionPlan.from_dict(changed, c)
    with pytest.raises(ServingPlanError):
        rt.execute(ServingRequest("wrong", req.token_ids, 1),
                   ServingExecutionPlan.from_dict(payload, c))


def test_disabled_missing_backend_duplicate_and_negative_capacity():
    with pytest.raises(ServingPlanError):
        CPUReplicaProfile("x", 1, kv_capacity_bytes=-1)
    with pytest.raises(ServingPlanError):
        CPUReplicaProfile("x", 1, execution_backend="")
    p = CPUReplicaProfile("x", 1)
    with pytest.raises(ServingPlanError):
        FunctionalClusterProfile("c", 2, (p, p))
    disabled = FunctionalClusterProfile("c", 1, (
        CPUReplicaProfile("x", 1, enabled=False),))
    payload = {
        "plan_id": "p", "request_id": "r", "cluster_id": "c",
        "cluster_profile_version": 1, "selected_replica_id": "x",
        "routing_policy": "round_robin",
        "routing_candidate_id": "round_robin_v1",
        "cache_mode": "metadata_only", "prompt_tokens": 1,
        "matched_tokens": 0, "uncached_tokens": 1, "matched_blocks": 0,
        "predicted_cost": {"total_ms": 1.0},
        "selection_reason": "test"}
    with pytest.raises(ServingPlanError):
        ServingExecutionPlan.from_dict(payload, disabled)


def test_functional_tensor_mode_is_explicit():
    cache = ReplicaPrefixCache(1024, block_size=4, bytes_per_token=4,
                               mode="functional_tensor")
    cache.insert(tuple(range(4)), tensor_refs={})
    assert cache.snapshot()["mode"] == "functional_tensor"
