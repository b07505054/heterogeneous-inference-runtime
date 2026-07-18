import numpy as np
import pytest

from deployment.cpu_sharding import (
    PersistentCPUShardRuntime, ShardingPlanError, make_plan,
    materialize_mlir_example, propagate_linear_subgraph, uneven_ranges,
    validate_sharding, vllm_metadata_boundary,
)
from deployment.execution_plan.loader import parse_execution_plan
from tests.test_execution_plan_loader import _plan


def test_mesh_validation_and_mlir_representation():
    plan = make_plan()
    validate_sharding(plan)
    text = materialize_mlir_example(plan)
    assert 'hir.sharding.mesh' in text and 'size = 8 : i64' in text
    bad = make_plan()
    bad["operator_sharding"]["mesh_axis"] = "missing"
    with pytest.raises(ShardingPlanError, match="nonexistent"):
        validate_sharding(bad)


def test_uneven_partition_is_balanced_and_complete():
    ranges = uneven_ranges(11, 8)
    assert ranges[0] == (0, 2) and ranges[-1] == (10, 11)
    assert [i for a, b in ranges for i in range(a, b)] == list(range(11))


def test_propagation_and_explicit_fallback():
    got = propagate_linear_subgraph([
        {"op": "linear"}, {"op": "bias_add"}, {"op": "relu"},
        {"op": "reshape", "dimension_mapping_unambiguous": True},
        {"op": "attention"},
    ], {"strategy": "split_m", "provenance": "user_specified"})
    assert all(x["sharding"]["strategy"] == "split_m" for x in got[:4])
    assert got[-1]["sharding"]["provenance"] == "fallback_replicated"


@pytest.mark.parametrize("strategy", ["replicated", "split_m",
                                      "row_parallel", "column_parallel"])
@pytest.mark.parametrize("m,k,n", [(16, 24, 12), (11, 13, 17)])
def test_all_strategies_match_complete_reference(strategy, m, k, n):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(m, k)).astype(np.float32)
    w = rng.normal(size=(k, n)).astype(np.float32)
    b = rng.normal(size=n).astype(np.float32)
    with PersistentCPUShardRuntime(make_plan(strategy=strategy)) as runtime:
        got, _ = runtime.linear(x, w, b, "relu")
        for _ in range(3):
            repeated, _ = runtime.linear(x, w, b, "relu")
            np.testing.assert_allclose(repeated, got, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(got, np.maximum(x @ w + b, 0),
                               rtol=1e-5, atol=2e-5)


def test_mixed_shapes_affinity_and_diagnostics():
    with PersistentCPUShardRuntime(make_plan()) as runtime:
        for m in (1, 7, 32):
            x = np.ones((m, 9), np.float32)
            y, _ = runtime.linear(x, np.ones((9, 5), np.float32))
            assert y.shape == (m, 5)
        assert len(runtime.affinity) == 8
        assert all(not affinity or len(affinity) == 1
                   for affinity in runtime.affinity.values())
        with pytest.raises(ShardingPlanError, match="shape mismatch"):
            runtime.linear(np.ones((2, 3), np.float32),
                           np.ones((4, 5), np.float32))


def test_vllm_boundary_is_explicitly_partial():
    metadata = vllm_metadata_boundary(
        {"request_id": "r1", "prompt_tokens": 32}, make_plan())
    assert metadata["integration_class"] == "partial_plan_driven_sidecar"
    assert metadata["generated_tokens_depend_on_sharded_result"] is False


def test_execution_plan_optional_sharding_round_trip_and_compatibility():
    legacy = parse_execution_plan(_plan())
    assert legacy.global_decisions.cpu_sharding == {}
    payload = _plan()
    payload["global_decisions"]["cpu_sharding"] = make_plan()
    parsed = parse_execution_plan(payload)
    assert parsed.global_decisions.cpu_sharding == payload["global_decisions"]["cpu_sharding"]
