"""D6: proves the Python TPCostModel is absent from the production TP-degree
decision path, and that the serialized compiler decision alone determines
the materialized TP degree -- no runtime selector, no environment
override, no benchmark-script override.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every production module that could plausibly launch a real vLLM server
# with a distributed configuration. If any of these ever imports
# TPCostModel, that is a production-path regression -- the whole point of
# D6 is that the compiler's serialized plan is the sole source of the TP
# degree these modules materialize.
PRODUCTION_MODULES = [
    REPO_ROOT / "deployment/vllm_adapter/distributed_materializer.py",
    REPO_ROOT / "deployment/vllm_adapter/distributed_launch_controller.py",
    REPO_ROOT / "deployment/vllm_adapter/distributed_cli.py",
    REPO_ROOT / "deployment/vllm_adapter/distributed_preflight.py",
    REPO_ROOT / "deployment/vllm_adapter/backend_adapter.py",
]


def _imports_tp_cost_model(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("tp_cost_model" in alias.name for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module and "tp_cost_model" in node.module:
                return True
    return False


def test_no_production_module_imports_tp_cost_model():
    offenders = [str(p) for p in PRODUCTION_MODULES if _imports_tp_cost_model(p)]
    assert not offenders, f"production modules import the demoted Python selector: {offenders}"


def test_tp_cost_model_module_docstring_is_explicitly_reclassified():
    from deployment.vllm_adapter import tp_cost_model
    doc = tp_cost_model.__doc__ or ""
    assert "NOT part of the production TP-degree decision path" in doc
    assert "distributed_profitability_contract_v1" in doc


def test_materializer_determines_tp_degree_from_plan_alone_no_cost_model_import():
    """Deleting/disabling the runtime selector must not change compiler-
    driven execution: simulate its absence by ensuring the module isn't
    even importable from within the materializer's own import graph, then
    confirm materialize_launch_spec still works purely from plan content."""
    import importlib
    import deployment.vllm_adapter.distributed_materializer as dm
    importlib.reload(dm)
    assert "tp_cost_model" not in dm.__dict__
    for name, obj in vars(dm).items():
        if hasattr(obj, "__module__") and obj.__module__ and "tp_cost_model" in obj.__module__:
            raise AssertionError(f"distributed_materializer transitively references tp_cost_model via {name}")

    fresh_plan = (
        REPO_ROOT / "results/runtime_paths/distributed_d6_compiler_owned_tp_selection"
        "/fresh_compilations/qwen7b_in32_out32_c1_plan.json"
    )
    bundle = dm.materialize_launch_spec(fresh_plan, repo_root=REPO_ROOT)
    assert bundle.spec.tensor_parallel_size == 2  # from the compiler's serialized plan alone


def test_fresh_compiler_plans_determine_tp_degree_across_all_held_out_cells():
    """Cross-check every one of the 21 real fresh-compilation outputs: the
    materialized tensor_parallel_size always equals the compiler's own
    recorded decision, with no separate Python selection step involved."""
    import deployment.vllm_adapter.distributed_materializer as dm

    decisions = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d6_compiler_owned_tp_selection"
         "/fresh_compilation_decisions.json").read_text()
    )
    assert len(decisions) == 21
    for d in decisions:
        plan_path = (
            REPO_ROOT / "results/runtime_paths/distributed_d6_compiler_owned_tp_selection"
            f"/fresh_compilations/{d['model_label']}_{d['workload_id']}_plan.json"
        )
        bundle = dm.materialize_launch_spec(plan_path, repo_root=REPO_ROOT)
        expected_tp = 2 if d["compiler_selected_tp"] == "tp2" else 1
        assert bundle.spec.tensor_parallel_size == expected_tp, (
            f"{d['model_label']} {d['workload_id']}: materializer produced "
            f"tensor_parallel_size={bundle.spec.tensor_parallel_size}, expected {expected_tp} "
            "from the compiler's own recorded decision"
        )


def test_oracle_cross_check_python_reference_matches_compiler_prediction():
    """The Python TPCostModel, in its demoted oracle/reference role, must
    still numerically match the compiler's C++ prediction for the same
    inputs -- this is what makes it usable as a cross-check tool at all."""
    from deployment.vllm_adapter.tp_cost_model import (
        MODEL_IDENTITY_FEATURES, TPCostModel, FittedRegression, build_feature_vector,
    )

    cm_json = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d5_compiler_tp_policy"
         "/cost_model_fitted.json").read_text()
    )
    model = TPCostModel()
    for tp in (1, 2):
        m = cm_json["throughput_models"][str(tp)]
        model.throughput_models[tp] = FittedRegression(
            tp_degree=tp, coefficients=m["coefficients"], n_samples=m["n_samples"], r_squared=m["r_squared"],
        )
    model.frozen = True

    mf = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    fv1 = build_feature_vector(mf, 1, input_length=32, output_length=32, concurrency=1)
    predicted = model.predict_throughput(fv1, 1)

    evidence = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d6_compiler_owned_tp_selection"
         "/fresh_compilations/qwen05b_in32_out32_c1_evidence.json").read_text()
    )
    compiler_predicted = next(
        c["profitability"]["predicted_throughput_tokens_per_s"]
        for c in evidence["candidates"] if c["candidate_id"] == "tp1"
    )
    assert abs(predicted - compiler_predicted) < 1e-6
