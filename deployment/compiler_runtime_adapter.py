"""CompilerRuntimeAdapter: bridge from compiler ServingExecutionPlan to runtime IR.

Consumes a ServingExecutionPlan-like dict produced by the
ml-graph-compiler-runtime serving-optimization-pipeline and converts it to a
list of RuntimeExecutionPlan instances for use by ExecutionEngine.

Architecture:
  ServingExecutionPlan dict
    → CompilerRuntimeAdapter.from_serving_plan()
    → per-function validation + contamination rejection
    → RuntimeExecutionPlanAdapter.from_dict()   (per-function converter)
    → list[RuntimeExecutionPlan]
    → ExecutionEngine.execute(plan)
    → RuntimeResult

Design boundaries:
- This adapter never creates RuntimeResult.
- This adapter never dispatches a backend.
- This adapter never mutates RuntimeExecutionPlan.
- Format-specific deserialization (JSON file, FlatBuffer, protobuf) belongs
  in a future loader layer that calls from_serving_plan(dict). This class
  is format-agnostic: it speaks Python dicts only.
- Truth boundary strings are copied verbatim from the compiler plan.
  The adapter invents no measurements and adds no runtime claims.
"""

from __future__ import annotations

from deployment.runtime_execution_plan import RuntimeExecutionPlan, RuntimeExecutionPlanAdapter

_CONTAMINATION_FIELDS: frozenset[str] = frozenset({
    "selected_backend",
    "measured_latency_ms",
    "actual_latency_ms",
    "measured_memory_mb",
    "cuda_graph_captured",
    "runtime_result",
    "backend_decision",
    "execution_statistics",
})


class CompilerRuntimeAdapter:
    """Schema-level transformer from compiler ServingExecutionPlan to runtime IR.

    All methods are static and stateless. No instance state is kept.
    """

    @staticmethod
    def from_serving_plan(d: dict) -> list[RuntimeExecutionPlan]:
        """Convert a full ServingExecutionPlan dict to an ordered list of RuntimeExecutionPlans.

        Reads module-level target_profile_id and propagates it to every
        function plan unless the function plan carries its own target_profile_id.
        Raises ValueError on structural violations or runtime contamination.
        """
        if not isinstance(d, dict):
            raise TypeError("ServingExecutionPlan input must be a dict")

        function_plans = d.get("function_plans")
        if function_plans is None:
            raise ValueError(
                "ServingExecutionPlan dict must contain a 'function_plans' key"
            )
        if not isinstance(function_plans, list):
            raise ValueError("'function_plans' must be a list")

        module_target_profile_id: str = d.get("target_profile_id", "")

        plans: list[RuntimeExecutionPlan] = []
        for fn_dict in function_plans:
            plan = CompilerRuntimeAdapter.from_function_plan(
                fn_dict,
                target_profile_id=module_target_profile_id,
            )
            plans.append(plan)
        return plans

    @staticmethod
    def as_function_map(d: dict) -> dict[str, RuntimeExecutionPlan]:
        """Convert a full ServingExecutionPlan dict to a function_name → plan mapping.

        Raises ValueError if duplicate function names appear in function_plans.
        """
        plans = CompilerRuntimeAdapter.from_serving_plan(d)
        result: dict[str, RuntimeExecutionPlan] = {}
        for plan in plans:
            if plan.function_name in result:
                raise ValueError(
                    f"Duplicate function_name '{plan.function_name}' in ServingExecutionPlan"
                )
            result[plan.function_name] = plan
        return result

    @staticmethod
    def from_function_plan(
        d: dict, *, target_profile_id: str = ""
    ) -> RuntimeExecutionPlan:
        """Convert a single FunctionExecutionPlan dict to a RuntimeExecutionPlan.

        Validates required fields and rejects any runtime-result contamination
        before delegating to RuntimeExecutionPlanAdapter.from_dict().

        Function-level target_profile_id overrides the module-level value when
        present and non-empty in the function dict.
        """
        if not isinstance(d, dict):
            raise TypeError("FunctionExecutionPlan input must be a dict")

        for contamination_key in _CONTAMINATION_FIELDS:
            if contamination_key in d:
                raise ValueError(
                    f"'{contamination_key}' is a runtime-result field and must not "
                    f"appear in a compiler plan; reject compiler plan that includes it"
                )

        function_name = d.get("function_name", "")
        if not function_name:
            raise ValueError("function_name is required in FunctionExecutionPlan")

        bep = d.get("backend_execution_plan")
        if bep is None:
            raise ValueError(
                f"backend_execution_plan is required in FunctionExecutionPlan "
                f"'{function_name}'"
            )
        if not isinstance(bep, dict) or not bep.get("primary_backend"):
            raise ValueError(
                f"primary_backend is required in backend_execution_plan for "
                f"'{function_name}'"
            )

        # Function-level target_profile_id overrides module-level if present and non-empty.
        effective_profile_id: str = d.get("target_profile_id") or target_profile_id

        return RuntimeExecutionPlanAdapter.from_dict(
            d, target_profile_id=effective_profile_id
        )
