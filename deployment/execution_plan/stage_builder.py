"""Build ExecutionStage IR from compiler ExecutionPlan v2 decisions."""

from __future__ import annotations

from deployment.execution_plan.schema import (
    EXECUTION_STAGE_TRUTH_BOUNDARY,
    ExecutionPlan,
    ExecutionStage,
    ExecutionStageKind,
    FunctionPlan,
    OpDecision,
)


def build_execution_stages(plan: ExecutionPlan) -> list[ExecutionStage]:
    stages: list[ExecutionStage] = []
    for function_plan in plan.function_plans:
        stages.append(_stage_from_function(function_plan))
        for op in function_plan.per_op_decisions:
            stages.append(_stage_from_op(function_plan, op))
    return stages


def _stage_from_function(function_plan: FunctionPlan) -> ExecutionStage:
    kind = _kind_from_phase(function_plan.serving_phase)
    return ExecutionStage(
        stage_id=function_plan.function_name,
        kind=kind,
        function_name=function_plan.function_name,
        serving_phase=function_plan.serving_phase,
        op_name=None,
        op_type=None,
        source_compiler_decision=function_plan.raw,
        truth_boundary=EXECUTION_STAGE_TRUTH_BOUNDARY,
    )


def _stage_from_op(function_plan: FunctionPlan, op: OpDecision) -> ExecutionStage:
    return ExecutionStage(
        stage_id=op.op_name or f"{function_plan.function_name}:{op.op_type}",
        kind=_kind_from_op(op.op_type),
        function_name=function_plan.function_name,
        serving_phase=function_plan.serving_phase,
        op_name=op.op_name,
        op_type=op.op_type,
        source_compiler_decision=op.raw,
        truth_boundary=EXECUTION_STAGE_TRUTH_BOUNDARY,
    )


def _kind_from_phase(phase: str) -> ExecutionStageKind:
    normalized = (phase or "other").lower()
    if normalized == "prefill":
        return ExecutionStageKind.PREFILL
    if normalized == "decode":
        return ExecutionStageKind.DECODE
    return ExecutionStageKind.OTHER


def _kind_from_op(op_type: str) -> ExecutionStageKind:
    normalized = (op_type or "").lower()
    if normalized == "rmsnorm":
        return ExecutionStageKind.RMSNORM
    if normalized == "matmul":
        return ExecutionStageKind.MATMUL
    # Phase P1B: hir.fused_matmul_bias_relu (compiler dialect-qualified name)
    # is a real, existing HIR op (mlir_passes/include/HIR/IR/HIROps.td:
    # HIR_FusedMatMulBiasReluOp) -- bucket it into the existing MATMUL stage
    # kind rather than adding new stage-kind vocabulary for it.
    if normalized.endswith("fused_matmul_bias_relu"):
        return ExecutionStageKind.MATMUL
    if normalized == "attention" or normalized.endswith("cpu_attention"):
        return ExecutionStageKind.ATTENTION
    return ExecutionStageKind.MICROBENCHMARK
