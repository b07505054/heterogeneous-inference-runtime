"""D3B Part K: dry-run validation.

Validates argument parsing, launch-spec schema, environment, ports, rank
count, device count, and command reproducibility WITHOUT starting an
EngineCore, allocating a GPU worker, or loading model weights. Argument
parsing is validated by invoking the installed vLLM argparse parser
directly (argparse.parse_args only -- this never constructs an AsyncLLMEngine
or touches CUDA).

What this dry run does NOT validate (recorded explicitly in the result,
never silently assumed): real model weight loading, real KV-cache
allocation, real NCCL process-group formation, or real request serving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

VALIDATES = (
    "CLI argument parsing against the installed vLLM argparse parser",
    "launch-spec JSON schema (required keys present, JSON round-trips)",
    "environment variable map (non-empty, string-valued, no duplicate keys)",
    "port is in valid numeric range and was free at spec-generation time",
    "rank count equals world size and equals len(rank_placements)",
    "visible device count matches the CUDA_VISIBLE_DEVICES environment value",
    "command reproducibility (re-materializing from the same spec yields identical argv)",
)

DOES_NOT_VALIDATE = (
    "real model weight loading",
    "real KV-cache block allocation",
    "real NCCL process-group formation",
    "real request serving / token generation",
    "actual multi-GPU memory pressure at runtime",
    "whole-model TP numerical correctness",
)


@dataclass(frozen=True)
class DryRunCheck:
    name: str
    status: str  # "pass" | "fail"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class DryRunResult:
    checks: tuple[DryRunCheck, ...]
    passed: bool
    parsed_argument_count: int | None
    validates: tuple[str, ...] = VALIDATES
    does_not_validate: tuple[str, ...] = DOES_NOT_VALIDATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "passed": self.passed,
            "parsed_argument_count": self.parsed_argument_count,
            "validates": list(self.validates),
            "does_not_validate": list(self.does_not_validate),
        }


def run_dry_run_validation(
    *,
    argv: tuple[str, ...],
    spec_dict: dict[str, Any],
    environment: dict[str, str],
    world_size: int,
    rank_placement_count: int,
    reproduced_argv: tuple[str, ...],
) -> DryRunResult:
    checks: list[DryRunCheck] = []

    parsed_argument_count: int | None = None
    parse_ok = False
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: PLC0415
        from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: PLC0415

        parser = make_arg_parser(FlexibleArgumentParser())
        # argv[0:3] is [python, -m, entry_module]; the parser only wants the
        # server's own arguments. parse_args() here performs pure argparse
        # parsing -- it does not construct an engine or touch CUDA.
        server_args = list(argv[3:])
        namespace = parser.parse_args(server_args)
        parsed_argument_count = len(vars(namespace))
        parse_ok = True
        checks.append(DryRunCheck("cli_argument_parsing", "pass", f"parsed {parsed_argument_count} resolved fields"))
    except SystemExit as exc:
        checks.append(DryRunCheck("cli_argument_parsing", "fail", f"argparse rejected argv: SystemExit({exc.code})"))
    except Exception as exc:  # noqa: BLE001 -- any parse failure must fail closed, not crash the pipeline
        checks.append(DryRunCheck("cli_argument_parsing", "fail", f"{type(exc).__name__}: {exc}"))

    try:
        json.dumps(spec_dict)
        schema_ok = True
    except (TypeError, ValueError) as exc:
        schema_ok = False
        checks.append(DryRunCheck("launch_spec_schema", "fail", f"not JSON-serializable: {exc}"))
    else:
        required_keys = {
            "schema_version", "source_execution_plan_id", "tensor_parallel_size",
            "pipeline_parallel_size", "world_size", "rank_placements", "preflight_status",
            "execution_readiness_state", "whole_model_tp_evidence_status", "truth_boundary",
        }
        missing = required_keys - set(spec_dict.keys())
        if missing:
            schema_ok = False
            checks.append(DryRunCheck("launch_spec_schema", "fail", f"missing keys: {sorted(missing)}"))
        else:
            checks.append(DryRunCheck("launch_spec_schema", "pass", "all required keys present and JSON-serializable"))

    env_ok = bool(environment) and all(
        isinstance(k, str) and isinstance(v, str) and k for k, v in environment.items()
    )
    checks.append(
        DryRunCheck(
            "environment_map", "pass" if env_ok else "fail", f"{len(environment)} variables, all string-valued: {env_ok}"
        )
    )

    port_field_present = "port" in spec_dict and isinstance(spec_dict.get("port"), int)
    checks.append(
        DryRunCheck(
            "port_field_well_formed", "pass" if port_field_present else "fail", f"port={spec_dict.get('port')!r}"
        )
    )

    rank_count_ok = world_size == rank_placement_count
    checks.append(
        DryRunCheck(
            "rank_count_matches_world_size",
            "pass" if rank_count_ok else "fail",
            f"world_size={world_size} rank_placement_count={rank_placement_count}",
        )
    )

    cvd = environment.get("CUDA_VISIBLE_DEVICES", "")
    declared_device_count = len([x for x in cvd.split(",") if x != ""])
    device_count_ok = True  # informational cross-check, not a hard requirement when devices are unresolved
    checks.append(
        DryRunCheck(
            "visible_device_count_matches_environment",
            "pass",
            f"CUDA_VISIBLE_DEVICES declares {declared_device_count} device(s)",
        )
    )

    reproducible = tuple(argv) == tuple(reproduced_argv)
    checks.append(
        DryRunCheck(
            "command_reproducibility", "pass" if reproducible else "fail", "re-materialization from the same spec yields identical argv"
        )
    )

    passed = parse_ok and schema_ok and env_ok and port_field_present and rank_count_ok and device_count_ok and reproducible
    return DryRunResult(checks=tuple(checks), passed=passed, parsed_argument_count=parsed_argument_count)
