"""D3B Part G: deterministic CLI representation.

Arguments are represented as an array internally (argv) to avoid shell
interpolation bugs; a shell-escaped string form is derived from that array
via shlex.join for display/logging only. The materialized command is never
executed by this module.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from deployment.vllm_adapter.distributed_argument_registry import (
    ArgumentCompatibilityRecord,
    check_argument,
)

PYTHON_EXECUTABLE = ".venv/bin/python"
ENTRY_MODULE = "vllm.entrypoints.openai.api_server"


@dataclass(frozen=True)
class CLIRepresentation:
    argv: tuple[str, ...]
    shell_command: str
    environment: dict[str, str]
    working_directory: str
    expected_process_count: int
    expected_gpu_assignment: dict[str, Any]
    argument_compatibility: tuple[ArgumentCompatibilityRecord, ...]
    all_arguments_supported: bool
    unsupported_arguments: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "shell_command": self.shell_command,
            "environment": self.environment,
            "working_directory": self.working_directory,
            "expected_process_count": self.expected_process_count,
            "expected_gpu_assignment": self.expected_gpu_assignment,
            "argument_compatibility": [r.to_dict() for r in self.argument_compatibility],
            "all_arguments_supported": self.all_arguments_supported,
            "unsupported_arguments": list(self.unsupported_arguments),
        }


def build_cli(
    fields: dict[str, Any],
    *,
    registry: dict[str, Any],
    environment: dict[str, str],
    working_directory: str,
    rank_placements: list[dict[str, Any]],
) -> CLIRepresentation:
    """Build a deterministic argv array for the installed vLLM version.

    ``fields`` carries resolved values keyed by argparse dest name (model,
    tokenizer, dtype, tensor_parallel_size, ...). Every field is checked
    against the installed registry (Part J) before being emitted; an
    unsupported field is recorded but NOT silently included in argv.
    """
    records: list[ArgumentCompatibilityRecord] = []
    argv: list[str] = [PYTHON_EXECUTABLE, "-m", ENTRY_MODULE]

    def emit_value(dest: str, value: Any, *, value_source: str) -> None:
        record = check_argument(dest, registry=registry, value_source=value_source)
        records.append(record)
        if record.installed_version_support_status != "supported":
            return
        if value is None:
            return
        argv.extend([record.resolved_spelling, str(value)])

    def emit_bool_flag(dest: str, enabled: bool, *, value_source: str) -> None:
        record = check_argument(dest, registry=registry, value_source=value_source)
        records.append(record)
        if record.installed_version_support_status != "supported":
            return
        arguments = registry.get("arguments", {})
        spec = arguments.get(dest, {})
        option_strings: list[str] = spec.get("option_strings", [])
        if enabled:
            argv.append(record.resolved_spelling)
        else:
            negative = next((o for o in option_strings if o.startswith("--no-")), None)
            if negative:
                argv.append(negative)

    emit_value("model", fields["model"], value_source=fields["_sources"]["model"])
    emit_value("tokenizer", fields["tokenizer"], value_source=fields["_sources"]["tokenizer"])
    emit_bool_flag(
        "trust_remote_code", fields["trust_remote_code"], value_source=fields["_sources"]["trust_remote_code"]
    )
    emit_value("dtype", fields["dtype"], value_source=fields["_sources"]["dtype"])
    emit_value("seed", fields["seed"], value_source=fields["_sources"]["seed"])
    if fields.get("revision") is not None:
        emit_value("revision", fields["revision"], value_source=fields["_sources"]["revision"])
    emit_value(
        "served_model_name", fields["served_model_name"], value_source=fields["_sources"]["served_model_name"]
    )
    emit_value("host", fields["host"], value_source=fields["_sources"]["host"])
    emit_value("port", fields["port"], value_source=fields["_sources"]["port"])
    emit_value("master_addr", fields["master_address"], value_source=fields["_sources"]["master_address"])
    emit_value("master_port", fields["master_port"], value_source=fields["_sources"]["master_port"])

    emit_value(
        "tensor_parallel_size", fields["tensor_parallel_size"], value_source=fields["_sources"]["tensor_parallel_size"]
    )
    emit_value(
        "pipeline_parallel_size",
        fields["pipeline_parallel_size"],
        value_source=fields["_sources"]["pipeline_parallel_size"],
    )
    emit_value(
        "data_parallel_size", fields["data_parallel_size"], value_source=fields["_sources"]["data_parallel_size"]
    )
    emit_value(
        "distributed_executor_backend",
        fields["distributed_executor_backend"],
        value_source=fields["_sources"]["distributed_executor_backend"],
    )

    emit_value("max_model_len", fields["max_model_len"], value_source=fields["_sources"]["max_model_len"])
    emit_value("max_num_seqs", fields["max_num_seqs"], value_source=fields["_sources"]["max_num_seqs"])
    emit_value(
        "max_num_batched_tokens",
        fields["max_num_batched_tokens"],
        value_source=fields["_sources"]["max_num_batched_tokens"],
    )
    emit_value(
        "gpu_memory_utilization",
        fields["gpu_memory_utilization"],
        value_source=fields["_sources"]["gpu_memory_utilization"],
    )
    emit_bool_flag(
        "enable_prefix_caching", fields["enable_prefix_caching"], value_source=fields["_sources"]["enable_prefix_caching"]
    )
    emit_bool_flag(
        "enable_chunked_prefill",
        fields["enable_chunked_prefill"],
        value_source=fields["_sources"]["enable_chunked_prefill"],
    )

    ok = all(r.installed_version_support_status == "supported" for r in records)
    unsupported = tuple(r.dest for r in records if r.installed_version_support_status != "supported")

    expected_process_count = fields["world_size"]
    expected_gpu_assignment = {
        str(p["rank_id"]): p["physical_device_index"] for p in rank_placements
    }

    return CLIRepresentation(
        argv=tuple(argv),
        shell_command=shlex.join(argv),
        environment=dict(environment),
        working_directory=working_directory,
        expected_process_count=expected_process_count,
        expected_gpu_assignment=expected_gpu_assignment,
        argument_compatibility=tuple(records),
        all_arguments_supported=ok,
        unsupported_arguments=unsupported,
    )
