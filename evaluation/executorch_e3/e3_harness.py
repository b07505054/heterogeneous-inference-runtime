#!/usr/bin/env python3
"""E3 live-compiler ExecuTorch/XNNPACK comparison harness.

The harness owns no XNNPACK selection policy. It invokes the compiler contract
producer, validates the emitted contract, and constructs the common-runner
invocation from the selected candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "e3_compiler_xnnpack_comparison_contract"
EXPECTED_RUNNER_CONTRACT = "executorch_xnnpack_runner_contract"
EXPECTED_PROVIDER = "executorch_xnnpack_candidate_provider"
EXPECTED_EXECUTORCH_COMMIT = "e2f18eb23c45bd22ca332b0b8b49a81de304b472"
EXPECTED_XNNPACK_COMMIT = "1adaa7c709d4839d29e1f219cb962b01c9e6a905"


class ContractError(ValueError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_contract_hash(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_and_validate_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("schema_mismatch")
    if contract.get("contract_sha256") != canonical_contract_hash(contract):
        raise ContractError("contract_hash_mismatch")
    if contract.get("runner_contract") != EXPECTED_RUNNER_CONTRACT:
        raise ContractError("runner_contract_mismatch")
    if contract.get("provider_id") != EXPECTED_PROVIDER:
        raise ContractError("provider_mismatch")
    selected = contract.get("selected_candidate") or {}
    if selected.get("candidate_id") != contract.get("selected_candidate_id"):
        raise ContractError("selected_candidate_id_mismatch")
    if selected.get("runtime_contract_kind") != EXPECTED_RUNNER_CONTRACT:
        raise ContractError("selected_runtime_contract_mismatch")
    if selected.get("implementation_kind") != "external_library_delegate":
        raise ContractError("implementation_kind_mismatch")
    if selected.get("library") != "xnnpack" or contract.get("library") != "xnnpack":
        raise ContractError("library_mismatch")
    if selected.get("backend") != "cpu" or contract.get("backend") != "cpu":
        raise ContractError("backend_mismatch")
    if selected.get("dtype") != "fp32":
        raise ContractError("dtype_mismatch")
    if selected.get("semantic_target_ref") != "fused_matmul_bias_relu":
        raise ContractError("semantic_scope_mismatch")
    threads = (((selected.get("thread_schedule") or {}).get("thread_count")))
    if threads != (contract.get("requested_thread_mode") or {}).get("threads"):
        raise ContractError("thread_mode_mismatch")
    if threads not in (1, 4):
        raise ContractError("unsupported_thread_mode")
    if (selected.get("feasibility") or {}).get("status") != "feasible":
        raise ContractError("selected_candidate_not_feasible")
    if (contract.get("executorch") or {}).get("commit") != EXPECTED_EXECUTORCH_COMMIT:
        raise ContractError("executorch_commit_mismatch")
    if (contract.get("xnnpack") or {}).get("commit") != EXPECTED_XNNPACK_COMMIT:
        raise ContractError("xnnpack_commit_mismatch")
    pte = contract.get("pte") or {}
    runner = contract.get("runner") or {}
    pte_path = Path(pte.get("path", ""))
    runner_path = Path(runner.get("path", ""))
    if not pte_path.exists():
        raise ContractError("pte_missing")
    if not runner_path.exists():
        raise ContractError("runner_missing")
    if sha_file(pte_path) != pte.get("sha256"):
        raise ContractError("pte_hash_mismatch")
    if sha_file(runner_path) != runner.get("sha256"):
        raise ContractError("runner_hash_mismatch")
    artifact = selected.get("artifact") or {}
    if artifact.get("pte_sha256") != pte.get("sha256"):
        raise ContractError("selected_pte_hash_mismatch")
    if artifact.get("runner_sha256") != runner.get("sha256"):
        raise ContractError("selected_runner_hash_mismatch")
    provenance = selected.get("provenance") or {}
    if provenance.get("executorch_commit") != EXPECTED_EXECUTORCH_COMMIT:
        raise ContractError("selected_executorch_commit_mismatch")
    if provenance.get("xnnpack_commit") != EXPECTED_XNNPACK_COMMIT:
        raise ContractError("selected_xnnpack_commit_mismatch")
    return contract


def invoke_live_compiler(args: argparse.Namespace) -> Path:
    out = Path(args.contract_out)
    cmd = [
        args.compiler_tool,
        "--shape", args.shape,
        "--pte", args.pte,
        "--runner", args.runner,
        "--out", str(out),
        "--selected-threads", str(args.selected_threads),
        "--executorch-commit", EXPECTED_EXECUTORCH_COMMIT,
        "--xnnpack-commit", EXPECTED_XNNPACK_COMMIT,
        "--xnnpack-delegated",
        "--input-binding-compatible",
    ]
    compiler_commit = getattr(args, "compiler_commit", None)
    if compiler_commit:
        cmd += ["--compiler-commit", compiler_commit]
    subprocess.run(cmd, check=True)
    return out


def build_runner_command(contract: dict[str, Any], input_a: Path, input_b: Path, input_bias: Path,
                         output: Path, result_json: Path, warmups: int, repeats: int) -> list[str]:
    threads = (contract["requested_thread_mode"] or {})["threads"]
    return [
        contract["runner"]["path"],
        "--model_path", contract["pte"]["path"],
        "--input_a", str(input_a),
        "--input_b", str(input_b),
        "--input_bias", str(input_bias),
        "--requested_threads", str(threads),
        "--warmups", str(warmups),
        "--repeats", str(repeats),
        "--output", str(output),
        "--result_json", str(result_json),
    ]


def validate_runner_report(contract: dict[str, Any], report: dict[str, Any]) -> None:
    checks = {
        "runner_sha256": contract["runner"]["sha256"],
        "pte_sha256": contract["pte"]["sha256"],
        "executorch_commit": EXPECTED_EXECUTORCH_COMMIT,
        "xnnpack_commit": EXPECTED_XNNPACK_COMMIT,
        "requested_threads": contract["requested_thread_mode"]["threads"],
    }
    for key, expected in checks.items():
        if report.get(key) != expected:
            raise ContractError(f"runner_self_report_{key}_mismatch")
    if report.get("backend") != "xnnpack":
        raise ContractError("runner_backend_mismatch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-tool", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--pte", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--selected-threads", required=True, type=int, choices=(1, 4))
    ap.add_argument("--contract-out", required=True)
    ap.add_argument("--compiler-commit")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--input-a")
    ap.add_argument("--input-b")
    ap.add_argument("--input-bias")
    ap.add_argument("--output")
    ap.add_argument("--result-json")
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    contract_path = invoke_live_compiler(args)
    contract = load_and_validate_contract(contract_path)
    if args.validate_only:
        print(json.dumps({"validated_contract_sha256": contract["contract_sha256"]}, sort_keys=True))
        return
    required = [args.input_a, args.input_b, args.input_bias, args.output, args.result_json]
    if any(v is None for v in required):
        raise SystemExit("runner inputs/output/result-json are required unless --validate-only is set")
    cmd = build_runner_command(
        contract, Path(args.input_a), Path(args.input_b), Path(args.input_bias),
        Path(args.output), Path(args.result_json), args.warmups, args.repeats)
    subprocess.run(cmd, check=True)
    validate_runner_report(contract, json.loads(Path(args.result_json).read_text()))


if __name__ == "__main__":
    main()
