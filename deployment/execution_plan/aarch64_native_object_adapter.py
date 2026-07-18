"""Exact fixed-shape AArch64 generated-object Runtime adapter (Slice 18B)."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess


class AArch64NativeObjectError(RuntimeError):
    pass


EXPECTED = {
    "operator": "hir.fused_matmul_bias_relu",
    "kernel_family": "aarch64_generated_fused_matmul_bias_relu",
    "dtype": "f32",
    "abi_version": "mlir_ciface_memref_f32_v1",
    "microkernel_id": "hir_fused_matmul_bias_relu_tiled_scheduled_v1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class AArch64NativeObjectAdapter:
    def __init__(self, contract: dict, *, plan_root: str | Path):
        self.contract = dict(contract)
        self.plan_root = Path(plan_root).resolve()

    def object_path(self) -> Path:
        ref = Path(str(self.contract.get("object_ref", "")))
        return ref if ref.is_absolute() else (self.plan_root / ref).resolve()

    def validate(self, *, require_running_target: bool = True) -> None:
        c = self.contract
        for key, value in EXPECTED.items():
            if c.get(key) != value:
                raise AArch64NativeObjectError(f"{key}_mismatch")
        shape=c.get("shape",{})
        try:
            m,n,k=(int(shape[x]) for x in ("m","n","k"))
            uk=int(c["lowering"]["schedule_unroll_k"])
        except (KeyError,TypeError,ValueError):
            raise AArch64NativeObjectError("shape_or_unroll_malformed")
        legacy=f"tile8x8x8_uk{uk}"
        exact=f"m{m}_n{n}_k{k}_tile8x8x8_uk{uk}"
        if c.get("candidate_id") == legacy and (m,n,k) != (32,32,32):
            raise AArch64NativeObjectError("shape_mismatch")
        if c.get("candidate_id") not in {legacy,exact} or uk not in {1,2,4}:
            raise AArch64NativeObjectError("unknown_candidate")
        expected_entry=f"_mlir_ciface_matmul_bias_relu_tiled_{m}x{n}x{k}"
        if c.get("entry_point") != expected_entry:
            raise AArch64NativeObjectError("entry_point_mismatch")
        target, lowering = c.get("target", {}), c.get("lowering", {})
        if target != {"triple": "aarch64-linux-gnu", "cpu": "cortex-a76",
                      "features": [], "target_profile_id": "raspberry-pi5-cortex-a76-cpu"}:
            raise AArch64NativeObjectError("target_mismatch")
        if lowering != {"pipeline_id": "aarch64_tiled_scheduled_v1",
                        "tile_m": 8, "tile_n": 8, "tile_k": 8,
                        "schedule_unroll_k": uk,
                        "vector_width_bits": 128,
                        "loop_order_id": "tiled_mnk_row_major_v1"}:
            raise AArch64NativeObjectError("lowering_identity_mismatch")
        if c.get("runtime_no_redecision") is not True:
            raise AArch64NativeObjectError("runtime_redecision_forbidden")
        obj = self.object_path()
        if not obj.is_file():
            raise AArch64NativeObjectError("object_missing")
        if sha256(obj) != c.get("object_sha256"):
            raise AArch64NativeObjectError("object_sha256_mismatch")
        if require_running_target and platform.machine().lower() not in {"aarch64", "arm64"}:
            raise AArch64NativeObjectError("running_architecture_mismatch")

    def verify_symbol(self) -> None:
        p = subprocess.run(["nm", "-g", str(self.object_path())],
                           text=True, capture_output=True)
        if p.returncode or self.contract["entry_point"] not in p.stdout:
            raise AArch64NativeObjectError("entry_point_symbol_missing")

    def proof(self, result: dict) -> dict:
        self.validate(require_running_target=False)
        selected = self.contract
        executed = result.get("executed", {})
        proof = {
            "compiler_selected_candidate": selected["candidate_id"],
            "runtime_executed_candidate": executed.get("candidate_id"),
            "selected_object_sha256": selected["object_sha256"],
            "executed_object_sha256": executed.get("object_sha256"),
            "selected_entry_point": selected["entry_point"],
            "executed_entry_point": executed.get("entry_point"),
            "runtime_redecision_count": int(executed.get("runtime_redecision_count", -1)),
        }
        if not (proof["compiler_selected_candidate"] == proof["runtime_executed_candidate"]
                and proof["selected_object_sha256"] == proof["executed_object_sha256"]
                and proof["selected_entry_point"] == proof["executed_entry_point"]
                and proof["runtime_redecision_count"] == 0):
            raise AArch64NativeObjectError("selected_executed_identity_mismatch")
        return proof

    @staticmethod
    def load_contract(plan_path: str | Path) -> tuple[dict, Path]:
        path = Path(plan_path).resolve()
        plan = json.loads(path.read_text())
        ops = plan["function_plans"][0]["per_op_decisions"]
        contract = next(op["native_execution"] for op in ops if "native_execution" in op)
        return contract, path.parent
