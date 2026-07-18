import hashlib
import pytest
from deployment.execution_plan.aarch64_native_object_adapter import AArch64NativeObjectAdapter

def c(tmp_path,uk):
 o=tmp_path/f"u{uk}.o";o.write_bytes(bytes([uk]))
 return {"decision_kind":"aarch64_native_exact_candidate_selection","candidate_id":f"tile8x8x8_uk{uk}",
 "operator":"hir.fused_matmul_bias_relu","kernel_family":"aarch64_generated_fused_matmul_bias_relu","dtype":"f32","shape":{"m":32,"n":32,"k":32},
 "target":{"triple":"aarch64-linux-gnu","cpu":"cortex-a76","features":[],"target_profile_id":"raspberry-pi5-cortex-a76-cpu"},
 "lowering":{"pipeline_id":"aarch64_tiled_scheduled_v1","tile_m":8,"tile_n":8,"tile_k":8,"schedule_unroll_k":uk,"vector_width_bits":128,"loop_order_id":"tiled_mnk_row_major_v1"},
 "microkernel_id":"hir_fused_matmul_bias_relu_tiled_scheduled_v1","entry_point":"_mlir_ciface_matmul_bias_relu_tiled_32x32x32","abi_version":"mlir_ciface_memref_f32_v1",
 "object_ref":o.name,"object_sha256":hashlib.sha256(o.read_bytes()).hexdigest(),"backend_evidence_ref":f"u{uk}.json",
 "selection_mode":"measurement_candidate","selection_trace_ref":"protocol.json","runtime_no_redecision":True}

@pytest.mark.parametrize("uk",[1,2,4])
def test_each_exact_candidate_validates_and_proves_no_redecision(tmp_path,uk):
 x=c(tmp_path,uk);a=AArch64NativeObjectAdapter(x,plan_root=tmp_path)
 a.validate(require_running_target=False)
 proof=a.proof({"executed":{"candidate_id":x["candidate_id"],"object_sha256":x["object_sha256"],
  "entry_point":x["entry_point"],"runtime_redecision_count":0}})
 assert proof["compiler_selected_candidate"]==proof["runtime_executed_candidate"]
