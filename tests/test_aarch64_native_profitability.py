import json
from pathlib import Path
import pytest
from deployment.execution_plan.aarch64_native_object_adapter import AArch64NativeObjectAdapter,AArch64NativeObjectError

def contract(tmp_path,shape=(16,16,32),uk=4):
 m,n,k=shape;obj=tmp_path/"x.o";obj.write_bytes(b"x")
 from deployment.execution_plan.aarch64_native_object_adapter import sha256
 return {"candidate_id":f"m{m}_n{n}_k{k}_tile8x8x8_uk{uk}","operator":"hir.fused_matmul_bias_relu",
 "kernel_family":"aarch64_generated_fused_matmul_bias_relu","dtype":"f32","shape":{"m":m,"n":n,"k":k},
 "entry_point":f"_mlir_ciface_matmul_bias_relu_tiled_{m}x{n}x{k}","abi_version":"mlir_ciface_memref_f32_v1",
 "microkernel_id":"hir_fused_matmul_bias_relu_tiled_scheduled_v1",
 "target":{"triple":"aarch64-linux-gnu","cpu":"cortex-a76","features":[],"target_profile_id":"raspberry-pi5-cortex-a76-cpu"},
 "lowering":{"pipeline_id":"aarch64_tiled_scheduled_v1","tile_m":8,"tile_n":8,"tile_k":8,"schedule_unroll_k":uk,
 "vector_width_bits":128,"loop_order_id":"tiled_mnk_row_major_v1"},"runtime_no_redecision":True,
 "object_ref":"x.o","object_sha256":sha256(obj)}
def test_cross_shape_contract_and_no_redecision(tmp_path):
 c=contract(tmp_path);a=AArch64NativeObjectAdapter(c,plan_root=tmp_path);a.validate(require_running_target=False)
 proof=a.proof({"executed":{"candidate_id":c["candidate_id"],"object_sha256":c["object_sha256"],
 "entry_point":c["entry_point"],"runtime_redecision_count":0}})
 assert proof["runtime_redecision_count"]==0
def test_wrong_shape_entry_and_sha_fail(tmp_path):
 c=contract(tmp_path);c["entry_point"]="_mlir_ciface_wrong"
 with pytest.raises(AArch64NativeObjectError,match="entry_point_mismatch"):
  AArch64NativeObjectAdapter(c,plan_root=tmp_path).validate(require_running_target=False)
