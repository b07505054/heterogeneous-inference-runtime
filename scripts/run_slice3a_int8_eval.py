#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from deployment.execution_plan.int8_quantization import *
from deployment.execution_plan.portable_cpu_kernel_adapter import Tensor, dispatch_fused_matmul_bias_relu
from deployment.execution_plan.slice3c_target_selection import (
    FP32_CANDIDATE_ID,
    INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,
    INT8_PACKED_GENERIC_CANDIDATE_ID,
    INT8_ROW_MAJOR_CANDIDATE_ID,
    create_build_manifest,
    create_measurement_artifact,
    enumerate_complete_candidates,
    load_codegen_capabilities,
    materialize_build_manifest,
    select_candidate,
    write_manifest,
)

FP32_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"


def memory_placement(m,n,k):
    return {"status":"selected","compute_unit":"cpu","selected_memory_space":"cpu_visible_host_memory","input_tile_bytes":m*k*4,"weight_tile_bytes":k*n*4,"output_tile_bytes":m*n*4,"scratch_bytes":0,"padding_bytes":0,"single_buffer_bytes":m*k*4+k*n*4+m*n*4,"additional_double_buffer_bytes":0,"total_required_local_memory_bytes":m*k*4+k*n*4+m*n*4,"buffer_placements":[{"buffer_id":"input_tile","role":"input","memory_space":"cpu_visible_host_memory","byte_count":m*k*4,"alignment":64},{"buffer_id":"weight_tile","role":"weight","memory_space":"cpu_visible_host_memory","byte_count":k*n*4,"alignment":64},{"buffer_id":"output_tile","role":"output","memory_space":"cpu_visible_host_memory","byte_count":m*n*4,"alignment":64},{"buffer_id":"scratch","role":"scratch","memory_space":"cpu_visible_host_memory","byte_count":0,"alignment":64}],"transfer_operations":[],"compute_dependency_ids":[]}

def tensors(m,n,k,seed):
    rng=random.Random(seed)
    return Tensor((m,k),[rng.uniform(-1,1) for _ in range(m*k)]), Tensor((k,n),[rng.uniform(-1,1) for _ in range(k*n)]), Tensor((n,),[rng.uniform(-0.2,0.2) for _ in range(n)])

def fp32_decision(m,n,k):
    return {"op_name":"op_2","op_type":"hir.fused_matmul_bias_relu","kernel_selection":{"contract_version":"kernel_selection_contract_v1","status":"selected","selected_kernel":FP32_KERNEL_ID,"source":"slice3a_eval"},"quantization":{"selected_candidate_id":f"fused_matmul_bias_relu:quant=fp32_baseline:kernel={FP32_KERNEL_ID}","scheme":"fp32_baseline","activation_dtype":"fp32","weight_dtype":"fp32","accumulation_dtype":"fp32","output_dtype":"fp32","required_kernel_capability":"quant_kernel.none","requires_calibration":False,"calibration_available":False},"thread_schedule":{"status":"selected","thread_count":1,"partition_axis":"none","partition_strategy":"serial"},"memory_placement":memory_placement(m,n,k)}

def int8_decision(artifact, artifact_path, m,n,k):
    return {"op_name":"op_2","op_type":"hir.fused_matmul_bias_relu","kernel_selection":{"contract_version":"kernel_selection_contract_v1","status":"selected","selected_kernel":INT8_KERNEL_ID,"source":"slice3a_eval"},"quantization":{"selected_candidate_id":f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:kernel={INT8_KERNEL_ID}","scheme":SCHEME,"activation_dtype":"int8","weight_dtype":"int8","accumulation_dtype":"int32","output_dtype":"fp32","activation_granularity":"per_tensor","weight_granularity":"per_tensor","granularity":"per_tensor","activation_scale":artifact["activation_scale"],"weight_scale":artifact["weight_scale"],"activation_zero_point":0,"weight_zero_point":0,"required_kernel_capability":KERNEL_CAPABILITY,"requires_calibration":True,"calibration_available":True,"calibration_artifact_ref":str(artifact_path),"calibration_artifact_id":artifact["artifact_id"],"calibration_artifact_sha256":artifact["artifact_sha256"],"workload_id":artifact["workload_id"],"policy_id":"slice3a_evidence_policy_v1","selection_reason":"pending_measurement"},"thread_schedule":{"status":"selected","thread_count":1,"partition_axis":"none","partition_strategy":"serial"},"memory_placement":memory_placement(m,n,k)}

def packed_int8_decision(artifact, artifact_path, packed_artifact, packed_artifact_path, m,n,k, *, selected_candidate_id=None, build=None, measurement_path=None, measurement=None):
    decision = int8_decision(artifact, artifact_path, m,n,k)
    decision["kernel_selection"]["selected_kernel"] = PACKED_INT8_KERNEL_ID
    decision["kernel_selection"]["source"] = "slice3b_eval"
    decision["quantization"]["selected_candidate_id"] = f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:kernel={PACKED_INT8_KERNEL_ID}:packed={packed_artifact['artifact_id']}"
    decision["quantization"]["selected_complete_candidate_id"] = selected_candidate_id or INT8_PACKED_A76_DOTPROD_CANDIDATE_ID
    decision["quantization"]["required_kernel_capability"] = PACKED_INT8_KERNEL_CAPABILITY
    decision["quantization"]["kernel_requires_packed_weight"] = True
    decision["quantization"]["packed_weight_artifact_ref"] = str(packed_artifact_path)
    decision["quantization"]["packed_weight_artifact_id"] = packed_artifact["artifact_id"]
    decision["quantization"]["packed_weight_sha256"] = packed_artifact["artifact_sha256"]
    decision["quantization"]["packed_layout"] = PACKED_B_TRANSPOSE_LAYOUT
    decision["quantization"]["packing_scheme"] = PACKED_B_TRANSPOSE_SCHEME
    decision["quantization"]["policy_id"] = "slice3b_packed_weight_policy_v1"
    decision["quantization"]["selection_reason"] = "compiler_owned_packed_weight_artifact_selected"
    if build:
        decision["quantization"]["codegen_target_id"] = build["codegen_target_id"]
        decision["quantization"]["target_architecture"] = build["target_architecture"]
        decision["quantization"]["target_microarchitecture"] = build["target_microarchitecture"]
        decision["quantization"]["required_isa_features"] = build["required_isa_features"]
        decision["quantization"]["compiler_flags"] = build["compiler_flags"]
        decision["quantization"]["build_manifest_ref"] = build.get("manifest_ref", "")
        decision["quantization"]["binary_sha256"] = build.get("binary_sha256", "")
    if measurement_path and measurement:
        decision["quantization"]["measurement_artifact_ref"] = str(measurement_path)
        decision["quantization"]["measurement_artifact_id"] = measurement["measurement_artifact_id"]
        decision["quantization"]["measurement_artifact_sha256"] = measurement["measurement_sha256"]
    return decision

def ensure_built(outdir: Path, target_profile: dict):
    candidates={c.candidate_id:c for c in enumerate_complete_candidates()}
    build_dir=outdir/"build"; build_dir.mkdir(parents=True, exist_ok=True)
    built={}
    for cid in [FP32_CANDIDATE_ID, INT8_ROW_MAJOR_CANDIDATE_ID, INT8_PACKED_GENERIC_CANDIDATE_ID, INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]:
        manifest=create_build_manifest(candidates[cid], source_root=REPO_ROOT, output_dir=build_dir)
        if cid == INT8_PACKED_GENERIC_CANDIDATE_ID:
            manifest["output_binary"]=str(build_dir/"portable_fused_matmul_bias_relu_int8_packed_b_generic")
            manifest["build_command"][manifest["build_command"].index("-o")+1]=manifest["output_binary"]
        if cid == INT8_PACKED_A76_DOTPROD_CANDIDATE_ID:
            manifest["output_binary"]=str(build_dir/"portable_fused_matmul_bias_relu_int8_packed_b_a76")
            manifest["build_command"][manifest["build_command"].index("-o")+1]=manifest["output_binary"]
        built_manifest=materialize_build_manifest(manifest)
        manifest_path=outdir/f"build_manifest_{cid.replace(':','_')}.json"
        built_manifest["manifest_ref"]=str(manifest_path)
        write_manifest(manifest_path,built_manifest)
        built[cid]=built_manifest
    return built

def run_shape(outdir: Path, m:int,n:int,k:int, seed:int, warmup:int, repeats:int, build_identity: dict, target_profile: dict):
    a,b,bias=tensors(m,n,k,seed)
    workload=f"slice3a_fused_matmul_bias_relu_{m}x{n}x{k}_seed{seed}"
    dataset={"dataset_id":"synthetic_slice3a_seeded_uniform","seed":seed,"distribution":"uniform[-1,1]","sample_count":1,"shape":{"M":m,"N":n,"K":k}}
    t0=time.perf_counter(); artifact=create_calibration_artifact(workload_id=workload,operator_kind="fused_matmul_bias_relu",m=m,n=n,k=k,activation_values=a.data,weight_values=b.data,calibration_dataset=dataset); calibration_ms=(time.perf_counter()-t0)*1000
    artifact_path=outdir/f"calibration_{m}x{n}x{k}.json"; write_json_deterministic(artifact_path,artifact)
    t1=time.perf_counter(); packed_manifest, packed_bytes = create_packed_weight_artifact(workload_id=workload,operator_kind="fused_matmul_bias_relu",m=m,n=n,k=k,source_weight_values=b.data,weight_scale=artifact["weight_scale"]); packing_ms=(time.perf_counter()-t1)*1000
    packed_path=outdir/f"packed_weight_{m}x{n}x{k}.json"; packed_bin=outdir/f"packed_weight_{m}x{n}x{k}.bin"; packed_artifact=write_packed_weight_artifact(packed_path,packed_bin,packed_manifest,packed_bytes)
    fp_dec=fp32_decision(m,n,k); i8_dec=int8_decision(artifact,artifact_path,m,n,k)
    packed_generic_dec=packed_int8_decision(artifact,artifact_path,packed_artifact,packed_path,m,n,k,selected_candidate_id=INT8_PACKED_GENERIC_CANDIDATE_ID,build=build_identity[INT8_PACKED_GENERIC_CANDIDATE_ID])
    packed_a76_dec=packed_int8_decision(artifact,artifact_path,packed_artifact,packed_path,m,n,k,selected_candidate_id=INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,build=build_identity[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID])
    for _ in range(warmup):
        dispatch_fused_matmul_bias_relu(op_decision=fp_dec,backend="cpu",a=a,b=b,bias=bias,repeats=1,kernel_executable=Path(build_identity[FP32_CANDIDATE_ID]["output_binary"]))
        dispatch_fused_matmul_bias_relu(op_decision=i8_dec,backend="cpu",a=a,b=b,bias=bias,repeats=1,int8_kernel_executable=Path(build_identity[INT8_ROW_MAJOR_CANDIDATE_ID]["output_binary"]))
        dispatch_fused_matmul_bias_relu(op_decision=packed_generic_dec,backend="cpu",a=a,b=b,bias=bias,repeats=1,packed_int8_kernel_executable=Path(build_identity[INT8_PACKED_GENERIC_CANDIDATE_ID]["output_binary"]))
        dispatch_fused_matmul_bias_relu(op_decision=packed_a76_dec,backend="cpu",a=a,b=b,bias=bias,repeats=1,packed_int8_kernel_executable=Path(build_identity[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]["output_binary"]))
    rss_before=rss_kb()
    fp=dispatch_fused_matmul_bias_relu(op_decision=fp_dec,backend="cpu",a=a,b=b,bias=bias,repeats=repeats,kernel_executable=Path(build_identity[FP32_CANDIDATE_ID]["output_binary"]))
    i8=dispatch_fused_matmul_bias_relu(op_decision=i8_dec,backend="cpu",a=a,b=b,bias=bias,repeats=repeats,int8_kernel_executable=Path(build_identity[INT8_ROW_MAJOR_CANDIDATE_ID]["output_binary"]))
    packed_generic=dispatch_fused_matmul_bias_relu(op_decision=packed_generic_dec,backend="cpu",a=a,b=b,bias=bias,repeats=repeats,packed_int8_kernel_executable=Path(build_identity[INT8_PACKED_GENERIC_CANDIDATE_ID]["output_binary"]))
    packed_i8=dispatch_fused_matmul_bias_relu(op_decision=packed_a76_dec,backend="cpu",a=a,b=b,bias=bias,repeats=repeats,packed_int8_kernel_executable=Path(build_identity[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]["output_binary"]))
    rss_after=rss_kb()
    ref=reference_fused_matmul_bias_relu(a.data,b.data,bias.data,m,n,k)
    metrics=numerical_metrics(ref,i8.output.data)
    packed_metrics=numerical_metrics(ref,packed_i8.output.data)
    fpstats=latency_stats(fp.samples_ms); i8stats=latency_stats(i8.samples_ms); packed_generic_stats=latency_stats(packed_generic.samples_ms); packedstats=latency_stats(packed_i8.samples_ms)
    mem=theoretical_memory(m,n,k,artifact_path.stat().st_size,packed_path.stat().st_size+packed_bin.stat().st_size)
    observed={"kind":"measured","value":{"process_rss_before_kb":rss_before,"process_rss_after_kb":rss_after,"fp32_binary_size_bytes":Path(build_identity[FP32_CANDIDATE_ID]["output_binary"]).stat().st_size,"int8_binary_size_bytes":Path(build_identity[INT8_ROW_MAJOR_CANDIDATE_ID]["output_binary"]).stat().st_size,"packed_generic_int8_binary_size_bytes":Path(build_identity[INT8_PACKED_GENERIC_CANDIDATE_ID]["output_binary"]).stat().st_size,"packed_int8_binary_size_bytes":Path(build_identity[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]["output_binary"]).stat().st_size,"artifact_size_bytes":artifact_path.stat().st_size,"packed_weight_manifest_bytes":packed_path.stat().st_size,"packed_weight_data_bytes":packed_bin.stat().st_size}}
    ev=build_evaluation_artifact(evaluation_id=f"slice3a-{m}x{n}x{k}-{int(time.time())}",workload_id=workload,shape={"M":m,"N":n,"K":k},input_hash=tensor_f32_sha256(a.data),weight_hash=tensor_f32_sha256(b.data),fp32_candidate_id=fp_dec["quantization"]["selected_candidate_id"],int8_candidate_id=i8_dec["quantization"]["selected_candidate_id"],calibration_artifact=artifact,thread_count=1,build_identity=build_identity,correctness_metrics=metrics,fp32_latency=fpstats,int8_latency=i8stats,theoretical_tensor_bytes=mem,observed_memory=observed,thresholds={"min_cosine_similarity":0.99,"max_relative_l2_error":0.05},timestamp=datetime.now(timezone.utc).isoformat())
    ev["calibration_time_ms"]={"kind":"measured","value":calibration_ms}
    ev["packing_time_ms"]={"kind":"measured","value":packing_ms}
    ev["packed_weight_artifact_id"]=packed_artifact["artifact_id"]
    ev["packed_weight_sha256"]=packed_artifact["artifact_sha256"]
    ev["packed_int8_candidate_id"]=packed_a76_dec["quantization"]["selected_candidate_id"]
    ev["packed_int8_correctness_metrics"]={"kind":"measured","value":packed_metrics}
    ev["packed_generic_int8_latency_statistics"]={"kind":"measured","value":packed_generic_stats}
    ev["packed_int8_latency_statistics"]={"kind":"measured","value":packedstats}
    ev["packed_int8_speedup_vs_current_int8"]={"kind":"derived","value":i8stats["median_ms"]/packedstats["median_ms"] if packedstats["median_ms"] > 0 else None}
    ev["packed_int8_speedup_vs_fp32"]={"kind":"derived","value":fpstats["median_ms"]/packedstats["median_ms"] if packedstats["median_ms"] > 0 else None}
    ev["policy_decision"]=select_with_evidence(ev,min_speedup_margin=0.02)
    candidates={c.candidate_id:c for c in enumerate_complete_candidates()}
    caps=load_codegen_capabilities(target_profile)
    measurements={}
    binary_by={}
    for cid, latency_key, metrics_key in [
        (FP32_CANDIDATE_ID,"fp32_latency_statistics","correctness_metrics"),
        (INT8_ROW_MAJOR_CANDIDATE_ID,"int8_latency_statistics","correctness_metrics"),
        (INT8_PACKED_GENERIC_CANDIDATE_ID,"packed_generic_int8_latency_statistics","packed_int8_correctness_metrics"),
        (INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,"packed_int8_latency_statistics","packed_int8_correctness_metrics"),
    ]:
        meas=create_measurement_artifact(evaluation=ev,candidate=candidates[cid],target=caps,binary_sha256=build_identity[cid]["binary_sha256"],build_manifest=build_identity[cid],packed_artifact=packed_artifact if "packed" in cid else None,calibration_artifact=artifact,latency_key=latency_key,metrics_key=metrics_key)
        mpath=outdir/f"measurement_{cid.replace(':','_')}_{m}x{n}x{k}.json"; write_json_deterministic(mpath,meas)
        measurements[cid]=meas; binary_by[cid]=build_identity[cid]["binary_sha256"]
    selection=select_candidate(list(candidates.values()),target=caps,shape={"M":m,"N":n,"K":k},measurements=measurements,binary_sha256_by_candidate=binary_by,packed_artifact=packed_artifact,calibration_artifact=artifact,has_packed_artifact=True,has_calibration_artifact=True,build_tool_flags=tuple(target_profile["cpuCodegenCapabilities"]["supportedCompilerTargetFlags"]))
    ev["slice3c_selection"]=selection
    sel_meas=measurements[selection["selected_candidate_id"]]
    sel_mpath=outdir/f"measurement_{selection['selected_candidate_id'].replace(':','_')}_{m}x{n}x{k}.json"
    selected_plan=packed_int8_decision(artifact,artifact_path,packed_artifact,packed_path,m,n,k,selected_candidate_id=selection["selected_candidate_id"],build=build_identity[selection["selected_candidate_id"]],measurement_path=sel_mpath,measurement=sel_meas)
    selected_runtime = dispatch_fused_matmul_bias_relu(
        op_decision=selected_plan,
        backend="cpu",
        a=a,
        b=b,
        bias=bias,
        repeats=1,
        packed_int8_kernel_executable=Path(build_identity[selection["selected_candidate_id"]]["output_binary"]),
    )
    ev["selected_runtime_validation"]={
        "kind":"measured",
        "value":{
            "selected_candidate_id":selection["selected_candidate_id"],
            "kernel_id":selected_runtime.kernel_id,
            "binary_sha256":build_identity[selection["selected_candidate_id"]]["binary_sha256"],
            "runtime_no_redecision":True,
            "sample_ms":selected_runtime.samples_ms,
        },
    }
    eval_path=outdir/f"evaluation_{m}x{n}x{k}.json"; write_json_deterministic(eval_path,ev)
    write_json_deterministic(outdir/f"execution_plan_fp32_{m}x{n}x{k}.json",fp_dec)
    write_json_deterministic(outdir/f"execution_plan_int8_{m}x{n}x{k}.json",i8_dec)
    write_json_deterministic(outdir/f"execution_plan_int8_packed_{m}x{n}x{k}.json",selected_plan)
    write_json_deterministic(outdir/f"selection_{m}x{n}x{k}.json",selection)
    return ev

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--warmup",type=int,default=2); ap.add_argument("--repeats",type=int,default=7); ap.add_argument("--shapes",default="4x5x6,16x16x32,37x41x29,128x128x128"); ap.add_argument("--target-profile",type=Path,default=REPO_ROOT/"configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json")
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); target_profile=json.loads(args.target_profile.read_text()); build_identity=ensure_built(args.out,target_profile)
    results=[]
    for idx, spec in enumerate(args.shapes.split(',')):
        m,n,k=map(int,spec.split('x')); results.append(run_shape(args.out,m,n,k,seed=100+idx,warmup=args.warmup,repeats=args.repeats,build_identity=build_identity,target_profile=target_profile))
    summary={"schema_version":"slice3b.packed_weight.summary.v1","results":[{"shape":r["shape"],"metrics":r["correctness_metrics"],"packed_metrics":r["packed_int8_correctness_metrics"],"fp32_latency":r["fp32_latency_statistics"],"int8_latency":r["int8_latency_statistics"],"packed_int8_latency":r["packed_int8_latency_statistics"],"speedup":r["speedup"],"packed_int8_speedup_vs_current_int8":r["packed_int8_speedup_vs_current_int8"],"packed_int8_speedup_vs_fp32":r["packed_int8_speedup_vs_fp32"],"policy_decision":r["policy_decision"]} for r in results]}
    write_json_deterministic(args.out/"summary.json",summary); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
