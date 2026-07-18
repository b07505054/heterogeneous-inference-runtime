#!/usr/bin/env python3
"""Link and execute one exact compiler-selected AArch64 object on target."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

try:
    from deployment.execution_plan.aarch64_native_object_adapter import (
        AArch64NativeObjectAdapter, sha256)
except ImportError:  # minimal Pi bundle: avoid requiring the full Runtime package
    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / "deployment/execution_plan/aarch64_native_object_adapter.py"
    spec = importlib.util.spec_from_file_location("aarch64_native_object_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    AArch64NativeObjectAdapter, sha256 = module.AArch64NativeObjectAdapter, module.sha256


def capture(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def cpu_model():
    text=capture("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":",1)[1].strip()
    p=subprocess.run(["lscpu"],text=True,capture_output=True)
    for line in p.stdout.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":",1)[1].strip()
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",required=True);ap.add_argument("--worker-source",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--warmups",type=int,default=20)
    ap.add_argument("--calls",type=int,default=500);ap.add_argument("--build-dir",required=True)
    ap.add_argument("--calls-per-sample",type=int,default=100)
    ap.add_argument("--session-id",type=int);ap.add_argument("--order-position",type=int)
    args=ap.parse_args()
    started=datetime.now(timezone.utc).isoformat()
    contract,root=AArch64NativeObjectAdapter.load_contract(args.plan)
    adapter=AArch64NativeObjectAdapter(contract,plan_root=root)
    adapter.validate(require_running_target=True);adapter.verify_symbol()
    environment_before={"governor":capture("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
      "current_frequency_khz":capture("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
      "temperature_millic":capture("/sys/class/thermal/thermal_zone0/temp")}
    build=Path(args.build_dir);build.mkdir(parents=True,exist_ok=True)
    exe=build/"aarch64_native_object_worker"
    shape=contract["shape"]
    cmd=["g++","-O3","-std=c++17",
         f"-DM_DIM={shape['m']}",f"-DN_DIM={shape['n']}",f"-DK_DIM={shape['k']}",
         f"-DENTRY_POINT={contract['entry_point']}",
         args.worker_source,str(adapter.object_path()),"-o",str(exe)]
    linked=subprocess.run(cmd,text=True,capture_output=True)
    if linked.returncode: raise RuntimeError(linked.stderr)
    run=subprocess.run([str(exe),str(args.warmups),str(args.calls),
                        str(args.calls_per_sample)],text=True,capture_output=True)
    if run.returncode: raise RuntimeError(run.stdout+run.stderr)
    metrics=json.loads(run.stdout)
    result={"schema_version":1,
      "benchmark_protocol_version":contract.get("benchmark_protocol_version","slice19_aarch64_native_batched_v1"),
      "started_at":started,"completed_at":datetime.now(timezone.utc).isoformat(),
      "plan_sha256":sha256(Path(args.plan)),"worker_source_sha256":sha256(Path(args.worker_source)),
      "session_id":args.session_id,"candidate_order_position":args.order_position,
      "cpu_affinity":sorted(os.sched_getaffinity(0)),
      "environment_before":environment_before,
      "executed":{"candidate_id":contract["candidate_id"],
      "object_sha256":sha256(adapter.object_path()),"entry_point":contract["entry_point"],
      "runtime_redecision_count":0},"metrics":metrics,
      "target":{"hostname":platform.node(),"architecture":platform.machine(),
        "cpu":cpu_model(),"core_count":os.cpu_count(),"kernel":platform.release(),
        "governor":capture("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "current_frequency_khz":capture("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "temperature_millic":capture("/sys/class/thermal/thermal_zone0/temp")},
      "link_command":cmd}
    result["identity_proof"]=adapter.proof(result)
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
