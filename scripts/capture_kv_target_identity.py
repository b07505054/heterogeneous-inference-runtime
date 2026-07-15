#!/usr/bin/env python3
"""Capture target evidence for the KV-selection benchmark."""
import argparse, json, os, pathlib, platform, subprocess, time

def read(path):
    try: return path.read_text().strip()
    except OSError: return None

def run(command):
    try: return subprocess.run(command, text=True, capture_output=True, check=False).stdout.strip()
    except OSError as error: return str(error)

parser=argparse.ArgumentParser();parser.add_argument("--output",type=pathlib.Path,required=True);args=parser.parse_args()
cpus=list(pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
result={
    "captured_at_epoch":time.time(),"hostname":platform.node(),"platform":platform.platform(),
    "architecture":platform.machine(),"cpu_count":os.cpu_count(),"lscpu":run(["lscpu"]),
    "gxx_version":run(["g++","--version"]).splitlines()[0],
    "governors":{p.name:read(p/"cpufreq/scaling_governor") for p in cpus},
    "frequencies":{p.name:{"current_khz":read(p/"cpufreq/scaling_cur_freq"),"max_khz":read(p/"cpufreq/cpuinfo_max_freq")} for p in cpus},
    "temperature_millicelsius":{p.name:read(p/"temp") for p in pathlib.Path("/sys/class/thermal").glob("thermal_zone*")},
    "throttling_status":run(["vcgencmd","get_throttled"]),
    "native_compile_flags":["-O3","-std=c++17","-fPIC","-shared"],"benchmark_process_threads":1,
}
args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))
