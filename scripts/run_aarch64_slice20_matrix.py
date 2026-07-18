#!/usr/bin/env python3
"""Run Slice 20 plans through the canonical native-object plan runner."""
import argparse,json,subprocess,sys
from pathlib import Path
ORDERS=((1,2,4),(2,4,1),(4,1,2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--bundle",required=True);ap.add_argument("--output-dir",required=True)
 a=ap.parse_args();b=Path(a.bundle).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((b/"domain_manifest.json").read_text());protocol=json.loads((b/"measurement_protocol.json").read_text())
 for d in manifest["domains"]:
  did=d["domain_id"];calls=protocol["calls_per_sample_by_domain"][did]
  for session,order in enumerate(ORDERS,1):
   for pos,uk in enumerate(order,1):
    candidate=f"{did}_uk{uk}";target=out/f"{did}_session{session}_{candidate}.json"
    cmd=["taskset","-c","3",sys.executable,str(b/"scripts/run_aarch64_native_object_plan.py"),
     "--plan",str(b/"plans"/f"{candidate}.json"),"--worker-source",str(b/"native/aarch64_native_object_worker.cpp"),
     "--output",str(target),"--build-dir",str(b/"build"/candidate), "--warmups","30","--calls","500",
     "--calls-per-sample",str(calls),"--session-id",str(session),"--order-position",str(pos)]
    print("RUN",did,session,candidate,flush=True);subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
if __name__=="__main__":main()
