#!/usr/bin/env python3
import argparse,json,itertools,statistics
from pathlib import Path
P=("decode_first","prefill_first","chunked_balanced","slo_aware")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
 ap.add_argument("inputs",nargs="+",type=Path);a=ap.parse_args();rows=[];reg={p:[] for p in P}
 for path in a.inputs:
  x=json.loads(path.read_text())["summary"]
  for trace,d in x.items():
   values={p:d[p]["objective"]["mean"] for p in P};best=min(values.values())
   for p in P:reg[p].append((values[p]-best)/best)
   for left,right in itertools.combinations(P,2):
    delta=values[left]-values[right];scale=min(values[left],values[right])
    rows.append({"source":str(path),"trace":trace,"policy_a":left,"policy_b":right,
     "measured_objective_delta":delta,"target":
      "tie" if abs(delta)/scale<=.02 else ("a_wins" if delta<0 else "b_wins"),
     "target_provenance":"measured_real_qwen_wall_clock"})
 (a.output_dir/"policy_pairwise_dataset.json").write_text(json.dumps(rows,indent=2)+"\n")
 baseline={p:{"mean_regret":statistics.fmean(v),"p95_regret":sorted(v)[round(.95*(len(v)-1))],
  "maximum_regret":max(v)} for p,v in reg.items()}
 robust=min(P,key=lambda p:(baseline[p]["mean_regret"],baseline[p]["p95_regret"],p))
 (a.output_dir/"robust_fixed_policy_baseline.json").write_text(json.dumps({
  "policies":baseline,"selected_robust_default":robust,
  "selection_data":"S2.8 training plus S2.9 development only"},indent=2)+"\n")
if __name__=="__main__":main()
