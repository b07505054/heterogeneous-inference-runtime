#!/usr/bin/env python3
import argparse,json,math,statistics,collections
from pathlib import Path

def median(xs): return statistics.median(xs) if xs else None
def gmean(vals):
 vals=[v for v in vals if v and v>0]
 return math.exp(sum(math.log(v) for v in vals)/len(vals)) if vals else None

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--raw',required=True); ap.add_argument('--out',required=True)
 args=ap.parse_args(); man=json.load(open(args.manifest)); raw=json.load(open(args.raw)); tie=man['analysis_formulas']['tie_threshold_percent']/100.0
 recs=raw['records']; expected=len(man['workloads'])*len(man['modes']['project']+man['modes']['executorch'])*man['protocol']['session_count']
 correctness=[r for r in recs if not r['result'].get('correctness_passed')]
 per=collections.defaultdict(list)
 for r in recs:
  per[(r['workload_id'],r['mode'])].append(r['warm_median_ms'])
 summary=[]
 for w in man['workloads']:
  row={'workload_id':w['workload_id'],'category':w['category'],'m':w['m'],'n':w['n'],'k':w['k'],'mnk':w['mnk'],'modes':{}}
  for mode in man['modes']['project']+man['modes']['executorch']:
   vals=per[(w['workload_id'],mode)]
   row['modes'][mode]={'session_medians_ms':vals,'median_of_session_medians_ms':median(vals),'mean_session_median_ms':statistics.fmean(vals) if vals else None}
  ps=row['modes']['P-SERIAL']['median_of_session_medians_ms']; p4=row['modes']['P-4T']['median_of_session_medians_ms']; pp=row['modes']['P-POLICY']['median_of_session_medians_ms']; etd=row['modes']['ET-DEFAULT']['median_of_session_medians_ms']; et1=row['modes']['ET-1T-REQUESTED']['median_of_session_medians_ms']; et4=row['modes']['ET-4T-REQUESTED']['median_of_session_medians_ms']
  bestp=min(ps,p4); row['project_oracle_mode']='P-SERIAL' if ps<=p4 else 'P-4T'; row['project_policy_regret_percent']=(pp-bestp)/bestp*100; row['p_policy_vs_et_default_speedup']=etd/pp if pp and etd else None; row['et_default_vs_p_policy_speedup']=pp/etd if pp and etd else None
  bestet=min(et1,et4); row['et_best_requested_mode']='ET-1T-REQUESTED' if et1<=et4 else 'ET-4T-REQUESTED'; row['et_default_regret_vs_requested_percent']=(etd-bestet)/bestet*100
  ratio=etd/pp
  row['practical_result']='tie' if abs(ratio-1)<=tie else ('project_faster' if ratio>1 else 'executorch_faster')
  summary.append(row)
 speedups=[r['p_policy_vs_et_default_speedup'] for r in summary]
 wins=sum(1 for r in summary if r['practical_result']=='project_faster'); losses=sum(1 for r in summary if r['practical_result']=='executorch_faster'); ties=sum(1 for r in summary if r['practical_result']=='tie')
 project_regrets=[r['project_policy_regret_percent'] for r in summary]
 practical={'geomean_speedup_project_policy_vs_et_default':gmean(speedups),'median_speedup':median(speedups),'win_tie_loss':{'project_faster':wins,'tie':ties,'executorch_faster':losses},'worst_project_slowdown_vs_et_default':max((1/s for s in speedups if s), default=None),'best_project_speedup_vs_et_default':max(speedups)}
 verdict='MIXED_RESULTS_NO_SINGLE_WINNER_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY'
 if wins and not losses: verdict='PROJECT_FASTER_ON_FROZEN_EXECUTORCH_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY'
 elif losses and not wins: verdict='EXECUTORCH_FASTER_ON_FROZEN_PROJECT_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY'
 elif not wins and not losses: verdict='STATISTICALLY_TIED_ON_FROZEN_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY'
 analysis={'schema':'e2_analysis','manifest_sha256':man['manifest_sha256'],'expected_records':expected,'actual_records':len(recs),'correctness_failures':len(correctness),'per_workload':summary,'project_decision_quality':{'mean_regret_percent':statistics.fmean(project_regrets),'median_regret_percent':median(project_regrets),'max_regret_percent':max(project_regrets),'exact_match_rate':sum(1 for r in summary if r['project_policy_regret_percent']<=1e-9)/len(summary)},'practical_p_policy_vs_et_default':practical,'verdict':verdict}
 Path(args.out).write_text(json.dumps(analysis,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'records':len(recs),'correctness_failures':len(correctness),'verdict':verdict,'geomean_speedup':practical['geomean_speedup_project_policy_vs_et_default'],'wtl':practical['win_tie_loss']},indent=2))
if __name__=='__main__': main()
