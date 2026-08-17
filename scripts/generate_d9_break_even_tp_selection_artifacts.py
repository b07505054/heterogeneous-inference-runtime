
#!/usr/bin/env python3
from __future__ import annotations
import json, statistics, sys, re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from deployment.vllm_adapter.tp_cost_model import (
    TPCostModel, FittedRegression, MODEL_IDENTITY_FEATURES, build_feature_vector,
    load_communication_predictor, throughput_to_latency_us,
)

ML_PROFILE = Path('/workspace/ml-graph-compiler-runtime/configs/target_profiles/nvidia_rtx4090_d6_distributed_profitability.json')
D9_DIR = REPO_ROOT/'results/runtime_paths/distributed_d9_break_even_tp_selection'
PHASE1 = REPO_ROOT/'results/runtime_paths/nccl_calibration'
PHASE4D = REPO_ROOT/'results/runtime_paths/distributed_d8_vllm_nccl_attribution/break_even'

def write(name, obj):
    D9_DIR.mkdir(parents=True, exist_ok=True)
    (D9_DIR/name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str)+'\n')
    print('wrote', D9_DIR/name)

def load_model():
    prof=json.loads(ML_PROFILE.read_text())['distributedProfitability']
    model=TPCostModel()
    names=['intercept','per_gpu_weight_mb','kv_cache_kb_per_token_per_gpu','gpu_count','input_length','output_length','concurrency']
    def coeffs(block): return [float(block[n]) for n in names]
    model.throughput_models={
        1:FittedRegression(1, coeffs(prof['tp1Coefficients']), prof.get('fitQuality',{}).get('tp1',{}).get('n_samples',0), prof.get('fitQuality',{}).get('tp1',{}).get('r_squared',0)),
        2:FittedRegression(2, coeffs(prof['tp2Coefficients']), prof.get('fitQuality',{}).get('tp2',{}).get('n_samples',0), prof.get('fitQuality',{}).get('tp2',{}).get('r_squared',0)),
    }
    model.frozen=True
    return model

def predictor():
    return load_communication_predictor(json.loads((PHASE1/'communication_cost_profile.json').read_text()), json.loads((PHASE1/'fit_report.json').read_text()))

def parse_wid(wid):
    m = re.match(r'^in(\d+)_out(\d+)_c(\d+)$', wid)
    if not m:
        raise ValueError(wid)
    return tuple(map(int, m.groups()))

def old_decision(model, mf, inp, out, conc, mode):
    fv1=build_feature_vector(mf,1,input_length=inp,output_length=out,concurrency=conc)
    fv2=build_feature_vector(mf,2,input_length=inp,output_length=out,concurrency=conc)
    p1=model.predict_throughput(fv1,1); p2=model.predict_throughput(fv2,2)
    if mode=='d6': return 'tp2' if p2>p1 else 'tp1'
    comm=predictor(); bytes_call=2*int(mf['hidden_size'])*2*2
    p2_adj=1_000_000/(throughput_to_latency_us(p2)+comm.predict_time_us(bytes_call))
    return 'tp2' if p2_adj>p1 else 'tp1'

def evaluate_cell(model, comm, cell):
    mf=MODEL_IDENTITY_FEATURES[cell['hf_model_id']]
    inp,out,conc=parse_wid(cell['workload_id'])
    result=model.decide(model_features=mf,input_length=inp,output_length=out,concurrency=conc,
                        gpu_total_mb=24564.0,gpu_memory_utilization=0.9,max_model_len=2048,max_num_seqs=4,
                        communication_calibration=comm)
    tp1=cell['tp1_tpot']['mean_us']; tp2=cell['tp2_tpot']['mean_us']
    measured=cell['measured_winner']; pred=result['decision']
    return {'model_key':cell['model_key'],'hf_model_id':cell['hf_model_id'],'workload_id':cell['workload_id'],
            'predicted_winner':pred,'measured_winner':measured,
            'net_predicted_benefit_us':result['break_even']['estimated_net_tp2_benefit_us'],
            'measured_tp1_minus_tp2_tpot_us':tp1-tp2,
            'regret_us':0.0 if pred==measured else abs(tp1-tp2),
            'candidate_evidence':result,
            'd6_decision':old_decision(model,mf,inp,out,conc,'d6'),
            'd7_decision':old_decision(model,mf,inp,out,conc,'d7')}

def summary(rows, key):
    regrets=[]; correct=0
    for r in rows:
        ok=r[key]==r['measured_winner']; correct+=int(ok)
        regrets.append(0.0 if ok else abs(r['measured_tp1_minus_tp2_tpot_us']))
    return {'accuracy': correct/len(rows), 'correct': correct, 'total': len(rows), 'mean_regret_us': statistics.mean(regrets), 'max_regret_us': max(regrets)}

def main():
    model=load_model(); comm=predictor()
    rows=[evaluate_cell(model,comm,c) for c in json.loads((PHASE4D/'end_to_end_results.json').read_text())['cells']]
    evidence=[]
    for r in rows:
        e=dict(r['candidate_evidence']); e.update({'model_key':r['model_key'],'workload_id':r['workload_id'],'final_decision':r['predicted_winner']})
        evidence.append(e)
    write('candidate_evidence.json', {'policy_id':'d9_break_even_tp_selector_v1','rows':evidence})
    answer=all(r['predicted_winner']==r['measured_winner'] for r in rows)
    write('measured_boundary_validation.json', {'status':'measured','rows':rows,'success_question':'Can the compiler recover both sides of the empirically measured TP1/TP2 break-even boundary using compute savings and calibrated collective cost, without model-name heuristics?','answer':answer})
    comp={'d6':summary(rows,'d6_decision'),'d7':summary(rows,'d7_decision'),'d9':summary(rows,'predicted_winner')}
    write('comparison_against_d6_d7.json', comp)
    flips=[]
    for r in rows:
        for key in ['d6_decision','d7_decision']:
            if r[key]!=r['predicted_winner']:
                flips.append({'baseline':key.replace('_decision',''),'model_key':r['model_key'],'workload_id':r['workload_id'],'from':r[key],'to':r['predicted_winner'],'measured_winner':r['measured_winner'],'corrective':r['predicted_winner']==r['measured_winner'] and r[key]!=r['measured_winner'],'harmful':r['predicted_winner']!=r['measured_winner'] and r[key]==r['measured_winner']})
    write('decision_flips.json', {'rows':flips,'corrective_flips':sum(f['corrective'] for f in flips),'harmful_flips':sum(f['harmful'] for f in flips)})
    write('regret_summary.json', {'d6':comp['d6'],'d7':comp['d7'],'d9':comp['d9']})
    flips_json=json.loads((D9_DIR/'decision_flips.json').read_text())
    text = (
        '# D9 Break-Even TP Selection\n\n'
        'Decision rule: select TP2 iff `estimated_compute_savings_us - '
        'estimated_communication_penalty_us - estimated_runtime_residual_us > decision_margin_us`.\n\n'
        'Communication penalty is collective-instance-aware: `call_count * '
        'Phase1Profile(collective_kind, bytes_per_call)`. Overlap assumption: `zero`.\n\n'
        'Compute savings uses the existing regression latency delta plus the Phase 4D structural '
        'compute-scale calibration recorded in the target profile; it does not branch on model name.\n\n'
        f'D6 accuracy: {comp["d6"]["accuracy"]:.3f}, mean regret us: {comp["d6"]["mean_regret_us"]:.3f}.\n'
        f'D7 accuracy: {comp["d7"]["accuracy"]:.3f}, mean regret us: {comp["d7"]["mean_regret_us"]:.3f}.\n'
        f'D9 accuracy: {comp["d9"]["accuracy"]:.3f}, mean regret us: {comp["d9"]["mean_regret_us"]:.3f}, '
        f'max regret us: {comp["d9"]["max_regret_us"]:.3f}.\n'
        f'Corrective flips: {flips_json["corrective_flips"]}; harmful flips: {flips_json["harmful_flips"]}.\n\n'
        'Success question answer: ' +
        ('yes, D9 recovers both sides of the measured TP1/TP2 boundary without model-name heuristics.'
         if answer else 'no, D9 does not recover all measured cells.') + '\n'
    )
    (D9_DIR/'README.md').write_text(text)
    print('wrote', D9_DIR/'README.md')
if __name__=='__main__': main()
