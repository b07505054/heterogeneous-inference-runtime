#!/usr/bin/env python3
import argparse, hashlib, json, random, struct, subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"
PROJECT_POLICY_ID = "p1d1_raspberry_pi_mnk_threshold_thread_policy"
PROJECT_POLICY_VERSION = "1"
PROJECT_THRESHOLD = 262144
EXECUTORCH_TAG = "v1.3.1"
EXECUTORCH_COMMIT = "e2f18eb23c45bd22ca332b0b8b49a81de304b472"
XNNPACK_COMMIT = "1adaa7c709d4839d29e1f219cb962b01c9e6a905"
RUNNER_SHA = "eb3068fb1742e4172a459f9f4c5aebd2dd9dd43151e214ff1402ea925d4e2809"
COMPILER_COMMIT = "b67cd644568e7f53a64370f926e241e4e42ebe10"
RUNTIME_COMMIT = "1ab411fab87f43da8c3f4540b4540534c9dbbf2b"
CAPABILITIES_COMMIT = "aac593da0bdde7a95c38c03920fc4d00b73011db"

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def gen_values(count, rng):
    return [rng.uniform(-3,3) for _ in range(count)]

def pack(vals):
    return struct.pack('<' + 'f'*len(vals), *vals)

def reference(a,b,bias,m,n,k):
    out=[]
    for i in range(m):
        for j in range(n):
            s=0.0
            for kk in range(k):
                s += a[i*k+kk]*b[kk*n+j]
            v=s+bias[j]
            out.append(v if v > 0.0 else 0.0)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--p1d-manifest', required=True)
    ap.add_argument('--pte-dir', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    p1d=json.load(open(args.p1d_manifest))
    pte_dir=Path(args.pte_dir)
    workloads=[]
    for w in p1d['workloads']:
        if w['split'] not in ('calibration','held_out'):
            continue
        m,n,k=w['m'],w['n'],w['k']
        rng=random.Random(w['seed'])
        a=gen_values(m*k,rng); b=gen_values(k*n,rng); bias=gen_values(n,rng)
        ref=reference(a,b,bias,m,n,k)
        pte=pte_dir/f"fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack.pte"
        report=pte_dir/f"fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack_export_report.json"
        if not pte.exists():
            raise SystemExit(f'missing pte {pte}')
        if not report.exists():
            raise SystemExit(f'missing report {report}')
        er=json.load(open(report))
        workloads.append({
            'workload_id': w['workload_id'], 'split': w['split'], 'category': w['category'],
            'm': m, 'n': n, 'k': k, 'mnk': m*n*k, 'seed': w['seed'],
            'input_hashes': {'a': sha_bytes(pack(a)), 'b': sha_bytes(pack(b)), 'bias': sha_bytes(pack(bias)), 'reference': sha_bytes(pack(ref))},
            'input_sizes_bytes': {'a': len(pack(a)), 'b': len(pack(b)), 'bias': len(pack(bias)), 'reference': len(pack(ref))},
            'executorch_xnnpack_pte': {'filename': pte.name, 'sha256': sha_file(pte), 'bytes': pte.stat().st_size, 'classification': er.get('classification'), 'delegate_call_count': er.get('delegate_call_count')},
            'project_policy_selection': {'mode': 'P-SERIAL' if m*n*k < PROJECT_THRESHOLD else 'P-4T', 'metric': 'matmul_mnk', 'threshold': PROJECT_THRESHOLD, 'boundary': '< threshold serial; >= threshold 4-thread split-M'},
        })
    manifest={
        'schema': 'e2_raspberry_pi5_project_vs_executorch_xnnpack_manifest',
        'schema_version': 1,
        'comparison_id': 'E2_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_2026_07_13',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'semantic_contract_version': 'fp32_matmul_bias_relu_v1',
        'semantic': 'Y = ReLU(A @ B + bias)',
        'input_generation': {'algorithm': 'python random.Random(seed).uniform(-3, 3), tensors generated in order A, B, bias and serialized little-endian float32', 'distribution': 'uniform[-3,3]', 'dtype': 'fp32', 'layout': 'contiguous row-major'},
        'correctness': {'abs_tolerance': 1e-3, 'rel_tolerance': 1e-4, 'nan_inf_rule': 'no NaN or Inf expected; mismatch fails'},
        'project': {'compiler_commit': COMPILER_COMMIT, 'runtime_commit': RUNTIME_COMMIT, 'capabilities_commit': CAPABILITIES_COMMIT, 'kernel_id': PROJECT_KERNEL_ID, 'dtype': 'fp32', 'tile': {'BM':32,'BN':128,'BK':32}, 'policy_id': PROJECT_POLICY_ID, 'policy_version': PROJECT_POLICY_VERSION, 'metric': 'matmul_mnk', 'threshold': PROJECT_THRESHOLD},
        'executorch': {'tag': EXECUTORCH_TAG, 'commit': EXECUTORCH_COMMIT, 'xnnpack_commit': XNNPACK_COMMIT, 'runner_sha256': RUNNER_SHA, 'graph_classification': 'FULL_REGION_DELEGATED_FUSION_UNKNOWN'},
        'pi_environment': {'hostname': 'edgeaiplatform', 'hardware': 'Raspberry Pi 5 Model B Rev 1.1', 'os': 'Debian 13', 'arch': 'aarch64', 'cpu': 'Cortex-A76 four cores', 'governor': 'performance', 'affinity': '0-3'},
        'modes': {'project': ['P-SERIAL','P-4T','P-POLICY'], 'executorch': ['ET-DEFAULT','ET-1T-REQUESTED','ET-4T-REQUESTED']},
        'protocol': {'warmup_count': 5, 'timed_repeat_count': 20, 'total_invocations_per_mode_session': 25, 'session_count': 3, 'rotated_execution_order_rule': 'session 0 listed order, session 1 reversed, session 2 deterministic shuffle seed 90210'},
        'timing_boundaries': {'load_time': 'program or artifact load reported separately where runner exposes it', 'cold_first_inference': 'first method/kernel invocation after load and input preparation', 'warm_execution': 'per-invocation method/kernel execution after first five warmups; excludes process startup, artifact load, input generation, reference computation, correctness, and result serialization', 'end_to_end_invocation': 'not equivalent with current unmodified runners; reported as unavailable for formal comparison'},
        'rss_methodology': 'external /proc sampler records VmRSS/status where process lifetime permits; final E2 uses available process RSS observations and discloses unresolved gaps',
        'thread_observation': 'runner requested thread logs plus /proc task observation where practical; requested configurations are not assumed to prove active worker utilization',
        'analysis_formulas': {'tie_threshold_percent': 5.0, 'per_workload_latency': 'median of per-session warm medians', 'speedup': 'denominator latency / numerator latency', 'project_regret_percent': '(P_POLICY - min(P_SERIAL,P_4T))/min(P_SERIAL,P_4T)*100', 'cross_system': 'latency ratios and win/tie/loss only; no cross-system regret'},
        'claim_boundaries': ['Raspberry Pi 5 only', 'FP32 fused MatMul + Bias + ReLU only', 'frozen workload suite only', 'implementation outcome comparison, not pure scheduling comparison', 'no general ExecuTorch superiority/inferiority claim'],
        'workloads': workloads,
    }
    # Freeze hash excludes the manifest_hash field itself.
    frozen=json.dumps(manifest, sort_keys=True, separators=(',',':')).encode()
    manifest['manifest_sha256']=hashlib.sha256(frozen).hexdigest()
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(manifest['comparison_id'])
    print(manifest['manifest_sha256'])
    print(len(workloads))
if __name__ == '__main__':
    main()
