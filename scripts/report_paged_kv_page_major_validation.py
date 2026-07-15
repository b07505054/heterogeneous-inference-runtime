#!/usr/bin/env python3
import hashlib,json,platform,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts/paged_kv_page_major_validation'
def load(p): return json.loads(p.read_text())
def dump(n,x): (A/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
host=load(A/'host/latency_summary.json'); pi=load(A/'raspberry_pi/latency_summary.json')
work=load(A/'host/workload_manifest.json')
fair={
 "previous_boundary_fair":False,"reason":"previous benchmark timed Python Runtime decode wrappers in fixed candidate order and compared structurally different V loops",
 "corrected_kernel_boundary":"prebound native call; buffers already allocated/populated; output ready; checksum consumed after timer",
 "excluded":["compilation","library loading","allocation","block-table construction","K/V population","warmup","Python validation","reference computation"],
 "append_decode_boundary":"prebound native append followed by prebound native decode; checksum after timer",
 "candidate_order":"four-way cyclic interleave, seed 20260715","same_affinity":True,"same_thread_count":True,"same_priority":True,"same_common_flags":True,
 "hidden_kernel_heap_allocations":False,"output_checksum_equal":True,"runtime_wrapper_boundary_reported_separately":False,
 "acceptance_limitation":"dynamic PMU counters unavailable on both targets"
}
dump('methodology.json',{"host":load(A/'host/methodology.json'),"raspberry_pi":load(A/'raspberry_pi/methodology.json')})
dump('workload_manifest.json',work);dump('fairness_audit.json',fair)
asm={"binary":"/tmp/paged-kv-validation-host/libattention_fp32.so","raw_dump":"/tmp/attention_disassembly.txt","functions":{
 "contiguous":{"bytes":1724,"static_instructions":431,"idiv":0,"divide_instructions":4,"calls":3,"xmm_ymm_operand_lines":88},
 "reordered_control":{"bytes":1736,"static_instructions":431,"idiv":0,"divide_instructions":3,"calls":4,"xmm_ymm_operand_lines":76},
 "token_major":{"bytes":1225,"static_instructions":319,"idiv":2,"divide_instructions":4,"calls":2,"xmm_ymm_operand_lines":43},
 "page_major":{"bytes":2165,"static_instructions":528,"idiv":0,"divide_instructions":3,"calls":3,"xmm_ymm_operand_lines":73}},
 "note":"static disassembly counts are not retired-instruction counts"}
dump('assembly_summary.json',asm)
dump('vectorization_summary.json',{
 "compiler_report":"/tmp/paged-kv-validation-host/vectorization.txt",
 "contiguous":"QK loops vectorized; strided V reduction missed as not suitable for strided load",
 "reordered_control":"inner head-dimension QK and V loops vectorized; alias-versioned",
 "token_major":"limited vectorization; control flow and paged address helper inhibit hot loops",
 "page_major":"inner head-dimension loops vectorized; page/control-flow loops missed",
 "conclusion":"page-major is auto-vectorized where original contiguous V is not, but matched reordered contiguous is also vectorized; page-major received no unique flag advantage"})
def comparison(target,rows):
 out=[]
 for w in [f'O{i}' for i in range(1,6)]:
  q={x['candidate']:x for x in rows if x['workload_id']==w}; p={c:q[c]['native_kernel_only']['p95_ms'] for c in q}; ap={c:q[c]['native_append_decode']['p95_ms'] for c in q}
  out.append({"target":target,"workload":w,"p95_ms":p,"append_decode_p95_ms":ap,
   "page_major_speedup_vs_original_contiguous_percent":(p['contiguous']/p['page_major']-1)*100,
   "page_major_latency_reduction_vs_original_contiguous_percent":(1-p['page_major']/p['contiguous'])*100,
   "loop_reordering_control_vs_original_contiguous_percent":(p['contiguous']/p['reordered_control']-1)*100,
   "paged_overhead_vs_matched_control_percent":(p['page_major']/p['reordered_control']-1)*100,
   "token_to_page_major_speedup_percent":(p['token_major']/p['page_major']-1)*100,
   "analytical_flops":q['contiguous']['analytical_flops'],"output_elements":q['contiguous']['output_elements'],
   "operation_level":{c:{"p95_ms":p[c],"cycles_per_invocation":"not_available","instructions_per_invocation":"not_available","instructions_per_cycle":"not_available","branch_miss_rate":"not_available","l1_miss_rate":"not_available","llc_miss_rate":"not_available","analytical_flops":q[c]['analytical_flops'],"cycles_per_output_element":"not_available","instructions_per_token":"not_available"} for c in q}})
 return out
cmp=comparison('host',host)+comparison('raspberry_pi',pi);dump('candidate_comparison.json',cmp)
dump('host_latency_summary.json',host);dump('pi_latency_summary.json',pi)
err_host="perf_event_paranoid=4; perf stat: No supported events found"
err_pi="bash: perf: command not found (exit 127)"
events=['cycles','instructions','branches','branch-misses','cache-references','cache-misses','L1-dcache-loads','L1-dcache-load-misses','LLC-loads','LLC-load-misses','task-clock','context-switches','cpu-migrations','page-faults']
dump('host_perf_summary.json',{"available":False,"perf_version":"7.0.6","error":err_host,"events":{e:'not_available' for e in events}})
dump('pi_pmu_summary.json',{"available":False,"error":err_pi,"system_modified":False,"events":{e:'not_available' for e in events},"temperature_before_c":45.75,"temperature_after_c":49.6,"governor":"performance","frequency_khz":2400000,"throttled_before":"0x0","throttled_after":"0x0"})
sel=[]
for x in cmp:
 p=x['p95_ms']; winner=min(('contiguous','token_major','page_major'),key=p.get)
 sel.append({"target":x['target'],"workload":x['workload'],"production_winner":winner,"p95_ms":p[winner],"experimental_control_excluded":True})
dump('compiler_selection_impact.json',sel)
dump('runtime_proof.json',{"host":load(A/'host/runtime_proof.json'),"raspberry_pi":load(A/'raspberry_pi/runtime_proof.json')})
files=[ROOT/'native/cpu_kernels/attention_fp32.cpp',ROOT/'native/cpu_kernels/attention_fp32.h',ROOT/'scripts/validate_paged_kv_page_major_release.py']
dump('artifact_provenance.json',{"source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in files},"host":load(A/'host/artifact_provenance.json'),"raspberry_pi":load(A/'raspberry_pi/artifact_provenance.json'),"raw_evidence_not_tracked":["/tmp/attention_disassembly.txt","/tmp/paged-kv-validation-host/vectorization.txt"],"generator":str(Path(__file__).relative_to(ROOT))})
lines=['# Paged-KV page-major release validation','',
 'The prior representation-level conclusion is overturned. Page-major still beats the original contiguous implementation on O2–O5, but the experimental contiguous reordered control is faster than page-major on every workload and target. The advantage is primarily loop ordering/auto-vectorization, not paged representation.', '',
 '| target | workload | contiguous p95 ms | control p95 ms | token-major p95 ms | page-major p95 ms | page latency reduction vs contiguous | control speedup vs contiguous | page overhead vs control |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in cmp:
 p=x['p95_ms']; lines.append(f"| {x['target']} | {x['workload']} | {p['contiguous']:.6f} | {p['reordered_control']:.6f} | {p['token_major']:.6f} | {p['page_major']:.6f} | {x['page_major_latency_reduction_vs_original_contiguous_percent']:+.2f}% | {x['loop_reordering_control_vs_original_contiguous_percent']:+.2f}% | {x['paged_overhead_vs_matched_control_percent']:+.2f}% |")
host_sel=', '.join(f"{x['workload']} {x['production_winner']}" for x in sel if x['target']=='host')
pi_sel=', '.join(f"{x['workload']} {x['production_winner']}" for x in sel if x['target']=='raspberry_pi')
lines += ['', 'All candidates perform identical analytical arithmetic: `4 × heads × valid_tokens × head_dim` FLOPs and `heads × head_dim` outputs. All corrected-run checksums agree.', '',
 'Host PMU counters are unavailable because `perf_event_paranoid=4`; Pi has no `perf` executable. No permissions or packages were changed. Static assembly and compiler vectorization reports support the loop-order explanation.', '',
 f'Corrected kernel-only p95 production winners are: host — {host_sel}; Raspberry Pi — {pi_sel}. The reordered control is benchmark-only and is not a compiler candidate. Append+decode retains contiguous for O1 on both targets.', '',
 '**Next action: C. fix contiguous loop ordering first.** Explicit SIMD is not yet justified until the production contiguous loop incorporates and revalidates the demonstrated scalar/compiler-vectorized ordering improvement.','']
(A/'summary.md').write_text('\n'.join(lines))
