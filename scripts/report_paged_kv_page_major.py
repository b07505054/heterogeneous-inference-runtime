#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def load(p,n):return json.load(open(p/n))
def fmt(x):return f"{x:.6f}"
def table(p):
 m=load(p,"latency_summary.json");lines=[];analysis=[]
 for wid in ["O1","O2","O3","O4","O5"]:
  d={x["candidate_id"]:x for x in m if x["workload_id"]==wid};c=d["cpu_contiguous_kv_fp32_v1"];t=d["cpu_paged_kv_fp32_token_major_v1"];o=d["cpu_paged_kv_fp32_page_major_v1"]
  for label,x in (("contiguous",c),("token-major",t),("page-major",o)):
   s=x["decode_attention"];a=x["append_decode"];lines.append(f"| {wid} | {label} | {fmt(s['median_ms'])} | {fmt(s['mean_ms'])} | {fmt(s['minimum_ms'])} | {fmt(s['p90_ms'])} | {fmt(s['p95_ms'])} | {fmt(s['p99_ms'])} | {fmt(s['stddev_ms'])} | {s['coefficient_of_variation']:.6f} | {s['sample_count']} | {fmt(a['p95_ms'])} |")
  analysis.append(f"| {wid} | {fmt(t['decode_attention']['p95_ms'])} | {fmt(o['decode_attention']['p95_ms'])} | {fmt(o['decode_attention']['p95_ms']-t['decode_attention']['p95_ms'])} | {(t['decode_attention']['p95_ms']/o['decode_attention']['p95_ms']-1)*100:.2f}% | {(o['decode_attention']['p95_ms']/c['decode_attention']['p95_ms']-1)*100:.2f}% |")
 return lines,analysis
ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();h=a.root/"host";p=a.root/"raspberry_pi";hl,ha=table(h);pl,pa=table(p)
counts=load(h,"operation_count_analysis.json");cl=[]
for w in ["O1","O2","O3","O4","O5"]:
 for x in [r for r in counts if r["workload_id"]==w]:cl.append(f"| {w} | {x['candidate_id'].replace('cpu_paged_kv_fp32_','').replace('_v1','')} | {x['block_table_lookup_count']} | {x['logical_division_count']} | {x['logical_modulo_count']} | {x['K_page_base_calculation_count']} | {x['V_page_base_calculation_count']} | {x['token_address_calculation_count']} |")
text=f"""# Paged KV page-major cached-page-base optimization

Real scalar FP32 single-request CPU decode attention. No SIMD, GPU, continuous batching, new layout, or temporary contiguous history.

## Host latency (ms)

| Workload | Candidate | Median | Mean | Min | p90 | p95 | p99 | Stddev | CV | N | Append+decode p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(hl)}

| Workload | Token p95 | Page p95 | Page-token delta | Page-major speedup | Page-major gap vs contiguous |
|---|---:|---:|---:|---:|---:|
{chr(10).join(ha)}

## Raspberry Pi latency (ms)

| Workload | Candidate | Median | Mean | Min | p90 | p95 | p99 | Stddev | CV | N | Append+decode p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pl)}

| Workload | Token p95 | Page p95 | Page-token delta | Page-major speedup | Page-major gap vs contiguous |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pa)}

## Analytical translation operations per decode

| Workload | Strategy | Block-table reads | Divisions | Modulos | K bases | V bases | Token addresses |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cl)}

The token-major formula is `H*T*(D+1)` for block-table reads/divisions/modulos, `H*T` K bases, and `H*D*T` V bases. The page-major formula is `ceil(T/P)` block-table reads per invocation, zero logical token divisions/modulos in page-internal loops, and `H*ceil(T/P)` K and V bases. Physical page IDs are cached once and reused across heads and the K/V passes.

O4 uses the non-sequential physical order recorded in the workload manifest. All other workloads use sequential physical page IDs.
"""
(a.root/"summary.md").write_text(text)
before=load(p,"target_identity_before.json");after=load(p,"target_identity_after.json");before["temperature_after_millicelsius"]=after["temperature_millicelsius"];before["throttling_status_after"]=after["throttling_status"];(p/"target_identity.json").write_text(json.dumps(before,indent=2,sort_keys=True)+"\n")
(a.root/"workload_manifest.json").write_text(json.dumps(load(h,"workload_manifest.json"),indent=2,sort_keys=True)+"\n")
(a.root/"artifact_provenance.json").write_text(json.dumps({"host":load(h,"artifact_provenance.json"),"raspberry_pi":load(p,"artifact_provenance.json")},indent=2,sort_keys=True)+"\n")
