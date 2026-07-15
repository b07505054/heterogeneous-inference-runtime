#!/usr/bin/env python3
"""Render complete numerical KV-selection reports from canonical JSON evidence."""
import argparse, json, math
from pathlib import Path

NAMES={"cpu_contiguous_kv_fp32_v1":"contiguous","cpu_paged_kv_fp32_v1":"paged"}
KERNEL={"cpu_contiguous_kv_fp32_v1":"cpu_attention_decode_fp32","cpu_paged_kv_fp32_v1":"cpu_attention_decode_paged_kv_fp32"}
def load(p,n): return json.loads((p/n).read_text())
def f(x): return f"{x:.6f}"
def pct(x): return f"{100*x:.2f}%"
def score(r,rows,w):
    return w["latency_weight"]*r["append_decode_p95_ms"]/min(x["append_decode_p95_ms"] for x in rows)+w["memory_weight"]*r["request_owned_bytes"]/max(x["request_owned_bytes"] for x in rows)+w["fragmentation_weight"]*r["internal_fragmentation_ratio"]
def render(root,title):
    matrix=load(root,"candidate_matrix.json"); manifest={x["workload_id"]:x for x in load(root,"workload_manifest.json")}; correctness={(x["workload_id"],x["candidate_id"],x.get("page_tokens")):x for x in load(root,"correctness_summary.json")}; selections=load(root,"selection_results.json"); regrets={(x["workload_id"],x["objective"]):x for x in load(root,"regret_analysis.json")}; proofs={(x["workload_id"],x["runtime_executed_candidate_id"],x.get("selected_page_size")):x for x in load(root,"runtime_proof_traces.json")}; objectives=load(root,"objective_definitions.json"); admission=load(root,"admission_analysis.json")
    by={}
    for r in matrix: by.setdefault(r["workload_id"],[]).append(r)
    latency=[];memory=[];pages=[];selection=[];runtime=[]
    for wid,rows in sorted(by.items()):
        best=min(x["append_decode_p95_ms"] for x in rows); w=manifest[wid]; bpt=2*w["heads"]*w["head_dim"]*4
        for r in sorted(rows,key=lambda x:(x["page_tokens"] is not None,x.get("page_tokens") or 0)):
            st=r["statistics"]; c=correctness[(wid,r["candidate_id"],r.get("page_tokens"))]
            latency.append(f"| {wid} | {NAMES[r['candidate_id']]} | {r.get('page_tokens') or '-'} | {f(st['median_ms'])} | {f(st['p95_ms'])} | {f(st['p99_ms'])} | {f(r['full_loop_latency_ms'])} | {pct(r['append_decode_p95_ms']/best-1)} | {st['sample_count']} | {'PASS' if r['correctness_passed'] else 'FAIL'} |")
            allocated=1 if r.get("page_tokens") is None else r["request_owned_bytes"]//(r["page_tokens"]*bpt)
            memory.append(f"| {wid} | {NAMES[r['candidate_id']]} | {w['maximum_capacity']} | {w['final_tokens']} | {r['logical_used_bytes']} | {r['request_owned_bytes']} | {r['total_reserved_pool_bytes']} | {r['request_owned_bytes']-r['logical_used_bytes']} | {r['logical_used_bytes']/r['request_owned_bytes']:.6f} | {r['internal_fragmentation_ratio']:.6f} | {allocated} | {r.get('page_tokens') or '-'} |")
        paged=[x for x in rows if x.get("page_tokens") is not None]; minlat=min(x["append_decode_p95_ms"] for x in paged); minmem=min(x["request_owned_bytes"] for x in paged); bw=objectives["balanced"]; mins=min(score(x,rows,bw) for x in paged)
        ties=lambda xs: ",".join(str(x["page_tokens"]) for x in xs)
        pages.append(f"| {wid} | {ties([x for x in paged if math.isclose(x['append_decode_p95_ms'],minlat)])} | {ties([x for x in paged if x['request_owned_bytes']==minmem])} | {ties([x for x in paged if math.isclose(score(x,rows,bw),mins)])} |")
    for s in selections:
        wid=s["workload_id"]; rows=by[wid]; chosen=next(x for x in rows if x["candidate_id"]==s["candidate_id"] and x.get("page_tokens")==s.get("page_tokens")); rg=regrets[(wid,s["objective"])]; oracle=next(x for x in rows if x["candidate_id"]==s["oracle_candidate_id"] and x.get("page_tokens")==s.get("oracle_page_tokens")); pr=proofs[(wid,s["candidate_id"],s.get("page_tokens"))]
        ident=lambda c,p:f"{NAMES[c]}{('-'+str(p)) if p else ''}"
        selection.append(f"| {wid} | {s['objective']} | {ident(s['oracle_candidate_id'],s.get('oracle_page_tokens'))} | {ident(s['candidate_id'],s.get('page_tokens'))} | {ident(pr['runtime_executed_candidate_id'],pr.get('selected_page_size'))} | {s['selection_reason']} | {f(chosen['append_decode_p95_ms'])} | {f(oracle['append_decode_p95_ms'])} | {f(rg['absolute_latency_regret_ms'])} | {rg['memory_regret_bytes']} | {f(rg['objective_score_regret'])} |")
        runtime.append({"workload_id":wid,"objective":s["objective"],"compiler_selected_candidate":s["candidate_id"],"runtime_executed_candidate":pr["runtime_executed_candidate_id"],"compiler_selected_kernel":KERNEL[s["candidate_id"]],"runtime_executed_kernel":pr["runtime_executed_kernel_id"],"plan_sha256":pr["execution_plan_hash"],"runtime_layout_reselection_count":pr["runtime_layout_reselection_count"],"runtime_kernel_reselection_count":pr["runtime_kernel_reselection_count"],"temporary_full_history_materialization_count":pr["temporary_full_history_materialization_count"]})
    adm=[]
    for x in admission:
        w=manifest[x["workload_id"]]; bpt=2*w["heads"]*w["head_dim"]*4; delta=x["paged_formula"]-x["contiguous"]
        rel="inf" if x["contiguous"]==0 else pct(x["paged_formula"]/x["contiguous"]-1)
        adm.append(f"| {x['workload_id']} | {x['distribution']} | {x['budget_mib']} | {x['contiguous']} | {x['paged_formula']} | {delta} | {rel} | {w['maximum_capacity']} | 16 | contiguous={w['maximum_capacity']}×{bpt}; paged=E[ceil(tokens/16)×16×{bpt}] |")
    text=f"""# {title}

Exact-target measured-profile evaluation; single-request native FP32 CPU execution.

## Latency

| Workload | Candidate | Page | Median ms | p95 ms | p99 ms | Full loop median ms | Slowdown vs best | Samples | Correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(latency)}

## Memory

| Workload | Candidate | Capacity tokens | Valid tokens | Logical bytes | Request-owned bytes | Reserved process/pool bytes | Unused owned bytes | Utilization | Fragmentation | Allocated pages/objects | Page |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(memory)}

## Compiler selections

| Workload | Objective | Best legal | Selected | Executed | Reason | Selected p95 | Best p95 | Latency regret ms | Memory regret bytes | Score regret |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(selection)}

## Paged page-size results

| Workload | Best p95 page | Best owned-memory page | Best balanced-score page |
|---|---|---|---|
{chr(10).join(pages)}

## Formula-based admission analysis, not real concurrent serving

| Workload | Distribution | MiB | Contiguous | Paged | Improvement | Relative | Capacity | Page | Byte formula |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(adm)}
"""
    (root/"summary.md").write_text(text);(root/"runtime_proof.json").write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
    return matrix,selections
parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
host,hs=render(args.root/"host","Host KV-selection evaluation")
pi,ps=render(args.root/"raspberry_pi","Raspberry Pi KV-selection evaluation")
def winners(rows):
    out=[]
    for wid in sorted({x["workload_id"] for x in rows}):
        rs=[x for x in rows if x["workload_id"]==wid];best=min(rs,key=lambda x:x["append_decode_p95_ms"]);paged=min((x for x in rs if x.get("page_tokens") is not None),key=lambda x:x["append_decode_p95_ms"]);out.append((wid,NAMES[best["candidate_id"]]+(f"-{best.get('page_tokens')}" if best.get("page_tokens") else ""),paged["page_tokens"],paged["append_decode_p95_ms"]/best["append_decode_p95_ms"]-1))
    return out
h={x[0]:x for x in winners(host)};p={x[0]:x for x in winners(pi)};common=sorted(set(h)&set(p))
cross=[f"| {w} | {h[w][1]} | {p[w][1]} | {h[w][2]} | {p[w][2]} | {pct(h[w][3])} | {pct(p[w][3])} |" for w in common]
(args.root/"cross_target_summary.md").write_text("# Cross-target KV-selection comparison\n\nHost and Raspberry Pi remain separate measured lookup identities.\n\n| Workload | Host winner | Pi winner | Host best paged page | Pi best paged page | Host paged overhead | Pi paged overhead |\n|---|---|---|---:|---:|---:|---:|\n"+"\n".join(cross)+"\n\nRequired lookup key: target identity, CPU identity, workload identity, candidate identity, page size, and objective.\n")
