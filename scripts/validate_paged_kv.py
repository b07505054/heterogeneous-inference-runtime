#!/usr/bin/env python3
"""Build and validate native paged-KV decode without test-only dependencies."""
from array import array
import argparse, hashlib, json, math, subprocess
from pathlib import Path

from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession

ROOT = Path(__file__).resolve().parents[1]

def values(count: int, seed: int) -> array:
    return array("f", ((((i * 1103515245 + seed * 12345) & 0xffff) / 32768) - 1 for i in range(count)))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    library = args.work_dir / "libattention_fp32.so"
    command = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared", str(ROOT / "native/cpu_kernels/attention_fp32.cpp"), "-o", str(library)]
    subprocess.run(command, check=True)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    heads, dim, page_tokens, pages, maximum, prompt = 2, 32, 8, 8, 64, 7
    one = heads * page_tokens * dim * 4
    blocks = (maximum + page_tokens - 1) // page_tokens
    contract = {"kv_candidate_id":"cpu_paged_kv_fp32_v1","kv_layout_kind":"paged_phd_contiguous","pool_artifact_ref":library.name,"pool_artifact_sha256":digest,"pool_artifact_version":"hir.paged_kv.v1","dtype":"fp32","batch":1,"num_kv_heads":heads,"head_dim":dim,"page_tokens":page_tokens,"num_physical_pages":pages,"maximum_logical_tokens":maximum,"maximum_logical_blocks":blocks,"block_table_length":blocks,"block_table_element_type":"int32","invalid_page_sentinel":-1,"k_page_strides":[heads*page_tokens*dim,page_tokens*dim,dim,1],"v_page_strides":[heads*page_tokens*dim,page_tokens*dim,dim,1],"bytes_per_token":2*heads*dim*4,"bytes_per_k_page":one,"bytes_per_v_page":one,"bytes_per_combined_page":2*one,"total_pool_bytes":pages*2*one,"alignment_bytes":4,"pool_create_entry_point":"hir_paged_kv_initialize","prefill_write_entry_point":"hir_paged_kv_prefill_write","append_entry_point":"hir_paged_kv_append","view_binding":"direct_int32_block_table_translation","reset_entry_point":"hir_paged_kv_reset","release_entry_point":"runtime_owned_pool_release","paged_attention_kernel_id":"cpu_attention_decode_paged_kv_fp32","contiguous_fallback_identity":"cpu_contiguous_kv_fp32_v1","runtime_no_layout_redecision":True,"runtime_no_kernel_redecision":True}
    session = PagedKVAttentionSession(contract, artifact_root=args.work_dir)
    k, v = values(heads*prompt*dim, 2), values(heads*prompt*dim, 3)
    session.prefill(k, v, prompt)
    session.append(values(heads*dim, 4), values(heads*dim, 5))
    q = values(heads*dim, 6)
    actual = session.decode(q)
    scale = 1/math.sqrt(dim); reference = []
    for head in range(heads):
        scores=[]
        for token in range(session.valid_tokens):
            block, offset = divmod(token, page_tokens); page = session.bt[block]
            base=((page*heads+head)*page_tokens+offset)*dim
            scores.append(sum(q[head*dim+x]*session.k[base+x] for x in range(dim))*scale)
        peak=max(scores); weights=[math.exp(x-peak) for x in scores]; total=sum(weights)
        for x in range(dim):
            reference.append(sum(weights[token]/total*session.v[((session.bt[token//page_tokens]*heads+head)*page_tokens+token%page_tokens)*dim+x] for token in range(session.valid_tokens)))
    errors=[abs(float(a)-b) for a,b in zip(actual,reference)]
    result={"schema_version":"paged_kv_validation.v1","artifact_sha256":digest,"max_absolute_error":max(errors),"mean_absolute_error":sum(errors)/len(errors),"valid_tokens":session.valid_tokens,"fragmentation":session.fragmentation(),"counters":session.trace(),"build_command":command}
    (args.work_dir/"validation_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__ == "__main__":
    main()
