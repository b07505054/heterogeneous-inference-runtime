#!/usr/bin/env python3
"""Focused real-Qwen S1 proof with isolated replica queues/cache metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.attention_planner import (AttentionWorkload, emit_execution_plan,
                                          select_attention_plan,
                                          widen_context_domain)
from deployment.attention_runtime import (
    ExecutionPlanAttentionAdapter, register_transformers_attention_interface,
    set_active_attention_plan_adapter, set_active_attention_runtime)
from deployment.execution_plan.loader import load_execution_plan
from deployment.serving_execution import (
    FunctionalClusterProfile, PlanOnlyServingRuntime, ServingDistributedCompiler,
    ServingRequest, deserialize_serving_plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20260717)
    torch.set_num_threads(4)
    register_transformers_attention_interface()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed_id = int(tokenizer("Compiler serving proof.", return_tensors="pt").input_ids[0, 0])
    common_prefix = tuple([seed_id] * 32)
    requests = [
        ServingRequest("qwen-request-0", common_prefix + (100, 101), 1, 0.0),
        ServingRequest("qwen-request-1", common_prefix + (102, 103), 1, 0.1),
        ServingRequest("qwen-request-2", tuple([seed_id + 1] * 32) + (104,), 1, 0.2),
        ServingRequest("qwen-request-3", common_prefix + (105, 106), 1, 0.3),
    ]
    cluster = FunctionalClusterProfile.local(2, total_logical_cores=8)
    runtime = PlanOnlyServingRuntime(cluster, cache_mode="metadata_only",
                                     block_size=16)
    compiler = ServingDistributedCompiler(cluster)
    original = model.config._attn_implementation
    serving_roundtrips, nested, all_worker_events = [], [], []

    try:
        for request in requests:
            prompt = torch.tensor([request.token_ids], dtype=torch.long)
            common = {
                "batch": 1, "query_heads": model.config.num_attention_heads,
                "kv_heads": model.config.num_key_value_heads,
                "head_dim": model.config.hidden_size // model.config.num_attention_heads,
                "available_logical_workers": 4,
            }
            prefill, _ = select_attention_plan(AttentionWorkload(
                phase="prefill", query_len=len(request.token_ids),
                context_len=len(request.token_ids), **common))
            decode, _ = select_attention_plan(AttentionWorkload(
                phase="decode", query_len=1, context_len=len(request.token_ids) + 1,
                **common))
            decode = widen_context_domain(decode, len(request.token_ids) + 1,
                                          len(request.token_ids) + 1)
            operator_id = f"operator-{request.request_id}"
            operator_payload = emit_execution_plan(
                plan_id=operator_id, model_id=str(args.model),
                prompt_tokens=len(request.token_ids), generated_tokens=1,
                prefill=prefill, decode=decode)
            operator_path = args.output_dir / f"{operator_id}.json"
            operator_path.write_text(json.dumps(operator_payload, indent=2) + "\n")
            loaded_operator = load_execution_plan(operator_path)
            selected = compiler.plan(request, runtime.replicas,
                                     policy="prefix_queue_cost",
                                     operator_plan_id=operator_id)
            loaded_serving = deserialize_serving_plan(selected.serialize(), cluster)
            serving_roundtrips.append({
                "request_id": request.request_id,
                "selected_replica_id": selected.selected_replica_id,
                "serialized_replica_id": json.loads(selected.serialize())[
                    "selected_replica_id"],
                "deserialized_replica_id": loaded_serving.selected_replica_id,
                "exact": selected.to_dict() == loaded_serving.to_dict(),
            })

            def execute(replica, req, serving_plan):
                adapter = ExecutionPlanAttentionAdapter(loaded_operator)
                oproj = []
                handles = []
                for layer in model.model.layers:
                    def capture(_module, values):
                        oproj.append((values[0].data_ptr(),
                                      float(values[0].double().sum())))
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
                started = time.perf_counter()
                model.config._attn_implementation = "compiler_cpu_attention"
                set_active_attention_plan_adapter(adapter)
                with torch.no_grad():
                    out = model(prompt, use_cache=True, logits_to_keep=1)
                    token = int(out.logits[:, -1].argmax(-1))
                measured = (time.perf_counter() - started) * 1000
                set_active_attention_plan_adapter(None)
                for h in handles:
                    h.remove()
                provenance = list(adapter.provenance)
                for record, (ptr, checksum) in zip(provenance, oproj):
                    record["serving_plan_id"] = serving_plan.plan_id
                    record["replica_id"] = replica.profile.replica_id
                    record["operator_plan_id"] = operator_id
                    record["output_entered_o_proj"] = (
                        record["returned_tensor_data_ptr"] == ptr and
                        record["returned_tensor_sum"] == checksum)
                result = {
                    "measured_execution_ms": measured,
                    "measured_first_token_ms": measured,
                    "generated_token_ids": [token],
                    "operator_provenance": provenance,
                    "operator_fallback_count": adapter.fallback_count,
                    "operator_candidate_mismatch_count": adapter.mismatch_count,
                }
                adapter.close()
                return result

            event = runtime.execute(request, loaded_serving, execute)
            nested.append(event)
            all_worker_events.extend(event["operator_provenance"])
    finally:
        model.config._attn_implementation = original
        set_active_attention_plan_adapter(None)
        set_active_attention_runtime(None)
    payload = {
        "artifact_type": "real_qwen_multi_replica_serving_s1",
        "truth_boundary": (
            "Real Qwen CPU model-forward on plan-selected logical CPU replicas; "
            "independent metadata-only prefix caches; immutable weights shared; "
            "not vLLM serving and no tensor KV reuse."),
        "model": str(args.model), "replica_count": 2, "request_count": 4,
        "shared_immutable_model_weights": True,
        "cache_mode": "metadata_only",
        "serving_plan_roundtrips": serving_roundtrips,
        "requests": nested,
        "serving_counters": runtime.counters(),
        "operator_invocations": len(all_worker_events),
        "outputs_entered_o_proj": sum(
            x.get("output_entered_o_proj", False) for x in all_worker_events),
        "operator_fallback_count": sum(
            x.get("fallback", False) for x in all_worker_events),
        "operator_candidate_mismatch_count": sum(
            x.get("selected_candidate") != x.get("executed_candidate")
            for x in all_worker_events),
        "operator_repartition_count": sum(
            x.get("runtime_repartition_count", 0) for x in all_worker_events),
        "request_to_replica_exact": all(
            x["planned_replica_id"] == x["executed_replica_id"] for x in nested),
        "all_outputs_entered_o_proj": all(
            x.get("output_entered_o_proj", False) for x in all_worker_events),
        "replica_cache_states": {
            rid: replica.cache.snapshot() for rid, replica in runtime.replicas.items()},
    }
    (args.output_dir / "qwen_multi_replica_integration.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "request_to_replica_provenance.json").write_text(
        json.dumps(nested, indent=2) + "\n")
    (args.output_dir / "request_to_operator_provenance.json").write_text(
        json.dumps(all_worker_events, indent=2) + "\n")


if __name__ == "__main__":
    main()
