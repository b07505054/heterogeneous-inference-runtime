#!/usr/bin/env python3
"""Focused real-Qwen S2 continuous batching and chunked-prefill proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.attention_planner import AttentionWorkload, emit_execution_plan, select_attention_plan
from deployment.attention_runtime import (
    ExecutionPlanAttentionAdapter, register_transformers_attention_interface,
    set_active_attention_plan_adapter, set_active_attention_runtime)
from deployment.execution_plan.loader import load_execution_plan
from deployment.serving_execution import (
    FunctionalClusterProfile, PlanOnlyServingRuntime, ServingDistributedCompiler,
    ServingRequest, deserialize_serving_plan)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState,
    SchedulerCompiler, SchedulerProfile, deserialize_schedule_plan)
from deployment.scheduler_calibration import SchedulerSelectorV1, make_objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scheduler-policy", default="selector_v0",
                    choices=("decode_first","prefill_first","chunked_balanced",
                             "slo_aware","selector_v0","selector_v1"))
    ap.add_argument("--output-name", default="qwen_s2_integration.json")
    ap.add_argument("--natural-multiworker-focus", action="store_true",
                    help="Use a legal q=32 prefill; compiler selection remains unforced.")
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20260717); torch.set_num_threads(4)
    register_transformers_attention_interface()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = int(tokenizer("S2 proof", return_tensors="pt").input_ids[0, 0])
    common = [seed] * 16
    prompts = [
        common + [100,101], common + [102,103,104,105],
        [seed+1]*12, [seed+2]*32, [seed+3]*24, common + [106]*12,
    ]
    arrivals = [0.0, .05, .1, .15, .2, .25]
    if args.natural_multiworker_focus:
        prompts = [[seed+2]*32]
        arrivals = [0.0]
    cluster = FunctionalClusterProfile.local(2, total_logical_cores=8)
    serving_runtime = PlanOnlyServingRuntime(cluster, cache_mode="metadata_only")
    serving_compiler = ServingDistributedCompiler(cluster)
    profile = SchedulerProfile(max_num_seqs=4,
                               max_num_batched_tokens=32 if args.natural_multiworker_focus else 12,
                               max_prefill_chunk_tokens=32 if args.natural_multiworker_focus else 8,
                               balanced_decode_reservation=2)
    states = {rid: ReplicaSchedulerState(rid, profile)
              for rid in serving_runtime.replicas}
    model_state = {}
    serving_records = []
    for i, tokens in enumerate(prompts):
        req = ServingRequest(f"qwen-s2-{i}", tuple(tokens), 2, arrivals[i])
        selected = serving_compiler.plan(
            req, serving_runtime.replicas, policy="round_robin")
        loaded = deserialize_serving_plan(selected.serialize(), cluster)
        serving_records.append({
            "request_id": req.request_id, "serving_plan_id": loaded.plan_id,
            "selected_replica_id": selected.selected_replica_id,
            "deserialized_replica_id": loaded.selected_replica_id,
            "exact": selected.selected_replica_id == loaded.selected_replica_id})
        execution = RequestExecutionState(
            req.request_id, loaded.plan_id, loaded.selected_replica_id,
            req.arrival_time_ms, len(tokens), 0, req.expected_output_tokens)
        states[loaded.selected_replica_id].ingest(execution)
        model_state[req.request_id] = {
            "tokens": torch.tensor([tokens], dtype=torch.long), "cache": None,
            "context": 0, "generated": [], "pending_logits": None}

    schedule_runtime = {rid: PlanOnlySchedulerRuntime() for rid in states}
    scheduler_compiler = {rid: SchedulerCompiler() for rid in states}
    epoch_policy = {}
    if args.scheduler_policy == "selector_v1":
        for rid,state in states.items():
            epoch_policy[rid] = SchedulerSelectorV1(
                make_objective(), prediction_mode="full_trace").select(state).policy_id
    elif args.scheduler_policy in ("decode_first","prefill_first",
                                   "chunked_balanced","slo_aware"):
        epoch_policy = {rid:args.scheduler_policy for rid in states}
    else:
        epoch_policy = {rid:None for rid in states}
    all_operator_records = []
    operator_fallback = operator_mismatch = 0
    original = model.config._attn_implementation

    def execute_item(request, item, schedule):
        nonlocal operator_fallback, operator_mismatch
        local = model_state[request.request_id]
        if item.phase == "decode" and local["pending_logits"] is not None:
            logits = local["pending_logits"]; local["pending_logits"] = None
            token = int(logits.argmax(-1))
            local["generated"].append(token)
            return {"model_invocation_id": f"consume-prefill-logits-{request.request_id}",
                    "generated_token_ids": [token], "operator_provenance": []}
        if item.phase == "prefill":
            current = local["tokens"][:, item.token_start:item.token_start+item.token_count]
            phase = "prefill"; qlen = item.token_count
        else:
            current = torch.tensor([[local["generated"][-1]]], dtype=torch.long)
            phase = "decode"; qlen = 1
        context = local["context"] + qlen
        workload = AttentionWorkload(
            phase=phase, batch=1, query_len=qlen, context_len=context,
            query_heads=model.config.num_attention_heads,
            kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            available_logical_workers=4)
        decision, _ = select_attention_plan(workload)
        operator_id = f"operator-{request.request_id}-step-{schedule.step_id}"
        payload = emit_execution_plan(
            plan_id=operator_id, model_id=str(args.model),
            prompt_tokens=max(1, qlen), generated_tokens=1,
            prefill=decision if phase == "prefill" else select_attention_plan(
                AttentionWorkload(phase="prefill", batch=1, query_len=2,
                    context_len=2, query_heads=model.config.num_attention_heads,
                    kv_heads=model.config.num_key_value_heads,
                    head_dim=model.config.hidden_size//model.config.num_attention_heads,
                    available_logical_workers=4))[0],
            decode=decision if phase == "decode" else select_attention_plan(
                AttentionWorkload(phase="decode", batch=1, query_len=1,
                    context_len=context+1, query_heads=model.config.num_attention_heads,
                    kv_heads=model.config.num_key_value_heads,
                    head_dim=model.config.hidden_size//model.config.num_attention_heads,
                    available_logical_workers=4))[0])
        path = args.output_dir / f"{operator_id}.json"
        path.write_text(json.dumps(payload, indent=2)+"\n")
        adapter = ExecutionPlanAttentionAdapter(load_execution_plan(path))
        hooks, oproj = [], []
        for layer in model.model.layers:
            def capture(_module, values):
                oproj.append((values[0].data_ptr(), float(values[0].double().sum())))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
        model.config._attn_implementation = "compiler_cpu_attention"
        set_active_attention_plan_adapter(adapter)
        with torch.no_grad():
            out = model(current, past_key_values=local["cache"], use_cache=True,
                        logits_to_keep=1)
        set_active_attention_plan_adapter(None)
        for hook in hooks: hook.remove()
        provenance = list(adapter.provenance)
        for record, (ptr, checksum) in zip(provenance, oproj):
            record.update({
                "request_id": request.request_id,
                "serving_plan_id": request.serving_plan_id,
                "replica_id": request.replica_id,
                "schedule_plan_id": schedule.plan_id,
                "scheduler_step_id": schedule.step_id,
                "scheduled_phase": item.phase,
                "scheduled_token_range": [item.token_start,
                                          item.token_start+item.token_count],
                "operator_plan_id": operator_id,
                "output_entered_o_proj":
                    record["returned_tensor_data_ptr"] == ptr and
                    record["returned_tensor_sum"] == checksum})
        operator_fallback += adapter.fallback_count
        operator_mismatch += adapter.mismatch_count
        adapter.close()
        local["cache"] = out.past_key_values
        local["context"] = context
        if item.phase == "prefill" and \
                item.token_start + item.token_count == request.uncached_prompt_tokens:
            local["pending_logits"] = out.logits[:, -1].float()
        generated = []
        if item.phase == "decode":
            token = int(out.logits[:, -1].argmax(-1))
            local["generated"].append(token); generated = [token]
        all_operator_records.extend(provenance)
        return {"model_invocation_id": f"qwen-{request.request_id}-{schedule.step_id}",
                "operator_plan_id": operator_id,
                "operator_provenance": provenance,
                "generated_token_ids": generated}

    try:
        # Interleave replica scheduler iterations; active membership changes as
        # future arrivals become ready and completed requests leave.
        real_run_started = time.perf_counter()
        while any(s.unfinished() for s in states.values()):
            progressed = False
            for rid, state in states.items():
                if not state.unfinished(): continue
                ready = state.ready()
                if not ready:
                    future = [r.arrival_time_ms for r in state.requests.values()
                              if r.phase == "WAITING"]
                    if future: state.clock_ms = min(future); ready = state.ready()
                if not ready: continue
                selected = scheduler_compiler[rid].compile(
                    state, forced_policy=epoch_policy[rid])
                loaded = deserialize_schedule_plan(selected.serialize(), state)
                schedule_runtime[rid].execute(state, loaded, execute_item)
                progressed = True
            if not progressed: raise RuntimeError("real Qwen scheduler made no progress")
    finally:
        model.config._attn_implementation = original
        set_active_attention_plan_adapter(None)
        set_active_attention_runtime(None)

    def eager_chunk_generate(tokens, chunk_size):
        cache, last_logits, chunks = None, None, []
        model.config._attn_implementation = "eager"
        with torch.no_grad():
            for begin in range(0, len(tokens), chunk_size):
                end = min(len(tokens), begin + chunk_size)
                out = model(torch.tensor([tokens[begin:end]], dtype=torch.long),
                            past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
                cache, last_logits = out.past_key_values, out.logits[:, -1].float()
                chunks.append([begin, end])
            generated, logits = [], []
            for step in range(2):
                if step:
                    out = model(torch.tensor([[generated[-1]]]), past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
                    cache, last_logits = out.past_key_values, out.logits[:, -1].float()
                logits.append(last_logits.cpu())
                generated.append(int(last_logits.argmax(-1)))
        return generated, logits, chunks

    causal_index = 0 if args.natural_multiworker_focus else 3
    causal_prompt = prompts[causal_index]
    whole_tokens, whole_logits, whole_chunks = eager_chunk_generate(causal_prompt, len(causal_prompt))
    chunk_tokens, chunk_logits, causal_chunks = eager_chunk_generate(causal_prompt, 8)
    causal = {
        "request_prompt_tokens": len(causal_prompt),
        "whole_prefill_chunks": whole_chunks,
        "chunked_prefill_chunks": causal_chunks,
        "whole_generated_tokens": whole_tokens,
        "chunked_generated_tokens": chunk_tokens,
        "tokens_equal": whole_tokens == chunk_tokens,
        "max_logit_abs_diff_by_step": [
            float((a-b).abs().max()) for a,b in zip(whole_logits, chunk_logits)],
        "coverage_equal": whole_chunks[0] == [0, len(causal_prompt)] and
            causal_chunks[0][0] == 0 and causal_chunks[-1][1] == len(causal_prompt) and
            all(a[1] == b[0] for a,b in zip(causal_chunks, causal_chunks[1:])),
        "truth_boundary": "real Qwen contiguous Transformers cache, not paged KV",
    }
    (args.output_dir/"qwen_chunking_causal_test.json").write_text(
        json.dumps(causal, indent=2)+"\n")

    schedule_events = [e for rt in schedule_runtime.values() for e in rt.events]
    real_run_ms = (time.perf_counter()-real_run_started)*1000
    payload = {
        "artifact_type": "real_qwen_serving_distributed_s2",
        "execution_mode": "real_qwen",
        "truth_boundary": (
            "Real Hugging Face Qwen CPU model-forward with metadata-mode chunk "
            "progression and contiguous per-request Transformers cache; not "
            "vLLM, PagedAttention, or production KV block persistence."),
        "model": str(args.model), "replicas": len(states), "requests": len(prompts),
        "scheduler_policy_argument":args.scheduler_policy,
        "effective_epoch_policy":epoch_policy,
        "measured_total_model_forward_ms":real_run_ms,
        "serving_plans": serving_records,
        "scheduler_steps": len(schedule_events),
        "schedule_events": schedule_events,
        "generated_outputs": {rid: x["generated"] for rid,x in model_state.items()},
        "prefill_chunks": {rid: s.prefill_chunks for state in states.values()
                           for rid,s in state.requests.items()},
        "operator_attention_invocations": len(all_operator_records),
        "attention_outputs_entered_o_proj": sum(
            x["output_entered_o_proj"] for x in all_operator_records),
        "all_attention_outputs_entered_o_proj": all(
            x["output_entered_o_proj"] for x in all_operator_records),
        "serving_counters": serving_runtime.counters(),
        "scheduler_counters": {
            rid: rt.counters() for rid,rt in schedule_runtime.items()},
        "operator_fallback_count": operator_fallback,
        "operator_candidate_mismatch_count": operator_mismatch,
        "operator_repartition_count": sum(
            x.get("runtime_repartition_count", 0) for x in all_operator_records),
        "all_requests_finished": all(
            r.finished for state in states.values() for r in state.requests.values()),
        "cache_mode": "metadata_only",
        "real_qwen_chunking_causal_test": causal,
    }
    (args.output_dir/args.output_name).write_text(
        json.dumps(payload, indent=2)+"\n")
    (args.output_dir/"request_scheduler_operator_provenance.json").write_text(
        json.dumps(all_operator_records, indent=2)+"\n")


if __name__ == "__main__":
    main()
