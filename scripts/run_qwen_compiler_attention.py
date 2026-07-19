#!/usr/bin/env python3
"""Real Qwen forward/generation proof through compiler-selected attention."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.attention_runtime import (  # noqa: E402
    ExecutionPlanAttentionAdapter, register_transformers_attention_interface,
    set_active_attention_plan_adapter, set_active_attention_runtime,
)
from deployment.attention_planner import (  # noqa: E402
    AttentionWorkload, emit_execution_plan, select_attention_plan,
    force_test_attention_plan, widen_context_domain,
)
from deployment.execution_plan.loader import load_execution_plan  # noqa: E402


def greedy(model, input_ids, steps, implementation, runtime=None):
    model.config._attn_implementation = implementation
    set_active_attention_runtime(runtime)
    ids = input_ids
    cache = None
    tokens, logits, step_ms = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            step_started = time.perf_counter()
            current = ids if step == 0 else ids[:, -1:]
            out = model(current, past_key_values=cache, use_cache=True,
                        logits_to_keep=1)
            cache = out.past_key_values
            last = out.logits[:, -1].float()
            token = last.argmax(-1, keepdim=True)
            tokens.append(int(token))
            logits.append(last.cpu())
            ids = torch.cat((ids, token), dim=1)
            step_ms.append((time.perf_counter() - step_started) * 1000)
    return tokens, logits, (time.perf_counter() - started) * 1000, step_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="Compiler attention proof.")
    ap.add_argument("--plan-output", type=Path)
    ap.add_argument("--selector-trace-output", type=Path)
    ap.add_argument("--provenance-output", type=Path)
    ap.add_argument("--prompt-token-count", type=int)
    ap.add_argument("--forced-test-algorithm",
                    choices=("dense_materialized", "fused_tiled_online_softmax"))
    ap.add_argument("--forced-test-implementation",
                    choices=("torch_tiled_online_softmax_exact_v1",
                             "native_scalar", "native_avx2"))
    ap.add_argument("--forced-test-workers", type=int, choices=(1,2,4,8), default=1)
    ap.add_argument("--perturb-worker-id", type=int)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20260717)
    torch.set_num_threads(4)
    register_transformers_attention_interface()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    if args.prompt_token_count:
        if args.prompt_token_count < 2:
            raise ValueError("--prompt-token-count must be at least 2")
        input_ids = input_ids[:, :1].repeat(1, args.prompt_token_count)
    baseline_tokens, baseline_logits, baseline_ms, baseline_steps = greedy(
        model, input_ids, args.steps, "eager")

    prompt_tokens = int(input_ids.shape[1])
    common = {
        "batch": int(input_ids.shape[0]), "query_heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "available_logical_workers": min(8, int(torch.get_num_threads()) * 2),
    }
    prefill_workload = AttentionWorkload(
        phase="prefill", query_len=prompt_tokens, context_len=prompt_tokens, **common)
    prefill_plan, prefill_trace = select_attention_plan(prefill_workload)
    first_decode_context = prompt_tokens + 1
    decode_workload = AttentionWorkload(
        phase="decode", query_len=1, context_len=first_decode_context, **common)
    decode_plan, decode_trace = select_attention_plan(decode_workload)
    if args.forced_test_algorithm:
        if args.forced_test_algorithm == "fused_tiled_online_softmax":
            prefill_plan, prefill_trace = force_test_attention_plan(
                prefill_workload, algorithm=args.forced_test_algorithm,
                strategy=("serial" if args.forced_test_workers==1 else "split_head"),
                workers=args.forced_test_workers, query_tile=4, key_tile=32,
                implementation=args.forced_test_implementation)
            decode_plan, decode_trace = force_test_attention_plan(
                decode_workload, algorithm=args.forced_test_algorithm,
                strategy=("serial" if args.forced_test_workers==1 else "split_head"),
                workers=args.forced_test_workers, query_tile=1, key_tile=32,
                implementation=args.forced_test_implementation)
        else:
            prefill_plan, prefill_trace = force_test_attention_plan(
                prefill_workload, algorithm=args.forced_test_algorithm,
                strategy="serial", workers=1, query_tile=0, key_tile=0)
            decode_plan, decode_trace = force_test_attention_plan(
                decode_workload, algorithm=args.forced_test_algorithm,
                strategy="serial", workers=1, query_tile=0, key_tile=0)
    decode_plan = widen_context_domain(
        decode_plan, first_decode_context,
        max(first_decode_context, prompt_tokens + args.steps - 1))
    plan_payload = emit_execution_plan(
        plan_id="qwen-compiler-selected-attention-level5-v1",
        model_id=str(args.model), prompt_tokens=prompt_tokens,
        generated_tokens=args.steps, prefill=prefill_plan, decode=decode_plan)
    plan_path = args.plan_output or args.output.with_name(
        args.output.stem + "_execution_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan_payload, indent=2) + "\n")
    # Mandatory persistence boundary: runtime receives only the normal loader's
    # deserialized plan, never the in-memory selector result.
    loaded_plan = load_execution_plan(plan_path)
    roundtrip_table = loaded_plan.global_decisions.attention_execution
    if roundtrip_table != plan_payload["global_decisions"]["attention_execution"]:
        raise RuntimeError("attention decision changed during ExecutionPlan round-trip")
    original = model.config._attn_implementation
    def compiler_greedy(perturb=0.0):
        adapter = ExecutionPlanAttentionAdapter(
            loaded_plan, perturbation=perturb,
            perturb_worker_id=(args.perturb_worker_id if perturb else None))
        oproj_inputs = []
        handles = []
        for layer in model.model.layers:
            layer_index = layer.self_attn.layer_idx
            def capture(_module, values, layer_index=layer_index):
                tensor = values[0]
                oproj_inputs.append({
                    "layer_index": layer_index,
                    "data_ptr": tensor.data_ptr(),
                    "sum": float(tensor.double().sum()),
                })
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
        ids, cache, tokens, logits, step_ms = input_ids, None, [], [], []
        started = time.perf_counter()
        model.config._attn_implementation = "compiler_cpu_attention"
        set_active_attention_plan_adapter(adapter)
        with torch.no_grad():
            for step in range(args.steps):
                step_started = time.perf_counter()
                current = ids if step == 0 else ids[:, -1:]
                out = model(current, past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
                cache = out.past_key_values
                last = out.logits[:, -1].float()
                token = last.argmax(-1, keepdim=True)
                tokens.append(int(token))
                logits.append(last.cpu())
                ids = torch.cat((ids, token), 1)
                step_ms.append((time.perf_counter() - step_started) * 1000)
        set_active_attention_plan_adapter(None)
        for handle in handles:
            handle.remove()
        provenance = list(adapter.provenance)
        if len(provenance) != len(oproj_inputs):
            raise RuntimeError("attention/o_proj invocation count differs")
        for record, oproj in zip(provenance, oproj_inputs):
            record["o_proj_input_data_ptr"] = oproj["data_ptr"]
            record["o_proj_input_sum"] = oproj["sum"]
            record["output_entered_o_proj"] = (
                record["returned_tensor_data_ptr"] == oproj["data_ptr"]
                and record["returned_tensor_sum"] == oproj["sum"])
        if not all(x["output_entered_o_proj"] for x in provenance):
            raise RuntimeError("compiler attention output did not enter o_proj")
        fallback_count, mismatch_count = adapter.fallback_count, adapter.mismatch_count
        adapter.close()
        return (tokens, logits, (time.perf_counter() - started) * 1000,
                step_ms, provenance, fallback_count, mismatch_count)
    try:
        (compiler_tokens, compiler_logits, compiler_ms, compiler_steps,
         clean_provenance, fallback_count, mismatch_count) = compiler_greedy()
        clean_prefill_calls = sum(x["phase"] == "prefill" for x in clean_provenance)
        clean_decode_calls = sum(x["phase"] == "decode" for x in clean_provenance)
        (perturbed_tokens, perturbed_logits, perturbed_ms, _,
         perturbed_provenance, perturbed_fallbacks,
         perturbed_mismatches) = compiler_greedy(5.0)
    finally:
        model.config._attn_implementation = original
        set_active_attention_plan_adapter(None)
        set_active_attention_runtime(None)
    logit_diffs = [
        float((a - b).abs().max()) for a, b in zip(baseline_logits, compiler_logits)
    ]
    perturb_diffs = [
        float((a - b).abs().max()) for a, b in zip(compiler_logits, perturbed_logits)
    ]
    payload = {
        "artifact_type": "qwen_real_model_compiler_attention_proof",
        "integration_level": 5,
        "truth_boundary": (
            "HuggingFace Qwen real model-forward path; generated tokens depend "
            "on registered compiler attention; not a vLLM serving request"),
        "model": str(args.model), "prompt": args.prompt,
        "prompt_token_count": int(input_ids.shape[1]), "steps": args.steps,
        "baseline_tokens": baseline_tokens,
        "compiler_tokens": compiler_tokens,
        "tokens_equal": baseline_tokens == compiler_tokens,
        "max_logit_abs_diff_by_step": logit_diffs,
        "compiler_attention_prefill_invocations": clean_prefill_calls,
        "compiler_attention_decode_invocations": clean_decode_calls,
        "selected_equals_executed": mismatch_count == 0,
        "fallback_count": fallback_count,
        "candidate_mismatch_count": mismatch_count,
        "manual_plan_invocation_count": 0,
        "outputs_entered_o_proj_count": sum(
            x["output_entered_o_proj"] for x in clean_provenance),
        "compiler_attention_total_ms": sum(
            x["attention_total_ms"] for x in clean_provenance),
        "selection_mode": prefill_plan["selection_mode"],
        "prefill_selected_algorithm": prefill_plan["algorithm"],
        "decode_selected_algorithm": decode_plan["algorithm"],
        "prefill_selected_implementation": prefill_plan["implementation"],
        "decode_selected_implementation": decode_plan["implementation"],
        "execution_plan_path": str(plan_path),
        "execution_plan_id": loaded_plan.plan_id,
        "prefill_selected_candidate": prefill_plan["native_kernel_id"],
        "decode_selected_candidate": decode_plan["native_kernel_id"],
        "selector_invocation_count": 2,
        "execution_plan_roundtrip_equal": True,
        "perturbation": 5.0,
        "perturbed_tokens": perturbed_tokens,
        "perturbed_tokens_differ": perturbed_tokens != compiler_tokens,
        "perturbed_logit_max_abs_diff_by_step": perturb_diffs,
        "baseline_generation_ms": baseline_ms,
        "compiler_generation_ms": compiler_ms,
        "perturbed_generation_ms": perturbed_ms,
        "baseline_ttft_ms": baseline_steps[0],
        "compiler_ttft_ms": compiler_steps[0],
        "baseline_tpot_median_ms": (
            statistics.median(baseline_steps[1:]) if len(baseline_steps) > 1 else None),
        "compiler_tpot_median_ms": (
            statistics.median(compiler_steps[1:]) if len(compiler_steps) > 1 else None),
        "baseline_tpot_p95_ms": (
            float(np.percentile(baseline_steps[1:], 95))
            if len(baseline_steps) > 1 else None),
        "compiler_tpot_p95_ms": (
            float(np.percentile(compiler_steps[1:], 95))
            if len(compiler_steps) > 1 else None),
        "perturbed_fallback_count": perturbed_fallbacks,
        "perturbed_candidate_mismatch_count": perturbed_mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.selector_trace_output:
        args.selector_trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.selector_trace_output.write_text(json.dumps({
            "prefill": prefill_trace, "decode": decode_trace}, indent=2) + "\n")
    if args.provenance_output:
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(json.dumps({
            "clean": clean_provenance, "perturbed": perturbed_provenance},
            indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
