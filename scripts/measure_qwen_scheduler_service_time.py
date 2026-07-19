#!/usr/bin/env python3
"""Focused real-Qwen CPU latency points for the S2.5 service model."""
from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import torch

def median_run(fn,warmup=1,runs=4):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(runs):
        t=time.perf_counter();fn();values.append((time.perf_counter()-t)*1000)
    return {"median_ms":statistics.median(values),
            "p95_ms":sorted(values)[-1],"samples_ms":values}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForCausalLM
    torch.manual_seed(20260717);torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(
        a.model,local_files_only=True,dtype=torch.float32,
        attn_implementation="eager").eval()
    vocab=model.config.vocab_size
    rows=[]
    with torch.no_grad():
        for tokens in (4,8,16,32):
            ids=torch.arange(tokens).remainder(vocab).unsqueeze(0)
            row=median_run(lambda:model(ids,use_cache=False,logits_to_keep=1))
            rows.append({"kind":"prefill","prefill_tokens":tokens,
                         "decode_sequences":0,"core_budget":4,**row})
        for batch in (1,2,4):
            ids=torch.arange(8).remainder(vocab).repeat(batch,1)
            base=model(ids,use_cache=True,logits_to_keep=1)
            token=torch.ones((batch,1),dtype=torch.long)
            row=median_run(lambda:model(token,past_key_values=base.past_key_values,
                                        use_cache=True,logits_to_keep=1))
            rows.append({"kind":"decode","prefill_tokens":0,
                         "decode_sequences":batch,"core_budget":4,**row})
        for pt,dec in ((8,1),(16,2),(32,4)):
            pids=torch.arange(pt).remainder(vocab).unsqueeze(0)
            dids=torch.arange(8).remainder(vocab).repeat(dec,1)
            base=model(dids,use_cache=True,logits_to_keep=1)
            tok=torch.ones((dec,1),dtype=torch.long)
            def mixed():
                model(pids,use_cache=False,logits_to_keep=1)
                model(tok,past_key_values=base.past_key_values,
                      use_cache=True,logits_to_keep=1)
            row=median_run(mixed)
            rows.append({"kind":"mixed","prefill_tokens":pt,
                         "decode_sequences":dec,"core_budget":4,**row})
    # Interpretable nonnegative linear models, fitted only to these calibration points.
    pre=[r for r in rows if r["kind"]=="prefill"]
    dec=[r for r in rows if r["kind"]=="decode"]
    mix=[r for r in rows if r["kind"]=="mixed"]
    pc=np.polyfit([r["prefill_tokens"] for r in pre],
                  [r["median_ms"] for r in pre],1)
    dc=np.polyfit([r["decode_sequences"] for r in dec],
                  [r["median_ms"] for r in dec],1)
    X=np.array([[1,r["prefill_tokens"],r["decode_sequences"]] for r in mix])
    y=np.array([r["median_ms"] for r in mix])
    mc=np.linalg.lstsq(X,y,rcond=None)[0]
    model_payload={
        "version":"real_qwen_cpu_service_model_v1",
        "coefficients":{
            "prefill":{"intercept_ms":max(0,float(pc[1])),
                       "per_token_ms":max(0,float(pc[0]))},
            "decode":{"intercept_ms":max(0,float(dc[1])),
                      "per_sequence_ms":max(0,float(dc[0]))},
            "mixed":{"intercept_ms":max(0,float(mc[0])),
                     "prefill_token_ms":max(0,float(mc[1])),
                     "decode_sequence_ms":max(0,float(mc[2]))}},
        "provenance":"measured_cpu_qwen2.5_0.5b_fp32",
        "mixed_is_sequential_functional_step":True}
    residuals=[]
    for r in rows:
        c=model_payload["coefficients"][r["kind"]]
        if r["kind"]=="prefill":pred=c["intercept_ms"]+c["per_token_ms"]*r["prefill_tokens"]
        elif r["kind"]=="decode":pred=c["intercept_ms"]+c["per_sequence_ms"]*r["decode_sequences"]
        else:pred=c["intercept_ms"]+c["prefill_token_ms"]*r["prefill_tokens"]+c["decode_sequence_ms"]*r["decode_sequences"]
        residuals.append({"kind":r["kind"],"prefill_tokens":r["prefill_tokens"],
                          "decode_sequences":r["decode_sequences"],
                          "measured_ms":r["median_ms"],"predicted_ms":pred,
                          "absolute_error_ms":abs(pred-r["median_ms"])})
    errors=[r["absolute_error_ms"] for r in residuals]
    validation={"mean_absolute_error_ms":statistics.fmean(errors),
                "median_absolute_error_ms":statistics.median(errors),
                "p95_absolute_error_ms":sorted(errors)[-1],
                "maximum_absolute_error_ms":max(errors),
                "points":residuals,
                "limitation":"fit and residuals use focused calibration points; real held-out policy trace is separate"}
    (a.output_dir/"service_time_measurements.json").write_text(json.dumps({
        "execution_mode":"real_qwen","model":str(a.model),"dtype":"fp32",
        "torch_threads":4,"rows":rows},indent=2)+"\n")
    (a.output_dir/"service_time_model.json").write_text(json.dumps(model_payload,indent=2)+"\n")
    (a.output_dir/"service_time_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    (a.output_dir/"real_qwen_latency_validation.json").write_text(json.dumps({
        "measurement_to_selector_provenance":"real Qwen points -> interpretable coefficients -> mixed-step latency feature in request-level predictor",
        "model":model_payload,"validation":validation},indent=2)+"\n")
if __name__=="__main__":main()
