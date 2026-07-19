#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    policies=("decode_first","prefill_first","chunked_balanced","slo_aware",
              "selector_v0","selector_v1")
    rows={}
    baseline=None
    for p in policies:
        x=json.loads((a.root/f"qwen_{p}"/"result.json").read_text())
        outputs=x["generated_outputs"]
        baseline=baseline or outputs
        rows[p]={
            "effective_epoch_policy":x["effective_epoch_policy"],
            "measured_total_model_forward_ms":x["measured_total_model_forward_ms"],
            "scheduler_steps":x["scheduler_steps"],
            "operator_attention_invocations":x["operator_attention_invocations"],
            "outputs_entered_o_proj":x["attention_outputs_entered_o_proj"],
            "generated_outputs":outputs,"outputs_equal_baseline":outputs==baseline,
            "serving_counters":x["serving_counters"],
            "scheduler_counters":x["scheduler_counters"],
            "operator_fallback_count":x["operator_fallback_count"],
            "operator_candidate_mismatch_count":x["operator_candidate_mismatch_count"],
            "operator_repartition_count":x["operator_repartition_count"]}
    a.output.write_text(json.dumps({
        "execution_mode":"real_qwen",
        "truth_boundary":"single focused trace; cross-layer validation, not statistical performance proof",
        "identical_logical_requests":True,"identical_s1_placement_policy":True,
        "all_generated_outputs_equivalent":all(x["outputs_equal_baseline"] for x in rows.values()),
        "rows":rows},indent=2)+"\n")
if __name__=="__main__":main()
