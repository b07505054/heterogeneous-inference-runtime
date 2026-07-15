#!/usr/bin/env python3
"""Execute every compiler-exported CPU attention contract exactly once."""
from array import array
import argparse, json
from pathlib import Path

from deployment.execution_plan.attention_cpu_adapter import PersistentAttentionRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    results = []
    for function in plan["function_plans"]:
        for decision in function.get("per_op_decisions", []):
            contract = decision.get("attention_execution")
            if not contract:
                continue
            b, h, ql, cl, d = (contract[k] for k in (
                "batch", "num_query_heads", "query_length", "context_length", "head_dim"))
            q = array("f", (float((i % 17) - 8) / 17 for i in range(b*h*ql*d)))
            k = array("f", (float((i % 19) - 9) / 19 for i in range(b*h*cl*d)))
            v = array("f", (float((i % 23) - 11) / 23 for i in range(b*h*cl*d)))
            runner = PersistentAttentionRunner(contract, artifact_root=args.artifact_root)
            output = runner.invoke(q, k, v)
            results.append({"function": function["function_name"], "checksum": float(sum(output)),
                            **runner.trace()})
    if len(results) != 2:
        raise SystemExit("expected exactly two compiler-exported attention contracts")
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
