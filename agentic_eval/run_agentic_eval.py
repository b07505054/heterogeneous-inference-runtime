from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_agent import DEFAULT_TASK, build_default_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MobileNetV2 backend-selection agentic eval.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    run = build_default_agent(args.repo_root).run(DEFAULT_TASK)
    payload = {
        "task": run.task,
        "answer": run.answer,
        "trace": [
            {
                "tool": call.tool,
                "args": call.args,
                "ok": call.ok,
                "error": call.error,
            }
            for call in run.trace
        ],
        "eval": run.eval,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

