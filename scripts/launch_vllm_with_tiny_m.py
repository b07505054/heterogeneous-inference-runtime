#!/usr/bin/env python3
"""E2E-8 launcher: registers the tiny-M GEMV OOT LogitsProcessor (via vLLM's
own PluggableLayer.register_oot extension point) BEFORE constructing the
engine, then runs vllm.entrypoints.openai.api_server exactly as `python -m
vllm.entrypoints.openai.api_server <args>` would. No vLLM source file is
imported differently, modified, or bypassed -- this only adds one import
before the normal entrypoint runs.

Whether the registered override actually changes behavior is controlled
entirely by VLLM_TINY_M_GEMV_ENABLE (default off) -- this launcher is safe
to use for the baseline runs too (it produces byte-identical output to a
stock `python -m vllm...` launch when the env var is unset).
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import perf_model.tiny_m_oot_logits_processor  # noqa: E402  (registration side effect, must run before engine construction)

if __name__ == "__main__":
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
