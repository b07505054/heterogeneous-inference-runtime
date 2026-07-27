"""Kernel-level and end-to-end performance-model input/prediction schema for
compiler-guided vLLM serving.

Scope (see docs/PERF_MODEL_SLICE_1.md-equivalent report printed by
scripts/run_perf_model_experiment.py): analytical operation-count and memory
estimates, combined with real vLLM runtime configuration capture
(/server_info) and real request-level + server-side metrics (/metrics),
producing normalized calibration rows and predicted-vs-measured error
attribution. No learned model. No vLLM modification.
"""
