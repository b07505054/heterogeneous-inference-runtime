# 7B legal-operating-range probe: finding

All 6 probed configurations (`max_model_len` from 2048 to 32768 -- the
model's real `max_position_embeddings` ceiling -- and `max_num_seqs` from
4 to 16) started successfully on **both** TP1 and TP2. See
`legal_range_probe_results.json` for the full per-config record
(`started`, peak per-GPU memory, log tail) and `logs/probe_tp*_*.log` for
raw server output.

**No startup-level memory-capacity crossover exists in this range** (i.e.
no configuration was found where TP1 is illegal but TP2 is legal). The
mechanism: vLLM's paged-attention KV cache manager profiles available
memory after weight loading and adaptively sizes the KV-cache block pool
to whatever is available within `gpu_memory_utilization`, rather than
pre-reserving a fixed worst-case allocation for `max_num_seqs *
max_model_len`. Peak GPU-0 memory usage during TP1 actually *decreases*
slightly as `max_model_len` grows (21096 MiB at 2048 -> 18060 MiB at
32768/16), which is the direct evidence for this adaptive-sizing
behavior, not memory pressure.

This does not rule out a *runtime* capacity effect (e.g. real concurrent
requests exhausting the profiled KV-cache block pool and causing
preemption/recomputation, which would show up as a latency/throughput
regression rather than a hard startup failure) -- that possibility is
covered by the main calibration sweep's concurrency axis, not by this
probe, and is reported honestly as a performance effect, not a legality
boundary, if observed.

**Consequence for D5 scope**: the model-size axis (0.5B vs 7B, at fixed
workload shape) remains the primary hypothesis under test for a genuine
TP1/TP2 crossover -- more compute per device on a larger model may still
make TP2's fixed NCCL overhead pay for itself, even though no hard memory
boundary forces the choice. Proceeding to the 7B calibration sweep to
measure this directly.
