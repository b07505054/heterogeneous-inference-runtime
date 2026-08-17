# D8 Multicell vLLM NCCL Attribution

Phase 4C generalizes the corrected Phase 4B attribution across a small workload matrix.

The parser uses authoritative measured request IDs, per-decode-step NVTX ranges, cross-rank NCCL wall-clock interval unions, and wall-clock NCCL/compute overlap. Summed per-rank NCCL GPU time is retained only as a diagnostic.

## Files

- `workload_matrix.json`
- `end_to_end_summary.json`
- `per_cell_attribution.json`
- `communication_prediction_validation.json`
- `overlap_generalization.json`
- `break_even_validation.json`

## Headline

Workloads: in32_out32_c1, in128_out32_c1, in32_out128_c1, in32_out32_c4, in32_out32_c8
Break-even explanations: 5/5
Overall zero-overlap MAE: 865.504474628274 us
Overall overlap-aware MAE: 881.416745748277 us

## Generalization Answers

- Collective call count changes with input length in this trace and changes with concurrency; output length mainly increases the number of decode steps.
- Overlap ratio changes across cells but remains small in absolute terms.
- Steady-state microbenchmark prediction is low-us for all-reduce, while all-gather and cold/capture/replay outliers remain visible.
- Zero-overlap was not materially worse here; its MAE was slightly lower than the measured-overlap correction for this matrix.
- The strongest future selector signal is compute savings versus exposed communication penalty; bytes alone are weak because observed messages stay in the same small-size bucket.

Nsight timings are profiler-perturbed; normal TP1/TP2 unprofiled runs remain the authoritative end-to-end latency source.
