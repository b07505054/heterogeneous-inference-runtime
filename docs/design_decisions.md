# Design Decisions

## Keep Runtime Evidence Modular

The repository uses small modules for each runtime concern instead of one monolithic application. CV benchmarking, async video inference, LLM scheduling simulation, native C++ experiments, CUDA kernels, TVM experiments, and agentic evaluation each have separate entry points.

Tradeoff: this makes it easier to demonstrate multiple systems concepts, but it also means there is no single canonical production runtime path.

Assumption: future maintainers should preserve this modular shape unless the project is intentionally productized around one workflow.

## Normalize Benchmark Results

`BenchmarkResult` gives backend adapters a shared output shape: backend, precision, device, average latency, percentile latencies, throughput, and extra metadata.

Tradeoff: a common result type simplifies comparison, but some adapters execute benchmarks while others read historical artifacts. Consumers must inspect `extra` and adapter behavior before treating results as fresh measurements.

Assumption: artifact-backed adapters are acceptable for handoff demos as long as they are clearly labeled.

## Prefer Local Executable Paths Where Practical

The project includes real ONNX Runtime and PyTorch inference paths, a real ONNX Runtime CV backend, native C++ source, and a real CUDA RMSNorm extension path.

Tradeoff: local execution provides stronger evidence but requires dependency setup across Python, native libraries, CUDA, and optional compiler stacks.

Assumption: developers may not have all optional runtimes installed on one machine.

## Use Simulation for Serving-System Concepts

LLM runtime behavior is modeled with local dataclasses, deterministic policies, cost models, and artifact generation rather than production serving frameworks.

Tradeoff: simulations are easy to run and inspect in CI-like environments, but they cannot prove production throughput, memory behavior, or failure semantics.

Assumption: serving-framework reports should be read as conceptual/runtime-model evidence, not as vLLM, SGLang, Triton Server, or TensorRT-LLM benchmark results.

## Make PD-Split Bandwidth a Hardware Assumption

`deployment/distributed_runtime_plan.py` models KV handoff bandwidth through `HardwareConfig` and `LinkType` instead of an unexplained fixed constant. The default is a two-GPU PD split (`gpu_count=2`) over PCIe Gen4 x16 at 32 GB/s. PCIe Gen5 x16 and NVLink presets are available, and a custom override remains for measured or experimental links.

Tradeoff: preset bandwidths are still nominal cost-model assumptions, not measured transfers, but the assumption is now named and inspectable in the plan (`link_type`, `gpu_count`, and `bandwidth_source`).

Assumption: this planner simulates prefill/decode disaggregation, not tensor parallelism. `gpu_count=1` means colocated single-GPU execution with zero cross-GPU KV transfer; it does not model all-reduce or per-layer sharding.

## Separate Scheduler and Paged-Attention Evidence Modes

The README documents two LLM artifact modes: scheduler-focused and paged-attention. This keeps scheduler wins separate from extra paged-attention read-cost accounting.

Tradeoff: maintaining multiple modes increases artifact complexity, but it avoids mixing policy effects with memory-read modeling effects.

Assumption: mode comparisons should continue to state what each mode isolates.

## Inflight Paged KV Prefetch Usefulness Score Is Reporting-Only

`PagedKVLifecycle.summary()` exposes `usefulness_score = prefetch_hits / (prefetch_hits + prefetch_waste)` for the `inflight_paged_kv_continuous_batching` policy's speculative next-page prefetch (`prefetch_next_decode_page`), with `0.0` when the denominator is zero (no speculative prefetch attempt has resolved yet).

This is distinct from `prefetch_hit_rate` (`hits / (hits + misses)`), which is access-centric: it measures how often a current-page access found a warm page, including accesses where no prefetch was ever attempted (first page of a request, a pressure-guard skip, or a failed speculative allocation). `usefulness_score` is spend-centric: misses are excluded by design, because it should only describe the fate of pages that were actually prefetched speculatively — consumed (`hit`) versus discarded unused (`waste`) — not whether prefetch had the opportunity to run at all.

Tradeoff: a low `usefulness_score` with very few `prefetch_attempts` is a weak signal (most resolved attempts could be hits by chance with little data), while `prefetch_attempts` in the same summary lets a reader distinguish "no data yet" from "this policy's prefetch effort is genuinely being wasted."

Assumption: this metric is reporting-only. It is read inside `summary()` and has no effect on `prefetch_next_decode_page`'s allocation/guard logic. It is not wired into an adaptive guard, and `cost_aware_memory_pressure_page_prefetch` is unaffected.

`PagedKVLifecycle` also tracks `usefulness_score_ema`, an exponential moving average over the same per-attempt outcomes (`1.0` on a hit, `0.0` on waste), updated incrementally at the same two resolution points (`access_current_page` on a hit, `release_request` on waste) with a fixed smoothing factor `usefulness_ema_alpha = 0.2`. The first resolved sample initializes the EMA directly rather than blending against an arbitrary seed value.

Tradeoff: unlike the cumulative `usefulness_score`, the EMA is order-sensitive — the same total hits and waste produce a different EMA depending on the sequence in which they resolved, which is the point: it reflects whether prefetch usefulness is trending recently versus over the whole run. The cost is that it carries less statistical weight per sample and can swing on a short run.

Assumption: `usefulness_score_ema` is exposed in `summary()` alongside `usefulness_ema_alpha` for transparency. The cumulative `usefulness_score` remains reporting-only with no consumer; `usefulness_score_ema` is now also read by the adaptive guard described below.

## Adaptive Usefulness Guard Is Secondary and Strictly Restrictive

`PagedKVLifecycle.prefetch_next_decode_page` gates speculative next-page prefetch with two independent checks, evaluated in a fixed order: the existing memory-pressure check first, then a second adaptive check based on `usefulness_score_ema`. The pressure check returns early on its own skip reason (`memory_pressure_above_prefetch_budget`) before the adaptive check ever runs, so the adaptive guard can never see a request that pressure has already blocked, and it has no code path that allocates a page or clears a pressure-driven skip. It can only add a new reason to skip (`usefulness_below_adaptive_guard_threshold`), never override or bypass the pressure decision.

The adaptive guard only evaluates once `prefetch_hits + prefetch_waste >= usefulness_min_samples` (default `5`) — below that floor it never activates, so a handful of early misses cannot disable prefetch for the rest of a run on a noisy signal. Once warmed up, it uses two distinct thresholds rather than one, with persisted on/off state (`adaptive_guard_active`) to avoid flapping: it disables prefetch once `usefulness_score_ema <= usefulness_disable_threshold` (default `0.3`), and only re-enables once the EMA recovers to `>= usefulness_reenable_threshold` (default `0.5`). The gap between the two thresholds is the anti-flap margin.

Tradeoff: hysteresis means the guard can lag a genuine recovery in usefulness by a few resolved samples (it won't re-enable the instant the EMA ticks above the disable threshold), but the alternative — a single threshold checked statelessly every call — would toggle on and off as the EMA oscillates near that one value, producing prefetch behavior that changes step to step for no externally visible reason.

Assumption: this is the only behavioral consumer of `usefulness_score_ema` introduced so far. `cost_aware_memory_pressure_page_prefetch` and the `KVPagePool` physical benchmark are unaffected. Because this guard changes `prefetch_next_decode_page`'s actual decisions (unlike the purely-reporting `usefulness_score`/`usefulness_score_ema` metrics themselves), committed artifacts under `results/llm_runtime_artifacts/` predate this change and will not reflect it until intentionally regenerated.

## Bounded Queue for Video Pipeline Backpressure

The async video pipeline uses a bounded `queue.Queue` and drops frames when the queue is full.

Tradeoff: dropping frames protects latency and memory under load, but it sacrifices complete frame coverage.

Assumption: edge CV workloads often prefer recent frames and bounded latency over processing every frame.

## Provider Fallback for ONNX Runtime

`ONNXRuntimeCVBackend` checks available providers and falls back to the configured fallback provider when the requested provider is missing.

Tradeoff: this improves portability across CPU, CoreML, and CUDA environments, but it can hide performance differences if users do not check active/session providers.

Assumption: metrics and docs should include requested provider, active provider, and actual session providers.

## Optional Dependencies Skip or Fail Explicitly

Tests for CUDA and TVM skip when required dependencies are unavailable. Native C++ builds require explicit paths such as `ONNXRUNTIME_DIR`.

Tradeoff: the core repo remains usable without every runtime installed, but coverage varies by environment.

Assumption: maintainers should document skipped coverage when reporting validation status.

## Dataclasses for Runtime State

The LLM runtime model and metrics use dataclasses heavily for request state, cost models, memory planners, KV pages, routing state, and scheduler outputs.

Tradeoff: dataclasses keep simulation state explicit and serializable, but very large stateful classes can still become hard to navigate.

Assumption: future changes should continue preferring simple typed dataclasses and functions over deep inheritance.

## Agentic Evaluation Is Deterministic

The agentic evaluation uses a deterministic policy and a deterministic judge rather than depending on an external LLM.

Tradeoff: deterministic CI is stable and cheap, but it does not measure the variability of a real model-driven agent.

Assumption: this scaffold is intended to evaluate tool-use structure and artifact discipline, not open-ended model intelligence.

## Keep Physical KV Microbenchmark Separate From Scheduler Simulation

`scripts/benchmark_kv_page_microbenchmark.py` allocates real tensors and times real gather/copy/scatter operations, while `deployment/llm_runtime_decision.py` tracks KV pages as plain IDs in a logical simulation with formula-based cost models. The two are not merged.

Tradeoff: keeping them separate means the scheduler simulation's policy invariants stay fast and deterministic for CI, while the physical-memory benchmark stays free to add real allocator stress (e.g. checkout/release churn, fragmentation) without affecting scheduler-reported numbers or pytest runtime.

Assumption: extensions to the physical KV microbenchmark must not modify `MemoryPlanner`, `RuntimeScheduler`, or `PagedKVLifecycle`, and must not claim to measure or invoke a live vLLM/PagedAttention CUDA kernel.

## Treat KV Microbenchmarks As Offline Calibration Evidence

The scheduler already uses a local paged-KV cost model through `CostModel`, `PagedAttentionCostModel`, and `PagedKVLifecycle`. The `KVPagePool` benchmark is a physical measurement layer that can inform those constants offline, but it is not an online scheduler input.

Tradeoff: reading a benchmark JSON directly at runtime would make scheduler behavior depend on stale, hardware-specific, workload-specific measurements. The safer flow is manual/offline: run the benchmark on target hardware, inspect provenance and p50/p95 movement costs, adjust constants only when justified, regenerate artifacts, then compare TPOT, throughput, OOM/reject, and page-lifecycle gates.

Assumption: a future helper may emit suggested values for constants such as `page_read_ms`, `non_contiguous_segment_penalty_ms`, `kv_update_ms_per_block`, and prefetch hit/miss terms, but it must not auto-edit code or auto-drive scheduler policy.

## KV Cache "Fragmentation" Must Measure Fragmentation

`kv_cache_trace.json` used to report a field named `fragmentation_ratio` computed as `(total_blocks - peak_allocated_blocks) / total_blocks * 0.11`, clamped to `[0.02, 0.32]`. That formula measures unused peak capacity, not fragmentation (wasted/unusable space within or between allocations), and the `0.11` scale and clamp bounds had no derivation. The field has been removed; no external consumer depended on it.

It is replaced by four honestly-named fields:

- `free_capacity_ratio` and `peak_allocation_utilization` (in `kv_cache_trace.json`, top level): exact complements of `total_blocks` vs `peak_allocated_blocks` — headroom and utilization at peak load. Neither is fragmentation.
- `kv_internal_fragmentation_ratio` (in `page_lifecycle`, from `PagedKVLifecycle.summary()`): `1 - tokens_written / tokens_capacity_allocated`, accumulated cumulatively inside `allocate_range` across every page ever allocated for the run's lifetime, not derived from a point-in-time snapshot (released pages are popped from `PagedKVLifecycle.pages`, so a snapshot would undercount). This is true internal fragmentation: unused token capacity within allocated pages, computed from the simulation's own token/page bookkeeping.
- `contiguous_free_run_ratio` (in `page_lifecycle`, from `PagedKVLifecycle.summary()`): largest contiguous run of free page indices over pool size, an end-of-run snapshot. This reuses the exact name and definition already established by `scripts/benchmark_kv_page_microbenchmark.py`'s allocator-churn pass — free-list *index* fragmentation, not real GPU/CPU allocator memory fragmentation, since the simulation's pages are logical indices with no memory addresses.

Tradeoff: `kv_internal_fragmentation_ratio` is a true lifetime measure but `contiguous_free_run_ratio` is necessarily a snapshot (free-list contiguity has no meaningful lifetime accumulation).

Assumption: these are "Simulated" metrics per the repo's truth-boundary labels — real bookkeeping inside a simulated scheduler, not measured hardware fragmentation. They must not be conflated with `scripts/benchmark_kv_page_microbenchmark.py`'s own `contiguous_free_run_ratio`, which is the same name/definition computed from a different, measured source.

## Metrics Are Evidence, Not Universal Claims

The repository contains many historical result artifacts. Documentation should describe where metrics come from and whether they are measured, artifact-backed, simulated, or estimated.

Tradeoff: preserving artifacts is useful for demos and analysis, but stale metrics can be mistaken for current measurements.

Assumption: no new benchmark numbers should be added to documentation unless regenerated and labeled with environment metadata.
