# E1 ExecuTorch Raspberry Pi Baseline Bring-Up

DOCUMENT STATUS: CURRENT E1 BASELINE BRING-UP RESULT

Last verified: 2026-07-13

Truth boundary: this document establishes a real ExecuTorch v1.3.1 XNNPACK-capable smoke baseline on Raspberry Pi 5 for FP32 `Y = ReLU(A @ B + bias)`. It is not a formal project-versus-ExecuTorch comparison and does not claim superiority.

## Source And Host State

- ExecuTorch upstream: `https://github.com/pytorch/executorch.git`
- ExecuTorch tag: `v1.3.1`
- ExecuTorch commit: `e2f18eb23c45bd22ca332b0b8b49a81de304b472`
- Compiler repository at audit start: `b67cd644568e7f53a64370f926e241e4e42ebe10`, branch `master`, clean, ahead 9 of `origin/master`.
- Runtime repository at audit start: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6`, branch `main`, clean, ahead 3 of `origin/main`.
- Capability repository at audit start: `aac593da0bdde7a95c38c03920fc4d00b73011db`, branch `main`.
- Raspberry Pi: `edgeaiplatform`, Raspberry Pi 5 Model B Rev 1.1, Debian 13, aarch64 Cortex-A76, four cores, performance governor, `throttled=0x0` during smoke.

## Mutation And Rollback

All E1 mutable work was isolated under:

- GPU: `/home/allen/executorch_e1`
- Raspberry Pi: `/home/allen/executorch_e1`

No project production source, Runtime adapter, project kernel, P1D evidence, boot config, or persistent governor setting was modified. Rollback is removal of the dedicated E1 workspaces.

The recorded mutation manifest is `results/executorch_e1/E1_MUTATION_ROLLBACK_MANIFEST.md`.

## Build Strategy

Native Raspberry Pi build was used after aarch64 cross-compiler tools were not present on the GPU host. The official source was cloned and pinned on the GPU, submodules were initialized there, and a source-tree archive was transferred to the Pi because direct Pi submodule fetch repeatedly failed on the Eigen submodule.

The Pi build used a local virtualenv only. PyTorch was needed by ExecuTorch CMake for header discovery. The pinned aarch64 PyTorch wheel was fetched with a resumable transfer and installed in the local E1 build virtualenv without installing CUDA dependencies; CMake only required the package path, not full `import torch` execution.

Runner build output:

- Binary: `/home/allen/executorch_e1/build/executorch-cmake3/executor_runner`
- SHA256: `eb3068fb1742e4172a459f9f4c5aebd2dd9dd43151e214ff1402ea925d4e2809`
- Type: ELF 64-bit LSB PIE executable, ARM aarch64, dynamically linked.
- XNNPACK: enabled.
- pthreadpool/threadpool: enabled.
- Full configure and build logs were retained in the dedicated E1 workspaces. They are not committed because repository ignore policy excludes `.log` files; committed artifacts retain source, export, smoke, and hash provenance.

## Workload Contract

The E1 workload computes exactly:

```text
Y = ReLU(A @ B + bias)
```

- `A`: contiguous FP32 tensor `[M, K]`
- `B`: contiguous FP32 tensor `[K, N]`
- `bias`: contiguous FP32 tensor `[N]`, broadcast across rows
- output: contiguous FP32 tensor `[M, N]`
- accumulation: FP32
- reference used for smoke: ExecuTorch runner default all-ones inputs; expected output value is `K + 1` for every output element.
- tolerance: max absolute error <= `1e-5` for the all-ones smoke reference.

The workload definition is `evaluation/executorch_e1/export_fused_matmul_bias_relu.py`. It does not call the project native kernel and does not define a custom ExecuTorch operator.

## Exported Artifacts

Fixed-shape `.pte` artifacts were exported for `8x8x8`, `64x64x64`, and `256x256x256`, both portable and XNNPACK-lowered. Large generated `.pte` binaries are not committed; their SHA256 values are recorded in `results/executorch_e1/e1_export_artifact_hashes.txt`.

Primary XNNPACK hashes:

- `8x8x8`: `0a4c4cbfa3812dbce9d31ab41d637e412ec9c5230dffb4c1221770c17bf8aff1`
- `64x64x64`: `0e080a8e3943dd2d2a100add24abb0d43a8d569d97ab31d214a3cf3ff60230bb`
- `256x256x256`: `98ca7e01d3e6070d857c0589dcbbf926035bddb4415f808846a3e4b5b8488285`

## Graph And Delegation Status

All three XNNPACK artifacts are classified as:

`FULL_REGION_DELEGATED_FUSION_UNKNOWN`

Evidence:

- Export reports show XNNPACK delegate calls for each shape.
- The full semantic region is delegated.
- Internal XNNPACK fusion is not proven by this evidence and must not be claimed.

Per-shape export reports are retained under `results/executorch_e1/export_reports/`.

## Thread Control

The official `executor_runner` exposes `--cpu_threads`. Logs show:

- default mode resets the threadpool to 4 threads on this Pi.
- `--cpu_threads 1` resets the threadpool to 1 thread.
- `--cpu_threads 4` resets the threadpool to 4 threads.

Truth boundary: threadpool reset logs and `taskset -c 0-3` affinity were verified. Hidden helper-thread sampling was not independently measured in E1, so detailed thread-behavior comparison remains a gate for E2.

## Timing Boundary

The runner reports per-iteration timing around `method->execute()`. E1 smoke timing excludes process startup, model load, reference computation, and output validation from the primary per-iteration values. The runner separately logs model load time. This is a smoke validation of timing instrumentation only, not a benchmark campaign.

## Smoke Results

Protocol:

- shapes: `8x8x8`, `64x64x64`, `256x256x256`
- modes: default, `--cpu_threads 1`, `--cpu_threads 4`
- affinity: `taskset -c 0-3`
- executions: 15 per run
- warm values summarized after excluding first 5 iterations
- governor: performance
- throttling: `0x0` in recorded smoke logs

| Shape | Mode | Correct | First Run ms | Warm Median ms | Warm P95 ms | Thread Evidence |
|---|---:|---:|---:|---:|---:|---|
| 8x8x8 | default | True | 0.030296 | 0.001075 | 0.001092 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |
| 8x8x8 | t1 | True | 0.020037 | 0.001056 | 0.001074 | Resetting threadpool with num threads = 1, Resetting threadpool to 1 threads. |
| 8x8x8 | t4 | True | 0.019334 | 0.001046 | 0.001056 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |
| 64x64x64 | default | True | 0.045648 | 0.020889 | 0.020926 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |
| 64x64x64 | t1 | True | 0.045908 | 0.020917 | 0.020963 | Resetting threadpool with num threads = 1, Resetting threadpool to 1 threads. |
| 64x64x64 | t4 | True | 0.046389 | 0.020908 | 0.020945 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |
| 256x256x256 | default | True | 1.269111 | 1.125047 | 1.216816 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |
| 256x256x256 | t1 | True | 1.247630 | 1.133574 | 1.174223 | Resetting threadpool with num threads = 1, Resetting threadpool to 1 threads. |
| 256x256x256 | t4 | True | 1.296575 | 1.122167 | 1.187241 | Resetting threadpool with num threads = 4, Resetting threadpool to 4 threads. |

Raw parsed smoke data is retained in `results/executorch_e1/e1_xnnpack_smoke_summary.json`.

## Limitations

- This is not a formal head-to-head benchmark.
- RSS was not collected because `/usr/bin/time -v` is absent on the Pi and no package was installed for E1.
- Hidden helper-thread sampling was not independently verified.
- Dynamic-shape `.pte` support was not established; E1 uses one fixed-shape artifact per smoke shape.
- Internal XNNPACK fusion is unknown.
- Smoke timings are single-session validation values and must not be used as final comparison claims.

## Gate Reassessment

| Gate | Status | Evidence |
|---|---|---|
| ExecuTorch Runtime executes on Pi | PASS | aarch64 `executor_runner` loads and runs XNNPACK `.pte` artifacts. |
| Same semantic workload is correct | PASS | all smoke outputs match all-ones FP32 reference exactly within tolerance. |
| Thread behavior controlled or observed | PARTIAL PASS | `--cpu_threads` changes threadpool reset logs; deeper helper-thread observation remains for E2. |
| Measurement boundaries documented | PASS | per-iteration `method->execute()` timing boundary identified. |
| Frozen comparison manifest ready | PARTIAL | artifact hashes and workload contract exist; formal E2 manifest still required. |
| Artifacts reproducible with hashes | PASS | source, export, `.pte`, wheel, and runner hashes recorded. |
| Governor/affinity/environment control works | PASS | `taskset -c 0-3`, performance governor, and no-throttle logs recorded. |
| Result ingestion contract defined | PARTIAL | machine-readable smoke JSON exists; IVP adapter not implemented. |
| Claim boundaries pre-registered | PASS | no superiority or formal comparison claim in E1. |
| No project-kernel custom operator | PASS | workload is native PyTorch graph lowered by ExecuTorch/XNNPACK. |

## Verdict

`READY_FOR_EXECUTORCH_FIXED_CONFIGURATION_BASELINE`

ExecuTorch v1.3.1 with XNNPACK is now built and smoke-validated on Raspberry Pi 5 for the target FP32 fused-region workload. E2 should not proceed as a scheduling-policy comparison until thread observability is strengthened and the formal comparison manifest is frozen.
