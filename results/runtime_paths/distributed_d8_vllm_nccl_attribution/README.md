# D8 vLLM NCCL Attribution

Phase 3 validates Phase 1 NCCL microbenchmark predictions against real vLLM TP2 decode execution.

Status: measurement attempted

Required boundary: 2x RTX 4090, PHB, single NUMA, CUDA P2P unavailable, NCCL SHM/direct/direct.

D7 assumption under test: `exposed_comm_time = raw_nccl_time`.
Candidate D8 correction: `exposed_comm_time = raw_nccl_time * (1 - measured_overlap_ratio)`.

Nsight overlap rule: overlap is computed from CUDA/NVTX timeline interval intersections, not from CUDA event duration alone.

Dependency inventory:
```json
{
  "can_collect_nsys": true,
  "can_run_real_vllm": true,
  "modules": {
    "numpy": "/workspace/d8-vllm-env/lib/python3.12/site-packages/numpy/__init__.py",
    "requests": "/workspace/d8-vllm-env/lib/python3.12/site-packages/requests/__init__.py",
    "torch": "/workspace/d8-vllm-env/lib/python3.12/site-packages/torch/__init__.py",
    "transformers": "/workspace/d8-vllm-env/lib/python3.12/site-packages/transformers/__init__.py",
    "vllm": "/workspace/d8-vllm-env/lib/python3.12/site-packages/vllm/__init__.py"
  },
  "nsys_path": "/usr/local/bin/nsys",
  "nsys_version": {
    "argv": [
      "/usr/local/bin/nsys",
      "--version"
    ],
    "returncode": 0,
    "stderr": "",
    "stdout": "NVIDIA Nsight Systems version 2025.3.1.0\n"
  },
  "python_executable": "/workspace/d8-vllm-env/bin/python"
}
```
