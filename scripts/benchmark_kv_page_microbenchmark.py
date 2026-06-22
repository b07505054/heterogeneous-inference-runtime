import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TRUTH_BOUNDARY = (
    "This benchmark measures tensor-backed paged KV storage and gather/copy overhead. "
    "It is not vLLM PagedAttention, not TensorRT-LLM paged attention, and not wired "
    "into live Qwen attention."
)


@dataclass(frozen=True)
class RequestShape:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    prefix_key: str


class _FallbackDType:
    def __init__(self, name, element_size):
        self.name = name
        self._element_size = element_size

    def __repr__(self):
        return f"torch.{self.name}"

    def element_size(self):
        return self._element_size


class _FallbackTensor:
    def __init__(self, shape, *, dtype, device="cpu"):
        if shape == ():
            self.shape = ()
        elif isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def element_size(self):
        return self.dtype.element_size()

    def normal_(self, mean=0.0, std=1.0):
        return self

    def contiguous(self):
        return self

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return _FallbackTensor(tuple(self.shape[dim] for dim in dims), dtype=self.dtype, device=self.device)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        out_shape = []
        for dim_size, index in zip(self.shape, key):
            if isinstance(index, int):
                continue
            if isinstance(index, slice):
                start, stop, step = index.indices(dim_size)
                out_shape.append(max(0, math.ceil((stop - start) / step)))
                continue
            raise TypeError(f"unsupported fallback tensor index {index!r}")
        return _FallbackTensor(tuple(out_shape), dtype=self.dtype, device=self.device)

    def __setitem__(self, key, value):
        return None


class _FallbackCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        return None


class _FallbackMps:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        return None


class _FallbackBackends:
    mps = _FallbackMps()


class _FallbackTorch:
    __version__ = "fallback-shape-only"
    float32 = _FallbackDType("float32", 4)
    float16 = _FallbackDType("float16", 2)
    bfloat16 = _FallbackDType("bfloat16", 2)
    cuda = _FallbackCuda()
    backends = _FallbackBackends()
    mps = _FallbackMps()

    @staticmethod
    def empty(shape, *, dtype=None, device="cpu"):
        return _FallbackTensor(shape, dtype=dtype or _FallbackTorch.float32, device=device)

    @staticmethod
    def randn(shape, *, dtype=None, device="cpu"):
        return _FallbackTensor(shape, dtype=dtype or _FallbackTorch.float32, device=device)

    @staticmethod
    def cat(tensors, dim=0):
        if not tensors:
            raise ValueError("torch.cat fallback needs at least one tensor")
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        return _FallbackTensor(tuple(shape), dtype=tensors[0].dtype, device=tensors[0].device)


def import_torch_or_fallback():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        fallback = _FallbackTorch()
        sys.modules.setdefault("torch", fallback)
        return fallback


import_torch_or_fallback()


class KVPagePool:
    def __init__(self, *, page_count, page_size_tokens, num_kv_heads, head_dim, dtype, device):
        self.page_count = page_count
        self.page_size_tokens = page_size_tokens
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.tensor = None
        self.free_pages = []
        self.owners = {}

    def init(self, torch):
        self.tensor = torch.empty(
            (self.page_count, 2, self.page_size_tokens, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.free_pages = list(range(self.page_count))
        self.owners = {}

    def checkout(self, request_id, count):
        if count > len(self.free_pages):
            raise MemoryError("insufficient KV pages")
        pages = self.free_pages[:count]
        del self.free_pages[:count]
        for page in pages:
            self.owners[page] = request_id
        return pages

    def release(self, pages):
        for page in pages:
            self.owners.pop(page, None)
        self.free_pages.extend(pages)
        self.free_pages.sort()

    def used_pages(self):
        return self.page_count - len(self.free_pages)

    def leaked_pages(self):
        return len(self.owners)


def synthetic_requests(count=32):
    seed = [
        (256, 64),
        (512, 128),
        (1024, 64),
        (2048, 128),
        (128, 32),
        (4096, 64),
        (512, 256),
        (1024, 128),
    ]
    rows = []
    for index, (prompt, output) in enumerate((seed * math.ceil(count / len(seed)))[:count]):
        rows.append(
            RequestShape(
                request_id=f"req-{index + 1:03d}",
                prompt_tokens=prompt,
                output_tokens=output,
                prefix_key=f"prefix-{index % 4}",
            )
        )
    return rows


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize_ms(values):
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "runs": 0}
    return {
        "p50_ms": round(percentile(values, 50), 6),
        "p95_ms": round(percentile(values, 95), 6),
        "mean_ms": round(statistics.mean(values), 6),
        "runs": len(values),
    }


def synchronize(torch, device):
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
    except Exception:
        pass


def timed_ms(torch, device, fn):
    synchronize(torch, device)
    start = time.perf_counter()
    out = fn()
    synchronize(torch, device)
    return (time.perf_counter() - start) * 1000, out


def dtype_from_name(torch, name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return None
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return None
    return requested


def page_count_for(request, page_size_tokens):
    return math.ceil((request.prompt_tokens + request.output_tokens) / page_size_tokens)


def materialize_request_pages(torch, pool, pages, total_tokens):
    assert pool.tensor is not None
    for logical_index, page in enumerate(pages):
        begin = logical_index * pool.page_size_tokens
        valid_tokens = max(0, min(pool.page_size_tokens, total_tokens - begin))
        if valid_tokens <= 0:
            continue
        pool.tensor[page, :, :valid_tokens].normal_(mean=0.0, std=0.02)


def gather_paged_kv(torch, pool, pages, total_tokens):
    assert pool.tensor is not None
    chunks = []
    remaining = total_tokens
    for page in pages:
        take = min(pool.page_size_tokens, remaining)
        if take <= 0:
            break
        chunks.append(pool.tensor[page, :, :take])
        remaining -= take
    if not chunks:
        return torch.empty(
            (0, 2, pool.num_kv_heads, pool.head_dim),
            dtype=pool.dtype,
            device=pool.device,
        )
    return torch.cat(chunks, dim=1).permute(1, 0, 2, 3).contiguous()


def contiguous_slice_copy(contiguous, total_tokens):
    return contiguous[:total_tokens].contiguous()


def scatter_one_token(torch, pool, pages, token_index):
    assert pool.tensor is not None
    page_offset = token_index // pool.page_size_tokens
    within_page = token_index % pool.page_size_tokens
    page = pages[page_offset]
    value = torch.randn((2, pool.num_kv_heads, pool.head_dim), dtype=pool.dtype, device=pool.device)
    pool.tensor[page, :, within_page] = value


def contiguous_update_one_token(torch, contiguous, token_index):
    value = torch.randn((2, contiguous.shape[2], contiguous.shape[3]), dtype=contiguous.dtype, device=contiguous.device)
    contiguous[token_index] = value


def unavailable_report(args, reason, dependency_status):
    return {
        "artifact_type": "kv_page_microbenchmark_report",
        "status": "unavailable",
        "reason": reason,
        "device": args.device,
        "dtype": args.dtype,
        "truth_boundary": TRUTH_BOUNDARY,
        "dependency_status": dependency_status,
    }


def dependency_status(torch=None):
    status = {"python": platform.python_version(), "platform": platform.platform(), "torch": {"available": False}}
    if torch is None:
        try:
            import torch as imported_torch  # type: ignore

            torch = imported_torch
        except Exception as exc:
            status["torch"]["error"] = f"{type(exc).__name__}: {exc}"
            return status
    if isinstance(torch, _FallbackTorch):
        status["torch"] = {
            "available": False,
            "fallback_active": True,
            "fallback": "shape_only",
            "version": getattr(torch, "__version__", None),
            "cuda_available": False,
            "mps_available": False,
        }
        return status
    status["torch"] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    return status


def run_benchmark(args):
    try:
        import torch  # type: ignore
    except Exception as exc:
        return unavailable_report(args, f"torch import failed: {type(exc).__name__}: {exc}", dependency_status())

    device = resolve_device(torch, args.device)
    if device is None:
        report = unavailable_report(
            args,
            f"requested device {args.device} is unavailable",
            dependency_status(torch),
        )
        if args.fail_on_unavailable:
            raise SystemExit(2)
        return report

    dtype = dtype_from_name(torch, args.dtype)
    requests = synthetic_requests(args.request_count)
    max_tokens = max(req.prompt_tokens + req.output_tokens for req in requests)
    max_pages_per_request = math.ceil(max_tokens / args.page_size_tokens)
    page_count = max(args.page_count, max_pages_per_request * min(len(requests), 8))
    pool = KVPagePool(
        page_count=page_count,
        page_size_tokens=args.page_size_tokens,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    init_ms, _ = timed_ms(torch, device, lambda: pool.init(torch))

    warmup_request = requests[0]
    warmup_pages = pool.checkout(warmup_request.request_id, page_count_for(warmup_request, args.page_size_tokens))
    materialize_request_pages(torch, pool, warmup_pages, warmup_request.prompt_tokens + warmup_request.output_tokens)
    _ = gather_paged_kv(torch, pool, warmup_pages, warmup_request.prompt_tokens)
    pool.release(warmup_pages)

    checkout_ms = []
    release_ms = []
    materialize_ms = []
    paged_gather_ms = []
    contiguous_copy_ms = []
    scatter_update_ms = []
    contiguous_update_ms = []
    reused_pages = 0
    avoided_copy_bytes = 0
    prefix_cache = {}

    for _ in range(args.iterations):
        for request in requests:
            total_tokens = request.prompt_tokens + request.output_tokens
            needed_pages = page_count_for(request, args.page_size_tokens)
            ms, pages = timed_ms(torch, device, lambda request=request, needed_pages=needed_pages: pool.checkout(request.request_id, needed_pages))
            checkout_ms.append(ms)

            prefix_pages = min(max(1, request.prompt_tokens // args.page_size_tokens), len(pages))
            if request.prefix_key in prefix_cache:
                reused_pages += min(prefix_pages, len(prefix_cache[request.prefix_key]))
                avoided_copy_bytes += (
                    min(prefix_pages, len(prefix_cache[request.prefix_key]))
                    * args.page_size_tokens
                    * 2
                    * args.num_kv_heads
                    * args.head_dim
                    * torch.empty((), dtype=dtype).element_size()
                )
            else:
                prefix_cache[request.prefix_key] = pages[:prefix_pages]

            ms, _ = timed_ms(torch, device, lambda pages=pages, total_tokens=total_tokens: materialize_request_pages(torch, pool, pages, total_tokens))
            materialize_ms.append(ms)

            contiguous = torch.empty((total_tokens, 2, args.num_kv_heads, args.head_dim), dtype=dtype, device=device)
            contiguous.normal_(mean=0.0, std=0.02)
            logical_tokens = request.prompt_tokens

            ms, paged_out = timed_ms(torch, device, lambda pages=pages, logical_tokens=logical_tokens: gather_paged_kv(torch, pool, pages, logical_tokens))
            paged_gather_ms.append(ms)
            ms, contiguous_out = timed_ms(torch, device, lambda contiguous=contiguous, logical_tokens=logical_tokens: contiguous_slice_copy(contiguous, logical_tokens))
            contiguous_copy_ms.append(ms)
            if paged_out.shape != contiguous_out.shape:
                raise AssertionError(f"paged gather shape {paged_out.shape} != contiguous shape {contiguous_out.shape}")

            update_index = min(total_tokens - 1, request.prompt_tokens)
            ms, _ = timed_ms(torch, device, lambda pages=pages, update_index=update_index: scatter_one_token(torch, pool, pages, update_index))
            scatter_update_ms.append(ms)
            ms, _ = timed_ms(torch, device, lambda contiguous=contiguous, update_index=update_index: contiguous_update_one_token(torch, contiguous, update_index))
            contiguous_update_ms.append(ms)

            ms, _ = timed_ms(torch, device, lambda pages=pages: pool.release(pages))
            release_ms.append(ms)

    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    logical_token_count = sum(req.prompt_tokens + req.output_tokens for req in requests)
    logical_bytes = logical_token_count * 2 * args.num_kv_heads * args.head_dim * bytes_per_element
    allocated_bytes = sum(page_count_for(req, args.page_size_tokens) for req in requests) * args.page_size_tokens * 2 * args.num_kv_heads * args.head_dim * bytes_per_element
    fragmentation_ratio = max(0.0, (allocated_bytes - logical_bytes) / max(allocated_bytes, 1))

    contiguous_summary = summarize_ms(contiguous_copy_ms)
    paged_summary = summarize_ms(paged_gather_ms)
    contiguous_update_summary = summarize_ms(contiguous_update_ms)
    paged_update_summary = summarize_ms(scatter_update_ms)
    gather_delta = paged_summary["p95_ms"] - contiguous_summary["p95_ms"]
    update_delta = paged_update_summary["p95_ms"] - contiguous_update_summary["p95_ms"]

    return {
        "artifact_type": "kv_page_microbenchmark_report",
        "status": "completed",
        "source": "scripts/benchmark_kv_page_microbenchmark.py",
        "device": device,
        "dtype": args.dtype,
        "page_size_tokens": args.page_size_tokens,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "truth_boundary": TRUTH_BOUNDARY,
        "dependency_status": dependency_status(torch),
        "workload": {
            "request_count": len(requests),
            "generated_decode_steps": args.iterations,
            "prompt_tokens": {
                "min": min(req.prompt_tokens for req in requests),
                "max": max(req.prompt_tokens for req in requests),
                "total": sum(req.prompt_tokens for req in requests),
            },
            "output_tokens": {
                "min": min(req.output_tokens for req in requests),
                "max": max(req.output_tokens for req in requests),
                "total": sum(req.output_tokens for req in requests),
            },
        },
        "contiguous_baseline": {
            "contiguous_slice_copy": contiguous_summary,
            "one_token_append_update": contiguous_update_summary,
        },
        "paged_kv": {
            "page_pool_init_cost": {"ms": round(init_ms, 6), "page_count": page_count},
            "page_checkout_cost": summarize_ms(checkout_ms),
            "page_release_cost": summarize_ms(release_ms),
            "free_list_reuse_cost": summarize_ms(checkout_ms + release_ms),
            "tensor_materialization_cost": summarize_ms(materialize_ms),
            "leaked_pages_after_run": pool.leaked_pages(),
        },
        "movement_cost": {
            "paged_gather_to_contiguous": paged_summary,
            "contiguous_slice_copy": contiguous_summary,
            "scatter_update_one_token_kv": paged_update_summary,
            "contiguous_update_one_token_kv": contiguous_update_summary,
        },
        "prefix_reuse_proxy": {
            "reused_pages": reused_pages,
            "avoided_copy_bytes": avoided_copy_bytes,
            "estimated_saved_prefill_copy_mb": round(avoided_copy_bytes / (1024 * 1024), 6),
        },
        "comparison": {
            "paged_gather_p95_over_contiguous_copy_p95_ms": round(gather_delta, 6),
            "paged_gather_overhead_pct": round((gather_delta / contiguous_summary["p95_ms"]) * 100, 3) if contiguous_summary["p95_ms"] else 0.0,
            "paged_update_p95_over_contiguous_update_p95_ms": round(update_delta, 6),
            "paged_update_overhead_pct": round((update_delta / contiguous_update_summary["p95_ms"]) * 100, 3) if contiguous_update_summary["p95_ms"] else 0.0,
            "fragmentation_ratio": round(fragmentation_ratio, 6),
            "logical_kv_mb": round(logical_bytes / (1024 * 1024), 6),
            "allocated_page_kv_mb": round(allocated_bytes / (1024 * 1024), 6),
        },
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure tensor-backed paged KV cache movement costs.")
    parser.add_argument("--output", default="results/llm_runtime_artifacts/kv_page_microbenchmark_report.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--request-count", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--page-size-tokens", type=int, default=16)
    parser.add_argument("--page-count", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--fail-on-unavailable", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_benchmark(args)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output,
                "device": report.get("device"),
                "comparison": report.get("comparison"),
                "reason": report.get("reason"),
            },
            indent=2,
        )
    )
    if args.fail_on_unavailable and report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
