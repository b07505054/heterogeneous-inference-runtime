import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class TinyLlamaBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 256,
        intermediate_size: int = 768,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm1 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    def prefill(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embed(tokens)
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        attn = self._merge_heads(self._attention(q, k, v))
        h = x + self.out_proj(attn)
        y = self.norm2(h)
        mlp = self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))
        logits = self.lm_head(h + mlp)
        return logits, k.detach(), v.detach()

    def decode_step(
        self,
        token: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embed(token)
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k_new = self._split_heads(k_new)
        v_new = self._split_heads(v_new)
        k = torch.cat([past_k, k_new], dim=2)
        v = torch.cat([past_v, v_new], dim=2)
        attn = self._merge_heads(self._attention(q, k, v))
        h = x + self.out_proj(attn)
        y = self.norm2(h)
        mlp = self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))
        logits = self.lm_head(h + mlp)
        return logits, k.detach(), v.detach()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil((p / 100.0) * len(values)) - 1))
    return values[index]


def measure(fn, device: torch.device) -> tuple[float, object]:
    synchronize(device)
    start = time.perf_counter()
    out = fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0, out


def available_devices(requested: list[str]) -> list[torch.device]:
    devices: list[torch.device] = []
    for name in requested:
        if name == "cpu":
            devices.append(torch.device("cpu"))
        elif name == "mps" and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        elif name == "cuda" and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
    return devices


def profile_backend(
    device: torch.device,
    batch_sizes: list[int],
    sequence_lengths: list[int],
    decode_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    torch.manual_seed(0)
    model = TinyLlamaBlock().to(device).eval()
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    rows = []
    operator_rows = []

    for batch in batch_sizes:
        for seq in sequence_lengths:
            tokens = torch.randint(0, 4096, (batch, seq), device=device)
            decode_token = torch.randint(0, 4096, (batch, 1), device=device)

            with torch.no_grad():
                for _ in range(warmup):
                    logits, past_k, past_v = model.prefill(tokens)
                    logits, past_k, past_v = model.decode_step(decode_token, past_k, past_v)
                prefill_latencies = []
                tpot_latencies = []

                for _ in range(runs):
                    prefill_ms, (logits, past_k, past_v) = measure(
                        lambda: model.prefill(tokens),
                        device,
                    )
                    prefill_latencies.append(prefill_ms)

                    step_latencies = []
                    for _step in range(decode_steps):
                        step_ms, (logits, past_k, past_v) = measure(
                            lambda: model.decode_step(decode_token, past_k, past_v),
                            device,
                        )
                        step_latencies.append(step_ms)
                    tpot_latencies.extend(step_latencies)

                # Operator breakdown for this shape. Run once per op group to keep it cheap.
                x = model.embed(tokens)
                op_timings = {}
                op_timings["embed"], x = measure(lambda: model.embed(tokens), device)
                op_timings["rmsnorm"], h = measure(lambda: model.norm1(x), device)
                op_timings["qkv_projection"], qkv = measure(lambda: model.qkv(h), device)
                q, k, v = qkv.chunk(3, dim=-1)
                q, k, v = model._split_heads(q), model._split_heads(k), model._split_heads(v)
                op_timings["attention"], attn_heads = measure(lambda: model._attention(q, k, v), device)
                attn = model._merge_heads(attn_heads)
                op_timings["output_projection"], h2 = measure(lambda: model.out_proj(attn), device)
                y = model.norm2(x + h2)
                op_timings["mlp"], mlp = measure(
                    lambda: model.down_proj(F.silu(model.gate_proj(y)) * model.up_proj(y)),
                    device,
                )
                op_timings["lm_head"], _ = measure(lambda: model.lm_head(x + h2 + mlp), device)

                total_op_ms = sum(op_timings.values()) or 1.0
                for op, latency_ms in sorted(op_timings.items(), key=lambda item: item[1], reverse=True):
                    operator_rows.append({
                        "backend": device.type,
                        "batch_size": batch,
                        "sequence_length": seq,
                        "op": op,
                        "latency_ms": round(latency_ms, 4),
                        "percent": round(latency_ms / total_op_ms * 100.0, 2),
                    })

                ttft_p95 = percentile(prefill_latencies, 95)
                tpot_p95 = percentile(tpot_latencies, 95)
                rows.append({
                    "backend": device.type,
                    "batch_size": batch,
                    "sequence_length": seq,
                    "decode_steps": decode_steps,
                    "ttft_ms": {
                        "p50": round(statistics.median(prefill_latencies), 4),
                        "p95": round(ttft_p95, 4),
                        "p99": round(percentile(prefill_latencies, 99), 4),
                    },
                    "tpot_ms": {
                        "p50": round(statistics.median(tpot_latencies), 4),
                        "p95": round(tpot_p95, 4),
                        "p99": round(percentile(tpot_latencies, 99), 4),
                    },
                    "tokens_per_second": round(1000.0 / max(statistics.mean(tpot_latencies), 1e-6), 4),
                    "model_size_mb": round(model_size_mb, 4),
                })

    return {
        "backend": device.type,
        "device": str(device),
        "rows": rows,
        "operator_breakdown": operator_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/llm_runtime_artifacts")
    parser.add_argument("--devices", default="mps,cpu")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--sequence-lengths", default="64,128")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    devices = available_devices(requested_devices)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    sequence_lengths = [int(item) for item in args.sequence_lengths.split(",") if item.strip()]

    payload = {
        "artifact_type": "real_llama_profile",
        "model": "tiny-llama-block-random-weights",
        "profile_source": "torch_real_backend_execution",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "requested_devices": requested_devices,
        "available_devices": [device.type for device in devices],
        "batch_sizes": batch_sizes,
        "sequence_lengths": sequence_lengths,
        "decode_steps": args.decode_steps,
        "backends": [],
        "unavailable_backends": [
            name for name in requested_devices if name not in {device.type for device in devices}
        ],
    }

    for device in devices:
        payload["backends"].append(
            profile_backend(
                device=device,
                batch_sizes=batch_sizes,
                sequence_lengths=sequence_lengths,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    (output_dir / "real_llama_profile.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "real_llama_profile.json")


if __name__ == "__main__":
    main()
