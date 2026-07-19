"""D3A Part C/D: real, deterministic Qwen2.5-0.5B-Instruct forward pass with
a forward hook capturing the exact compiler-selected module's activation.

No synthetic fallback: every failure here raises rather than substituting
seeded random tensors. Captured tensors are always detached, cloned, and
moved off any autograd graph before being returned.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_SEED = 1234


class LiveCaptureError(RuntimeError):
    """Fail-closed: live model load, forward, or hook capture did not
    behave as required. Never silently substitutes synthetic data."""


@dataclass(frozen=True)
class LiveModelHandle:
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    load_time_s: float


def load_live_model(*, device: str = "cpu", dtype: str = "float32") -> LiveModelHandle:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    load_time_s = time.perf_counter() - t0
    return LiveModelHandle(model=model, tokenizer=tokenizer, device=device,
                           dtype=dtype, load_time_s=load_time_s)


def _require_single_invocation(invocation_count: int, module_path: str) -> None:
    """Fail-closed invocation-count check, factored out so Part M negative
    tests can exercise 'hook never fired' / 'hook fired unexpected number
    of times' directly without needing to contrive a real model forward
    that produces those counts."""
    if invocation_count == 0:
        raise LiveCaptureError(
            f"module {module_path!r} hook never fired during the forward pass -- "
            "the module was not reached; refusing to substitute synthetic data"
        )
    if invocation_count != 1:
        raise LiveCaptureError(
            f"module {module_path!r} hook fired {invocation_count} times during a single "
            "forward pass; invocation selection would be ambiguous for a one-shot prefill "
            "call, refusing to guess which invocation is intended"
        )


@dataclass(frozen=True)
class CapturedActivation:
    module_path: str
    input_activation: np.ndarray
    output_activation: np.ndarray
    weight: np.ndarray
    bias: np.ndarray | None
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    device: str
    invocation_count: int
    selected_invocation_index: int
    invocation_semantics: str
    tokenization_time_s: float
    forward_time_s: float
    hook_overhead_s: float
    capture_copy_time_s: float
    prompt: str
    token_ids: list[int]


def capture_module_activation(
    handle: LiveModelHandle, module_path: str, *,
    prompt: str = DEFAULT_PROMPT, seed: int = DEFAULT_SEED,
) -> CapturedActivation:
    import torch

    named = dict(handle.model.named_modules())
    if module_path not in named:
        raise LiveCaptureError(f"module {module_path!r} does not exist in the loaded model")
    module = named[module_path]

    torch.manual_seed(seed)

    invocations: list[tuple[Any, Any, float]] = []
    hook_time_total = 0.0

    def hook(_mod, inputs, output):
        nonlocal hook_time_total
        t0 = time.perf_counter()
        captured_in = inputs[0].detach().clone().cpu()
        captured_out = output.detach().clone().cpu()
        hook_time_total += time.perf_counter() - t0
        invocations.append((captured_in, captured_out, time.time()))

    handle_ref = module.register_forward_hook(hook)
    try:
        t0 = time.perf_counter()
        tokenized = handle.tokenizer(prompt, return_tensors="pt")
        tokenized = {k: v.to(handle.device) for k, v in tokenized.items()}
        tokenization_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            handle.model(**tokenized, use_cache=False)
        forward_time_s = time.perf_counter() - t0
    finally:
        handle_ref.remove()

    _require_single_invocation(len(invocations), module_path)

    selected_index = 0
    input_act, output_act, _ts = invocations[selected_index]

    t0 = time.perf_counter()
    weight = module.weight.detach().clone().cpu().numpy()
    bias = module.bias.detach().clone().cpu().numpy() if module.bias is not None else None
    input_np = input_act.numpy()
    output_np = output_act.numpy()
    capture_copy_time_s = time.perf_counter() - t0

    return CapturedActivation(
        module_path=module_path,
        input_activation=input_np, output_activation=output_np,
        weight=weight, bias=bias,
        input_shape=tuple(input_np.shape), output_shape=tuple(output_np.shape),
        dtype=str(input_act.dtype).replace("torch.", ""), device=handle.device,
        invocation_count=len(invocations), selected_invocation_index=selected_index,
        invocation_semantics=(
            "single forward(use_cache=False) call over the full prompt token sequence "
            "-- this is a prefill-shaped invocation (the whole sequence processed in one "
            "call), not a decode/generation loop; exactly one invocation is expected and "
            "required for this module in this call"
        ),
        tokenization_time_s=tokenization_time_s, forward_time_s=forward_time_s,
        hook_overhead_s=hook_time_total, capture_copy_time_s=capture_copy_time_s,
        prompt=prompt, token_ids=tokenized["input_ids"].cpu().tolist()[0],
    )
