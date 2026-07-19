"""D4A Part G: whole-model TP=2 replay.

Preferred Part G approach: TP-relevant linear/embedding modules are
replaced by serialized TP-contract-aware computation (attention_contract_
executor, mlp_contract_executor, vocab_parallel_executor), while every
other real model operation -- RMSNorm, rotary embedding cos/sin
computation, the causal mask, residual adds -- is executed by calling the
REAL model's own submodules/functions, unchanged.

The serialized TP model never calls the original full q_proj/k_proj/
v_proj/o_proj/gate_proj/up_proj/down_proj/embed_tokens/lm_head as its
output path -- those real modules are used ONLY as the weight source (via
.weight/.bias) and, separately, as the untouched reference model for
comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from deployment.execution_plan.schema import DistributedTensorShard
from deployment.tp_process_runtime.attention_contract_executor import run_serialized_tp_attention_block
from deployment.tp_process_runtime.mlp_contract_executor import run_serialized_tp_mlp_block
from deployment.tp_process_runtime.vocab_parallel_executor import (
    build_vocab_rank_shards,
    rank_local_lm_head_logits,
    rank_local_masked_embedding,
    reconstruct_embedding,
    reconstruct_lm_head_logits,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class WholeModelReplayError(RuntimeError):
    """Fail-closed: the whole-model TP replay could not be composed or
    validated. Never falls back to the original full linear output."""


def load_eager_model():
    """A fresh, eager-attention-forced load (attn_implementation='eager'),
    needed for exact numerical parity with the numpy attention
    reimplementation -- distinct from live_capture.load_live_model(), which
    does not pin attn_implementation and is used by D3A's existing,
    unmodified single-operator path."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    load_time_s = time.perf_counter() - t0
    return model, tokenizer, load_time_s


def group_shards_by_tensor_id(
    tensor_shards: tuple[DistributedTensorShard, ...],
) -> dict[str, tuple[DistributedTensorShard, ...]]:
    grouped: dict[str, list[DistributedTensorShard]] = {}
    for s in tensor_shards:
        grouped.setdefault(s.tensor_id, []).append(s)
    return {k: tuple(sorted(v, key=lambda s: s.shard_index)) for k, v in grouped.items()}


@dataclass(frozen=True)
class ReferenceForwardResult:
    logits: np.ndarray
    hidden_states_per_layer: tuple[np.ndarray, ...]  # index 0 = embedding output, index i = after layer i-1
    forward_time_s: float
    token_ids: list[int]


def run_reference_forward(model, tokenizer, prompt: str, *, seed: int = 1234) -> ReferenceForwardResult:
    """The unmodified real model forward -- used only for comparison, never
    as part of the TP-simulated output path.

    Per-layer hidden states are captured via direct forward hooks on each
    real Qwen2DecoderLayer (the same unambiguous mechanism D3A uses for its
    single-operator capture), NOT via output_hidden_states=True: this
    model's HF version special-cases the LAST output_hidden_states entry to
    be the POST-final-norm hidden state rather than decoder layer 23's own
    raw output, which would silently misalign an index-based comparison at
    the last layer. Hooks avoid that ambiguity entirely.
    """
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")

    captured_layers: list[torch.Tensor] = []

    def hook(_mod, _args, _kwargs, output):
        captured_layers.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    handles = [layer.register_forward_hook(hook, with_kwargs=True) for layer in model.model.layers]
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    forward_time_s = time.perf_counter() - t0

    embed_output = model.model.embed_tokens(inputs["input_ids"]).detach()
    hidden_states_per_layer = tuple(
        h.numpy() for h in ([embed_output] + captured_layers)
    )
    return ReferenceForwardResult(
        logits=out.logits.detach().numpy(),
        hidden_states_per_layer=hidden_states_per_layer,
        forward_time_s=forward_time_s,
        token_ids=inputs["input_ids"][0].tolist(),
    )


@dataclass(frozen=True)
class LayerBlockTrace:
    layer_id: int
    attn_max_abs_error_vs_real_module: float | None
    hidden_state_max_abs_error_vs_reference: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "hidden_state_max_abs_error_vs_reference": self.hidden_state_max_abs_error_vs_reference,
        }


@dataclass(frozen=True)
class SerializedTPForwardResult:
    logits: np.ndarray
    hidden_states_per_layer: tuple[np.ndarray, ...]
    layer_traces: tuple[LayerBlockTrace, ...]
    forward_time_s: float
    attention_rank_traces: dict[int, Any]
    mlp_rank_traces: dict[int, Any]


def run_serialized_tp_whole_model_forward(
    model, tokenizer, prompt: str, *, shards_by_tensor_id: dict[str, tuple[DistributedTensorShard, ...]],
    reference: ReferenceForwardResult | None = None, seed: int = 1234,
) -> SerializedTPForwardResult:
    """Manual re-implementation of Qwen2Model.forward/Qwen2DecoderLayer.forward:
    RMSNorm, rotary cos/sin, and residual adds use the REAL model's own
    submodules/tensors unchanged; q/k/v/o_proj, gate/up/down_proj, embedding,
    and lm_head are computed exclusively through the D4A TP executors.
    """
    from transformers.masking_utils import create_causal_mask

    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    cfg = model.config
    num_heads_per_rank = cfg.num_attention_heads // 2
    num_kv_heads_per_rank = max(1, cfg.num_key_value_heads // 2)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    t0 = time.perf_counter()

    embed_shards = shards_by_tensor_id["qwen_prefill::llm.embed_tokens::model"]
    embed_weight_np = model.model.embed_tokens.weight.detach().numpy()
    vshards = build_vocab_rank_shards(embed_weight_np, embed_shards)
    token_ids_np = input_ids.numpy()
    embed_rank_outs = {rid: rank_local_masked_embedding(token_ids_np, s) for rid, s in vshards.items()}
    hidden_np = reconstruct_embedding(embed_rank_outs)  # [batch, seq, hidden]
    hidden_states_per_layer = [hidden_np]

    hidden_torch = torch.from_numpy(hidden_np)
    position_ids = torch.arange(hidden_torch.shape[1]).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_torch, position_ids)
    cos_np, sin_np = cos.numpy(), sin.numpy()

    mask_kwargs = {
        "config": cfg, "inputs_embeds": hidden_torch, "attention_mask": None,
        "past_key_values": None, "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}

    layer_traces = []
    attn_rank_traces: dict[int, Any] = {}
    mlp_rank_traces: dict[int, Any] = {}

    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        residual = hidden_np

        normed_torch = layer.input_layernorm(torch.from_numpy(hidden_np))
        normed_np = normed_torch.detach().numpy()

        self_attn = layer.self_attn
        q_shards = shards_by_tensor_id[f"qwen_prefill::llm.q_proj::layer_{layer_idx}"]
        k_shards = shards_by_tensor_id[f"qwen_prefill::llm.k_proj::layer_{layer_idx}"]
        v_shards = shards_by_tensor_id[f"qwen_prefill::llm.v_proj::layer_{layer_idx}"]
        o_shards = shards_by_tensor_id[f"qwen_prefill::llm.o_proj::layer_{layer_idx}"]

        attn_result = run_serialized_tp_attention_block(
            hidden_states=normed_np,
            q_weight=self_attn.q_proj.weight.detach().numpy(), q_bias=self_attn.q_proj.bias.detach().numpy(),
            k_weight=self_attn.k_proj.weight.detach().numpy(), k_bias=self_attn.k_proj.bias.detach().numpy(),
            v_weight=self_attn.v_proj.weight.detach().numpy(), v_bias=self_attn.v_proj.bias.detach().numpy(),
            o_weight=self_attn.o_proj.weight.detach().numpy(),
            cos=cos_np, sin=sin_np, num_heads_per_rank=num_heads_per_rank,
            num_kv_heads_per_rank=num_kv_heads_per_rank, head_dim=head_dim, world_size=2,
            q_shards=q_shards, k_shards=k_shards, v_shards=v_shards, o_shards=o_shards,
        )
        attn_rank_traces[layer_idx] = [t.to_dict() for t in attn_result.rank_traces]
        hidden_np = residual + attn_result.reconstructed_output

        residual = hidden_np
        normed2_torch = layer.post_attention_layernorm(torch.from_numpy(hidden_np))
        normed2_np = normed2_torch.detach().numpy()

        mlp = layer.mlp
        gate_shards = shards_by_tensor_id[f"qwen_prefill::llm.gate_proj::layer_{layer_idx}"]
        up_shards = shards_by_tensor_id[f"qwen_prefill::llm.up_proj::layer_{layer_idx}"]
        down_shards = shards_by_tensor_id[f"qwen_prefill::llm.down_proj::layer_{layer_idx}"]
        mlp_result = run_serialized_tp_mlp_block(
            hidden_states=normed2_np,
            gate_weight=mlp.gate_proj.weight.detach().numpy(), up_weight=mlp.up_proj.weight.detach().numpy(),
            down_weight=mlp.down_proj.weight.detach().numpy(),
            gate_shards=gate_shards, up_shards=up_shards, down_shards=down_shards,
        )
        mlp_rank_traces[layer_idx] = [t.to_dict() for t in mlp_result.rank_traces]
        hidden_np = residual + mlp_result.reconstructed_output

        hidden_states_per_layer.append(hidden_np)
        ref_err = (
            float(np.abs(hidden_np - reference.hidden_states_per_layer[layer_idx + 1]).max())
            if reference is not None else float("nan")
        )
        layer_traces.append(LayerBlockTrace(
            layer_id=layer_idx, attn_max_abs_error_vs_real_module=None,
            hidden_state_max_abs_error_vs_reference=ref_err,
        ))

    final_norm_torch = model.model.norm(torch.from_numpy(hidden_np))
    final_hidden_np = final_norm_torch.detach().numpy()

    lm_shards = shards_by_tensor_id["qwen_prefill::llm.lm_head::model"]
    lm_rank_shards = build_vocab_rank_shards(embed_weight_np, lm_shards)
    logit_parts = {rid: rank_local_lm_head_logits(final_hidden_np, s) for rid, s in lm_rank_shards.items()}
    logits_np = reconstruct_lm_head_logits(logit_parts, org_vocab_size=cfg.vocab_size)

    forward_time_s = time.perf_counter() - t0

    return SerializedTPForwardResult(
        logits=logits_np, hidden_states_per_layer=tuple(hidden_states_per_layer),
        layer_traces=tuple(layer_traces), forward_time_s=forward_time_s,
        attention_rank_traces=attn_rank_traces, mlp_rank_traces=mlp_rank_traces,
    )
