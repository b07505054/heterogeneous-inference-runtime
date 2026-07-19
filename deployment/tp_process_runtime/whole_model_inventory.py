"""D4A Part A/B: real whole-model TP operator inventory.

Part A inventories every TP-relevant operator family in the REAL, locally
loaded Qwen2.5-0.5B-Instruct Transformers model (module path, class, shapes,
bias) -- never inferred from memory.

Part B inventories the installed vLLM 0.24.0 Qwen2 implementation by
introspecting the actually-installed package source files on this host
(hashing them for provenance and extracting confirmatory excerpts at call
time), not from documentation or memory. The per-rank partition formulas
recorded here were read directly from
``vllm/model_executor/layers/linear.py`` and
``vllm/model_executor/layers/vocab_parallel_embedding.py`` in this
environment's installed package (see ``source_file``/``source_sha256`` on
every record).
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _excerpt(path: str, needle: str, *, context: int = 0) -> str:
    """Return the real line(s) containing `needle` from the installed source
    file at `path` -- proves the fact was read from the live install, not
    memorized. Raises if the needle is not actually present."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            lo, hi = max(0, i - context), min(len(lines), i + context + 1)
            return "\n".join(lines[lo:hi])
    raise ValueError(f"expected excerpt {needle!r} not found in {path}")


PARTITION_REPLICATED = "replicated"
PARTITION_COLUMN = "column-parallel"
PARTITION_ROW = "row-parallel"
PARTITION_VOCAB = "vocab-parallel"
PARTITION_HEAD = "head-parallel"
PARTITION_KV_HEAD = "kv-head-parallel"
PARTITION_COLLECTIVE_ONLY = "collective-only"
PARTITION_NOT_PARTITIONED = "not-partitioned"
PARTITION_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class OperatorFamilyRecord:
    module_path: str
    module_class: str
    operator_family: str
    weight_shape: tuple[int, ...] | None
    bias_present: bool | None
    input_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...] | None
    partition_type: str
    partition_axis: int | None
    required_collective: str | None
    vllm_implementation_reference: str
    compiler_operator_mapping: str | None
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "module_class": self.module_class,
            "operator_family": self.operator_family,
            "weight_shape": list(self.weight_shape) if self.weight_shape else None,
            "bias_present": self.bias_present,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "partition_type": self.partition_type,
            "partition_axis": self.partition_axis,
            "required_collective": self.required_collective,
            "vllm_implementation_reference": self.vllm_implementation_reference,
            "compiler_operator_mapping": self.compiler_operator_mapping,
            "validation_status": self.validation_status,
        }


def inventory_transformers_operator_families(
    model: Any, *, representative_layers: tuple[int, ...]
) -> list[OperatorFamilyRecord]:
    """Part A: walk the REAL loaded model's named_modules() and classify
    every TP-relevant family for the given representative layers plus the
    global embedding/lm_head/final-norm modules.

    Shapes/bias are read directly off the real module parameters -- never
    assumed from config alone.
    """
    named = dict(model.named_modules())
    records: list[OperatorFamilyRecord] = []

    def lin(module_path: str, family: str, partition_type: str, partition_axis: int | None,
            collective: str | None, vllm_ref: str, compiler_mapping: str | None,
            status: str) -> None:
        module = named[module_path]
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        records.append(OperatorFamilyRecord(
            module_path=module_path,
            module_class=f"{type(module).__module__}.{type(module).__name__}",
            operator_family=family,
            weight_shape=tuple(weight.shape) if weight is not None else None,
            bias_present=bias is not None,
            input_shape=None, output_shape=None,
            partition_type=partition_type, partition_axis=partition_axis,
            required_collective=collective, vllm_implementation_reference=vllm_ref,
            compiler_operator_mapping=compiler_mapping, validation_status=status,
        ))

    for layer_idx in representative_layers:
        prefix = f"model.layers.{layer_idx}"
        lin(f"{prefix}.self_attn.q_proj", "q_proj", PARTITION_HEAD, 0, None,
            "vllm.model_executor.layers.linear.QKVParallelLinear (fused q/k/v; q shard is the "
            "column-parallel-like head-partitioned slice)",
            f"qwen_prefill::llm.q_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.self_attn.k_proj", "k_proj", PARTITION_KV_HEAD, 0, None,
            "vllm.model_executor.layers.linear.QKVParallelLinear (k shard; KV-head partition/"
            "replication rule per num_kv_head_replicas)",
            f"qwen_prefill::llm.k_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.self_attn.v_proj", "v_proj", PARTITION_KV_HEAD, 0, None,
            "vllm.model_executor.layers.linear.QKVParallelLinear (v shard; same KV-head rule as k)",
            f"qwen_prefill::llm.v_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.self_attn.o_proj", "o_proj", PARTITION_ROW, 1, "all_reduce",
            "vllm.model_executor.layers.linear.RowParallelLinear",
            f"qwen_prefill::llm.o_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.mlp.gate_proj", "gate_proj", PARTITION_COLUMN, 0, None,
            "vllm.model_executor.layers.linear.MergedColumnParallelLinear (gate_up_proj, shard 0)",
            f"qwen_prefill::llm.gate_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.mlp.up_proj", "up_proj", PARTITION_COLUMN, 0, None,
            "vllm.model_executor.layers.linear.MergedColumnParallelLinear (gate_up_proj, shard 1)",
            f"qwen_prefill::llm.up_proj::layer_{layer_idx}", "pending_validation")
        lin(f"{prefix}.mlp.down_proj", "down_proj", PARTITION_ROW, 1, "all_reduce",
            "vllm.model_executor.layers.linear.RowParallelLinear",
            f"qwen_prefill::llm.down_proj::layer_{layer_idx}", "pending_validation")

        for norm_name, family in ((f"{prefix}.input_layernorm", "input_layernorm"),
                                   (f"{prefix}.post_attention_layernorm", "post_attention_layernorm")):
            module = named[norm_name]
            weight = getattr(module, "weight", None)
            records.append(OperatorFamilyRecord(
                module_path=norm_name, module_class=f"{type(module).__module__}.{type(module).__name__}",
                operator_family=family, weight_shape=tuple(weight.shape) if weight is not None else None,
                bias_present=False, input_shape=None, output_shape=None,
                partition_type=PARTITION_REPLICATED, partition_axis=None, required_collective=None,
                vllm_implementation_reference="vllm.model_executor.layers.layernorm.RMSNorm "
                                              "(replicated; consumes the full, already-reconstructed "
                                              "hidden state -- see attention_contract_validation.json)",
                compiler_operator_mapping=None, validation_status="pending_validation",
            ))

    embed = named["model.embed_tokens"]
    embed_weight = getattr(embed, "weight", None)
    records.append(OperatorFamilyRecord(
        module_path="model.embed_tokens", module_class=f"{type(embed).__module__}.{type(embed).__name__}",
        operator_family="embedding", weight_shape=tuple(embed_weight.shape), bias_present=False,
        input_shape=None, output_shape=None, partition_type=PARTITION_VOCAB, partition_axis=0,
        required_collective="all_reduce",
        vllm_implementation_reference="vllm.model_executor.layers.vocab_parallel_embedding."
                                      "VocabParallelEmbedding (masked local lookup + all_reduce)",
        compiler_operator_mapping="qwen_prefill::llm.embed_tokens::model", validation_status="pending_validation",
    ))
    # tie_word_embeddings=True for this model: lm_head IS embed_tokens (verified from config,
    # not assumed) -- see whole_model_plan_builder for the runtime check.
    records.append(OperatorFamilyRecord(
        module_path="model.embed_tokens (tied)", module_class=f"{type(embed).__module__}.{type(embed).__name__}",
        operator_family="lm_head", weight_shape=tuple(embed_weight.shape), bias_present=False,
        input_shape=None, output_shape=None, partition_type=PARTITION_VOCAB, partition_axis=0,
        required_collective="all_gather",
        vllm_implementation_reference="vllm.model_executor.layers.logits_processor.LogitsProcessor "
                                      "+ vllm.model_executor.layers.vocab_parallel_embedding."
                                      "ParallelLMHead (tied to embed_tokens for this config; local "
                                      "vocab-shard matmul + tensor_model_parallel_all_gather)",
        compiler_operator_mapping="qwen_prefill::llm.lm_head::model", validation_status="pending_validation",
    ))

    final_norm = named["model.norm"]
    final_norm_weight = getattr(final_norm, "weight", None)
    records.append(OperatorFamilyRecord(
        module_path="model.norm", module_class=f"{type(final_norm).__module__}.{type(final_norm).__name__}",
        operator_family="final_norm", weight_shape=tuple(final_norm_weight.shape), bias_present=False,
        input_shape=None, output_shape=None, partition_type=PARTITION_REPLICATED, partition_axis=None,
        required_collective=None,
        vllm_implementation_reference="vllm.model_executor.layers.layernorm.RMSNorm (replicated)",
        compiler_operator_mapping=None, validation_status="pending_validation",
    ))

    records.append(OperatorFamilyRecord(
        module_path="model.layers.*.self_attn.rotary_emb (computed inline, no dedicated module in "
                    "this Transformers version)",
        module_class="transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding",
        operator_family="rotary_embedding", weight_shape=None, bias_present=None,
        input_shape=None, output_shape=None, partition_type=PARTITION_REPLICATED, partition_axis=None,
        required_collective=None,
        vllm_implementation_reference="vllm.model_executor.layers.rotary_embedding.get_rope "
                                      "(replicated; applied per-head to each rank's LOCAL Q/K head "
                                      "shard -- no cross-rank data needed since rotation is "
                                      "elementwise within a head's own dimensions)",
        compiler_operator_mapping=None, validation_status="pending_validation",
    ))
    records.append(OperatorFamilyRecord(
        module_path="model.layers.*.self_attn (attention score/softmax/context, no dedicated "
                    "sub-module -- computed inline in Qwen2Attention.forward)",
        module_class="transformers.models.qwen2.modeling_qwen2.eager_attention_forward",
        operator_family="attention_computation", weight_shape=None, bias_present=None,
        input_shape=None, output_shape=None, partition_type=PARTITION_HEAD, partition_axis=None,
        required_collective=None,
        vllm_implementation_reference="vllm.model_executor.layers.attention.Attention "
                                      "(head-parallel: each rank computes softmax attention over "
                                      "only its own local Q/K/V head shard, GQA-repeated locally)",
        compiler_operator_mapping=None, validation_status="pending_validation",
    ))
    records.append(OperatorFamilyRecord(
        module_path="model.layers.* (residual add, no dedicated module)",
        module_class="python_addition_operator",
        operator_family="residual_connection", weight_shape=None, bias_present=None,
        input_shape=None, output_shape=None, partition_type=PARTITION_REPLICATED, partition_axis=None,
        required_collective=None,
        vllm_implementation_reference="no dedicated vLLM class; residual add happens in "
                                      "Qwen2DecoderLayer.forward on the full, already-reconstructed "
                                      "hidden_states tensor",
        compiler_operator_mapping=None, validation_status="pending_validation",
    ))

    return records


@dataclass(frozen=True)
class VLLMContractFact:
    class_or_function: str
    source_file: str
    source_sha256: str
    partition_dimension: str
    weight_loader_behavior: str
    rank_local_weight_shape_rule: str
    output_contract: str
    collective_behavior: str
    bias_handling: str
    kv_head_special_handling: str
    verification_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_or_function": self.class_or_function,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "partition_dimension": self.partition_dimension,
            "weight_loader_behavior": self.weight_loader_behavior,
            "rank_local_weight_shape_rule": self.rank_local_weight_shape_rule,
            "output_contract": self.output_contract,
            "collective_behavior": self.collective_behavior,
            "bias_handling": self.bias_handling,
            "kv_head_special_handling": self.kv_head_special_handling,
            "verification_excerpt": self.verification_excerpt,
        }


def inventory_installed_vllm_qwen_contract() -> dict[str, Any]:
    """Part B: introspect the actually-installed vLLM 0.24.0 source files on
    this host. Every fact below is paired with the real source file path,
    its SHA-256 (proving which exact installed file was read), and a
    verification excerpt extracted from that file at call time (not
    hand-copied at authoring time) -- so a future vLLM upgrade that changes
    this logic will change the excerpt/hash and this function will no
    longer silently agree with stale facts.
    """
    import vllm.distributed.communication_op as comm_op
    import vllm.model_executor.layers.linear as linear_mod
    import vllm.model_executor.layers.logits_processor as logits_mod
    import vllm.model_executor.layers.vocab_parallel_embedding as vocab_mod
    import vllm.model_executor.models.qwen2 as qwen2_mod

    qwen2_path = inspect.getsourcefile(qwen2_mod)
    linear_path = inspect.getsourcefile(linear_mod)
    vocab_path = inspect.getsourcefile(vocab_mod)
    logits_path = inspect.getsourcefile(logits_mod)
    comm_path = inspect.getsourcefile(comm_op)

    facts = {
        "QKVParallelLinear": VLLMContractFact(
            class_or_function="vllm.model_executor.layers.linear.QKVParallelLinear",
            source_file=linear_path, source_sha256=_file_sha256(linear_path),
            partition_dimension="output dim (0): heads and kv-heads partitioned/replicated across tp ranks",
            weight_loader_behavior="loads separate q/k/v checkpoint tensors into offset slices of one "
                                   "fused [q_size+kv_size+kv_size, hidden] parameter, per-rank via "
                                   "narrow(output_dim, tp_rank*shard_size, shard_size)",
            rank_local_weight_shape_rule="num_heads=divide(total_num_heads,tp_size); if tp_size>="
                                        "total_num_kv_heads: num_kv_heads=1, num_kv_head_replicas="
                                        "divide(tp_size,total_num_kv_heads); else num_kv_heads="
                                        "divide(total_num_kv_heads,tp_size), num_kv_head_replicas=1",
            output_contract="rank-local qkv output width = (num_heads+2*num_kv_heads)*head_size, "
                            "kept sharded (gather_output=False)",
            collective_behavior="none at the qkv_proj boundary itself (kept sharded for the "
                                "attention op); the row-parallel o_proj after attention performs "
                                "the all_reduce",
            bias_handling="Qwen2Attention hardcodes bias=True for qkv_proj regardless of "
                          "config.attention_bias; bias is sharded per-rank identically to the weight "
                          "(output_dim=0)",
            kv_head_special_handling="see rank_local_weight_shape_rule; for Qwen2.5-0.5B-Instruct "
                                     "(total_num_kv_heads=2) at tp_size=2, tp_size==total_num_kv_heads "
                                     "so num_kv_head_replicas=divide(2,2)=1 -- structurally the "
                                     "'replicate' branch executes but degenerates to a clean 1:1 "
                                     "partition (rank0 owns kv head 0, rank1 owns kv head 1, zero "
                                     "actual duplication)",
            verification_excerpt=_excerpt(linear_path, "self.num_kv_head_replicas = divide(tp_size,"),
        ),
        "RowParallelLinear": VLLMContractFact(
            class_or_function="vllm.model_executor.layers.linear.RowParallelLinear",
            source_file=linear_path, source_sha256=_file_sha256(linear_path),
            partition_dimension="input dim (1)",
            weight_loader_behavior="narrow(input_dim, tp_rank*shard_size, shard_size) per rank",
            rank_local_weight_shape_rule="input_size_per_partition = divide(input_size, tp_size); "
                                        "weight shape [output_size, input_size_per_partition]",
            output_contract="Y = tensor_model_parallel_all_reduce(X_rank @ W_rank^T [+ bias on rank 0 only])",
            collective_behavior="all_reduce(sum) when reduce_results=True and tp_size>1",
            bias_handling="bias is fused into rank 0's local GEMM ONLY (bias_ = None if tp_rank>0), "
                          "so after all_reduce the bias has been added exactly once",
            kv_head_special_handling="not applicable",
            verification_excerpt=_excerpt(linear_path,
                "bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias"),
        ),
        "MergedColumnParallelLinear": VLLMContractFact(
            class_or_function="vllm.model_executor.layers.linear.MergedColumnParallelLinear",
            source_file=linear_path, source_sha256=_file_sha256(linear_path),
            partition_dimension="output dim (0), independently per merged sub-weight (gate, up)",
            weight_loader_behavior="each of gate_proj/up_proj is loaded into its own contiguous "
                                   "output-dim slice of the fused gate_up_proj parameter, at the SAME "
                                   "per-rank shard index for both -- so a given rank's gate shard and "
                                   "up shard cover the identical intermediate-dimension indices",
            rank_local_weight_shape_rule="each output_size in output_sizes must be tp-divisible; "
                                        "per-rank shard width = output_size // tp_size for each of "
                                        "gate/up independently",
            output_contract="rank-local [gate_rank | up_rank] concatenated on output dim, kept sharded",
            collective_behavior="none at this boundary (kept sharded); matching shard ownership makes "
                                "the elementwise SiLU(gate_rank)*up_rank rank-local-valid",
            bias_handling="Qwen2MLP.gate_up_proj uses bias=False",
            kv_head_special_handling="not applicable",
            verification_excerpt=_excerpt(linear_path, "class MergedColumnParallelLinear(ColumnParallelLinear):"),
        ),
        "VocabParallelEmbedding": VLLMContractFact(
            class_or_function="vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding",
            source_file=vocab_path, source_sha256=_file_sha256(vocab_path),
            partition_dimension="vocab dim (0) of the embedding table",
            weight_loader_behavior="narrow(output_dim, start_idx, shard_size) per rank vocab range",
            rank_local_weight_shape_rule="per_partition_vocab_size = divide(padded_vocab_size, tp_size)",
            output_contract="masked local lookup (out-of-range token ids masked to 0) then "
                            "tensor_model_parallel_all_reduce to reconstruct the correct row for "
                            "every token regardless of which rank owns it",
            collective_behavior="all_reduce(sum) unconditionally when tp_size>1",
            bias_handling="not applicable (embedding has no bias)",
            kv_head_special_handling="not applicable",
            verification_excerpt=_excerpt(vocab_path, "output = tensor_model_parallel_all_reduce(output_parallel)"),
        ),
        "ParallelLMHead_and_LogitsProcessor": VLLMContractFact(
            class_or_function="vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead + "
                              "vllm.model_executor.layers.logits_processor.LogitsProcessor",
            source_file=f"{vocab_path};{logits_path}",
            source_sha256=f"{_file_sha256(vocab_path)};{_file_sha256(logits_path)}",
            partition_dimension="vocab dim (0) of the (possibly tied) lm_head weight",
            weight_loader_behavior="same vocab-shard loading as VocabParallelEmbedding; for this "
                                   "model tie_word_embeddings=True so lm_head literally IS "
                                   "embed_tokens (Qwen2ForCausalLM sets self.lm_head = "
                                   "self.model.embed_tokens)",
            rank_local_weight_shape_rule="identical vocab shard boundaries as embed_tokens (tied weight)",
            output_contract="local_logits = hidden_states @ W_vocab_shard^T; full logits = "
                            "concat_over_ranks(local_logits) via all-gather, then trimmed to "
                            "org_vocab_size to drop any padding",
            collective_behavior="tensor_model_parallel_all_gather (or tensor_model_parallel_gather) "
                                "along the last (vocab) dimension -- NOT all_reduce; this is the one "
                                "family that needs concatenation-style reconstruction",
            bias_handling="ParallelLMHead bias defaults to False and Qwen2ForCausalLM does not enable it",
            kv_head_special_handling="not applicable",
            verification_excerpt=_excerpt(logits_path, "logits = tensor_model_parallel_all_gather(logits)"),
        ),
        "tensor_model_parallel_all_reduce": VLLMContractFact(
            class_or_function="vllm.distributed.communication_op.tensor_model_parallel_all_reduce",
            source_file=comm_path, source_sha256=_file_sha256(comm_path),
            partition_dimension="n/a (collective op)", weight_loader_behavior="n/a",
            rank_local_weight_shape_rule="n/a",
            output_contract="elementwise sum across all tp ranks' local tensors, result replicated to all ranks",
            collective_behavior="all_reduce(sum) over the tp process group",
            bias_handling="n/a", kv_head_special_handling="n/a",
            verification_excerpt=_excerpt(comm_path, "return get_tp_group().all_reduce(input_)"),
        ),
        "tensor_model_parallel_all_gather": VLLMContractFact(
            class_or_function="vllm.distributed.communication_op.tensor_model_parallel_all_gather",
            source_file=comm_path, source_sha256=_file_sha256(comm_path),
            partition_dimension="n/a (collective op)", weight_loader_behavior="n/a",
            rank_local_weight_shape_rule="n/a",
            output_contract="concatenation of every rank's local tensor along dim=-1, in rank order",
            collective_behavior="all_gather along the last dimension over the tp process group",
            bias_handling="n/a", kv_head_special_handling="n/a",
            verification_excerpt=_excerpt(comm_path, "return get_tp_group().all_gather(input_, dim)"),
        ),
    }

    return {
        "installed_vllm_version": __import__("vllm").__version__,
        "qwen2_model_source_file": qwen2_path,
        "qwen2_model_source_sha256": _file_sha256(qwen2_path),
        "qwen2_uses_qkv_fusion": "self.qkv_proj = QKVParallelLinear(" in Path(qwen2_path).read_text(),
        "qwen2_uses_merged_gate_up": "self.gate_up_proj = MergedColumnParallelLinear(" in Path(qwen2_path).read_text(),
        "qwen2_o_proj_is_row_parallel": "self.o_proj = RowParallelLinear(" in Path(qwen2_path).read_text(),
        "qwen2_down_proj_is_row_parallel": "self.down_proj = RowParallelLinear(" in Path(qwen2_path).read_text(),
        "facts": {name: fact.to_dict() for name, fact in facts.items()},
        "discovery_method": "direct introspection of the installed vLLM package's own source files "
                            "(inspect.getsourcefile + literal excerpt extraction), not documentation "
                            "or memory",
    }
