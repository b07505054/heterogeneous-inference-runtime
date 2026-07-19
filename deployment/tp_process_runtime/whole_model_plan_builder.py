"""D4A Part C: whole-model distributed work-item plan.

Extends the compiler distributed representation only as necessary: this
module builds a "distributed" block that is schema-identical to D1/D2's
DistributedPlan (deployment/execution_plan/schema.py, unmodified dataclass
shapes) and validated by the SAME loader
(deployment.execution_plan.loader.validate_execution_plan, unmodified
except the additive KNOWN_COLLECTIVE_KINDS widening). It represents every
TP-relevant operator instance across the whole real model -- not only one
o_proj -- while being honest that column-parallel families (q/k/v/gate/up)
require NO collective in real vLLM execution (they stay sharded and feed
directly into the next rank-local op), matching
whole_model_inventory.inventory_installed_vllm_qwen_contract() exactly.

This is a Python-side expansion of the same schema/legality vocabulary the
production DistributedStrategyPlanningPass (ml-graph-compiler-runtime,
C++) already emits for a single operator -- it does not modify or rebuild
that C++ pass. See the D4A report's "Known limitations" section for why
that is an explicit, documented scope boundary rather than a silent gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deployment.tp_process_runtime.whole_model_inventory import (
    PARTITION_COLUMN,
    PARTITION_KV_HEAD,
    PARTITION_ROW,
    PARTITION_VOCAB,
)

WORLD_SIZE = 2
TENSOR_PARALLEL_SIZE = 2
PIPELINE_PARALLEL_SIZE = 1


@dataclass(frozen=True)
class ModelDims:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    tie_word_embeddings: bool


def read_model_dims(model: Any) -> ModelDims:
    """Read every dimension directly off the real loaded model's config --
    never hardcoded, so a different checkpoint would produce a different
    (still-correct) plan."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return ModelDims(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_hidden_layers=cfg.num_hidden_layers,
        tie_word_embeddings=bool(cfg.tie_word_embeddings),
    )


def _even_split(total: int, parts: int) -> list[tuple[int, int]]:
    if total % parts != 0:
        raise ValueError(f"{total} is not evenly divisible by {parts}")
    chunk = total // parts
    return [(i * chunk, (i + 1) * chunk) for i in range(parts)]


@dataclass(frozen=True)
class WorkItem:
    """Part C required per-work-item fields."""

    operator_id: str
    layer_id: int | str
    operator_family: str
    partition_strategy: str
    partition_axis: int | None
    world_size: int
    rank_shard_offsets: list[int]
    rank_shard_extents: list[int]
    weight_partition: str
    activation_partition: str
    output_partition: str
    collective_kind: str | None
    collective_sequence_id: int | None
    bias_policy: str
    replicated_inputs: list[str]
    reconstructed_output_contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id, "layer_id": self.layer_id,
            "operator_family": self.operator_family, "partition_strategy": self.partition_strategy,
            "partition_axis": self.partition_axis, "world_size": self.world_size,
            "rank_shard_offsets": self.rank_shard_offsets, "rank_shard_extents": self.rank_shard_extents,
            "weight_partition": self.weight_partition, "activation_partition": self.activation_partition,
            "output_partition": self.output_partition, "collective_kind": self.collective_kind,
            "collective_sequence_id": self.collective_sequence_id, "bias_policy": self.bias_policy,
            "replicated_inputs": self.replicated_inputs,
            "reconstructed_output_contract": self.reconstructed_output_contract,
        }


def build_whole_model_plan(model: Any, *, source_tp2_plan_id: str) -> tuple[dict[str, Any], list[WorkItem]]:
    """Returns (execution_plan_dict, work_items).

    execution_plan_dict is schema-compatible with
    deployment.execution_plan.loader.validate_execution_plan (schema
    "execution_plan" v2.0.0, "distributed" block using the exact D1/D2
    DistributedPlan/DistributedTensorShard/DistributedCollectiveStep field
    names).
    """
    dims = read_model_dims(model)
    if not dims.tie_word_embeddings:
        raise ValueError(
            "this plan builder's lm_head handling assumes tie_word_embeddings=True "
            "(verified false on the loaded model) -- refusing to silently guess an "
            "untied ParallelLMHead contract instead"
        )

    tensor_shards: list[dict[str, Any]] = []
    collectives: list[dict[str, Any]] = []
    work_items: list[WorkItem] = []
    seq = 0

    def add_shards(tensor_id: str, axis: int, total: int) -> list[tuple[int, int]]:
        ranges = _even_split(total, WORLD_SIZE)
        for idx, (start, end) in enumerate(ranges):
            tensor_shards.append({
                "tensor_id": tensor_id, "partition_axis": axis, "partition_count": WORLD_SIZE,
                "shard_index": idx, "range_start": start, "range_end": end,
            })
        return ranges

    def add_collective(tensor_id: str, kind: str, reduction: str) -> int:
        nonlocal seq
        cid = f"{kind}_{tensor_id.replace('::', '_')}"
        collectives.append({
            "collective_id": cid, "sequence_id": seq, "kind": kind,
            "participants": list(range(WORLD_SIZE)), "tensor_id": tensor_id, "reduction": reduction,
        })
        seq += 1
        return seq - 1

    # --- embedding (first op in the real forward) ---
    embed_id = "qwen_prefill::llm.embed_tokens::model"
    ranges = add_shards(embed_id, 0, dims.vocab_size)
    embed_seq = add_collective(embed_id, "all_reduce", "sum")
    work_items.append(WorkItem(
        operator_id=embed_id, layer_id="model", operator_family="embedding",
        partition_strategy=PARTITION_VOCAB, partition_axis=0, world_size=WORLD_SIZE,
        rank_shard_offsets=[r[0] for r in ranges], rank_shard_extents=[r[1] - r[0] for r in ranges],
        weight_partition="[vocab_size, hidden_size] sharded along dim 0 (vocab)",
        activation_partition="input token IDs replicated to every rank; each rank masks IDs "
                             "outside its vocab range to 0 before lookup",
        output_partition="each rank produces a full [seq, hidden] tensor that is 0 for tokens "
                         "not owned by that rank",
        collective_kind="all_reduce", collective_sequence_id=embed_seq, bias_policy="no_bias",
        replicated_inputs=["input_token_ids"],
        reconstructed_output_contract="all_reduce(sum) of the two masked lookups reconstructs the "
                                      "correct embedding row for every token",
    ))

    for layer in range(dims.num_hidden_layers):
        prefix = f"qwen_prefill::llm"

        # q_proj: column-parallel over heads
        q_id = f"{prefix}.q_proj::layer_{layer}"
        q_out = dims.num_attention_heads * dims.head_dim
        q_ranges = add_shards(q_id, 0, q_out)
        work_items.append(WorkItem(
            operator_id=q_id, layer_id=layer, operator_family="q_proj",
            partition_strategy=PARTITION_COLUMN + "_head_partitioned", partition_axis=0, world_size=WORLD_SIZE,
            rank_shard_offsets=[r[0] for r in q_ranges], rank_shard_extents=[r[1] - r[0] for r in q_ranges],
            weight_partition="[num_heads*head_dim, hidden_size] sharded along dim 0 into "
                             "num_heads_per_rank*head_dim chunks",
            activation_partition="input hidden_states fully replicated to every rank",
            output_partition="rank owns num_heads_per_rank query heads; kept sharded (no collective) "
                             "and consumed directly by that rank's local attention computation",
            collective_kind=None, collective_sequence_id=None, bias_policy="bias_sharded_per_rank_output_dim",
            replicated_inputs=["hidden_states"],
            reconstructed_output_contract="not materialized in real execution; validated standalone "
                                          "via concatenation for operator-level comparison only",
        ))

        # k_proj / v_proj: kv-head partition (degenerate replicas=1 at tp_size==total_kv_heads)
        for name, fam in (("k_proj", "k_proj"), ("v_proj", "v_proj")):
            kv_id = f"{prefix}.{name}::layer_{layer}"
            kv_out = dims.num_key_value_heads * dims.head_dim
            kv_ranges = add_shards(kv_id, 0, kv_out)
            work_items.append(WorkItem(
                operator_id=kv_id, layer_id=layer, operator_family=fam,
                partition_strategy=PARTITION_KV_HEAD, partition_axis=0, world_size=WORLD_SIZE,
                rank_shard_offsets=[r[0] for r in kv_ranges], rank_shard_extents=[r[1] - r[0] for r in kv_ranges],
                weight_partition="[num_kv_heads*head_dim, hidden_size] sharded along dim 0; "
                                "num_kv_head_replicas=divide(tp_size,total_kv_heads)=1 for this "
                                "model+tp_size (clean partition, no duplication)",
                activation_partition="input hidden_states fully replicated to every rank",
                output_partition="rank owns num_kv_heads_per_rank key/value heads; kept sharded",
                collective_kind=None, collective_sequence_id=None,
                bias_policy="bias_sharded_per_rank_output_dim",
                replicated_inputs=["hidden_states"],
                reconstructed_output_contract="not materialized in real execution; validated "
                                              "standalone via concatenation for operator-level "
                                              "comparison only",
            ))

        # o_proj: row-parallel, reuses D3A's exact 448/448 shard contract at layer 0
        o_id = f"{prefix}.o_proj::layer_{layer}"
        o_ranges = add_shards(o_id, 1, q_out)
        o_seq = add_collective(o_id, "all_reduce", "sum")
        work_items.append(WorkItem(
            operator_id=o_id, layer_id=layer, operator_family="o_proj",
            partition_strategy=PARTITION_ROW, partition_axis=1, world_size=WORLD_SIZE,
            rank_shard_offsets=[r[0] for r in o_ranges], rank_shard_extents=[r[1] - r[0] for r in o_ranges],
            weight_partition="[hidden_size, num_heads*head_dim] sharded along dim 1 (input/contraction)",
            activation_partition="input attention-output activation sharded identically to q_proj's "
                                 "head ownership (rank's local attention-head-concat output)",
            output_partition="each rank produces a full [seq, hidden_size] PARTIAL sum",
            collective_kind="all_reduce", collective_sequence_id=o_seq, bias_policy="no_bias",
            replicated_inputs=[],
            reconstructed_output_contract="all_reduce(sum) of the two partial outputs reconstructs "
                                          "the true o_proj output (no bias in this model)",
        ))

        # gate_proj / up_proj: column-parallel, matching shard ownership
        for name, fam in (("gate_proj", "gate_proj"), ("up_proj", "up_proj")):
            g_id = f"{prefix}.{name}::layer_{layer}"
            g_ranges = add_shards(g_id, 0, dims.intermediate_size)
            work_items.append(WorkItem(
                operator_id=g_id, layer_id=layer, operator_family=fam,
                partition_strategy=PARTITION_COLUMN, partition_axis=0, world_size=WORLD_SIZE,
                rank_shard_offsets=[r[0] for r in g_ranges], rank_shard_extents=[r[1] - r[0] for r in g_ranges],
                weight_partition="[intermediate_size, hidden_size] sharded along dim 0",
                activation_partition="input hidden_states fully replicated to every rank",
                output_partition="rank owns the SAME intermediate-dim shard index for gate and up "
                                 "(matching ownership), so SiLU(gate_rank)*up_rank is rank-local-valid",
                collective_kind=None, collective_sequence_id=None,
                bias_policy="no_bias", replicated_inputs=["hidden_states"],
                reconstructed_output_contract="not materialized in real execution; validated "
                                              "standalone via concatenation for operator-level "
                                              "comparison only",
            ))

        # down_proj: row-parallel
        d_id = f"{prefix}.down_proj::layer_{layer}"
        d_ranges = add_shards(d_id, 1, dims.intermediate_size)
        d_seq = add_collective(d_id, "all_reduce", "sum")
        work_items.append(WorkItem(
            operator_id=d_id, layer_id=layer, operator_family="down_proj",
            partition_strategy=PARTITION_ROW, partition_axis=1, world_size=WORLD_SIZE,
            rank_shard_offsets=[r[0] for r in d_ranges], rank_shard_extents=[r[1] - r[0] for r in d_ranges],
            weight_partition="[hidden_size, intermediate_size] sharded along dim 1 (input/contraction)",
            activation_partition="input is the rank-local SiLU(gate_rank)*up_rank shard",
            output_partition="each rank produces a full [seq, hidden_size] PARTIAL sum",
            collective_kind="all_reduce", collective_sequence_id=d_seq, bias_policy="no_bias",
            replicated_inputs=[],
            reconstructed_output_contract="all_reduce(sum) of the two partial outputs reconstructs "
                                          "the true down_proj/MLP output",
        ))

    # --- lm_head (tied to embed_tokens; last op) ---
    lm_id = "qwen_prefill::llm.lm_head::model"
    lm_ranges = add_shards(lm_id, 0, dims.vocab_size)
    lm_seq = add_collective(lm_id, "all_gather", "concat")
    work_items.append(WorkItem(
        operator_id=lm_id, layer_id="model", operator_family="lm_head",
        partition_strategy=PARTITION_VOCAB, partition_axis=0, world_size=WORLD_SIZE,
        rank_shard_offsets=[r[0] for r in lm_ranges], rank_shard_extents=[r[1] - r[0] for r in lm_ranges],
        weight_partition="tied to embed_tokens: [vocab_size, hidden_size] sharded along dim 0",
        activation_partition="input final hidden_states fully replicated to every rank",
        output_partition="each rank produces its own [seq, vocab_shard] local logits",
        collective_kind="all_gather", collective_sequence_id=lm_seq, bias_policy="no_bias",
        replicated_inputs=["final_hidden_states"],
        reconstructed_output_contract="all_gather (concat along vocab dim, rank order) of the two "
                                      "local logit shards reconstructs the full [seq, vocab] logits; "
                                      "no padding trim needed since vocab_size is exactly divisible "
                                      "by world_size for this model",
    ))

    plan = {
        "schema": "execution_plan", "schema_version": "2.0.0",
        "plan_id": "d4a-whole-model-tp-contract-plan",
        "provenance": {
            "compiler_tool": "d4a_whole_model_plan_builder (Python, generated additively; does not "
                             "modify or re-invoke the C++ DistributedStrategyPlanningPass -- see "
                             "known_limitations in the D4A report)",
            "model_spec_ref": source_tp2_plan_id,
            "capability_bundle": {"hardware_profile_ref": "nvidia-gtx1650-maxq-d4a-whole-model",
                                  "backend_profile_refs": [], "kernel_profile_refs": []},
            "truth_boundary": "d4a_whole_model_serialized_tp_contract_validation_not_measured_runtime",
        },
        "model_identity": {
            "model_id": "qwen2.5-0.5b", "model_family": "qwen2",
            "hidden_size": dims.hidden_size, "num_attention_heads": dims.num_attention_heads,
            "num_kv_heads": dims.num_key_value_heads, "num_layers": dims.num_hidden_layers,
            "intermediate_size": dims.intermediate_size, "vocab_size": dims.vocab_size,
            "tie_word_embeddings": dims.tie_word_embeddings,
            "truth_boundary": "declared_model_config_not_full_graph_import",
        },
        "global_decisions": {}, "function_plans": [],
        "distributed": {
            "strategy": "tensor_parallel", "world_size": WORLD_SIZE,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE, "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
            "ranks": [{"rank_id": i, "logical_device": f"simulated_cpu_process_{i}"} for i in range(WORLD_SIZE)],
            "tensor_shards": tensor_shards, "collectives": collectives,
            "truth_boundary": "d1_simulated_localhost_multiprocess_ipc_not_real_gpu_not_nccl_not_measured_gpu_performance",
        },
    }
    return plan, work_items
