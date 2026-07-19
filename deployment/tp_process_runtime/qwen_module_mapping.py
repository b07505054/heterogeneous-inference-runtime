"""D3A Part B: fail-closed mapping between a compiler distributed-plan
operator ID (e.g. "qwen_prefill::llm.o_proj::layer_0") and the real
Transformers module that implements it.

Never selects a module by substring match. Every candidate module is found
by scanning the live model's named_modules() against an explicit structural
regex, and every required check (layer number, operator kind, weight shape,
uniqueness) is verified before the mapping is accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# compiler operator_id format: "{function_name}::{op_type}::layer_{layer_index}"
_OPERATOR_ID_RE = re.compile(r"^(?P<function_name>[^:]+)::(?P<op_type>[^:]+)::layer_(?P<layer_index>\d+)$")

# Only operator kinds this module knows how to map. Extending this table is
# the only way to add a new operator -- never a generic substring fallback.
#
# D4A additive extension: q_proj/k_proj/v_proj/gate_proj/up_proj/down_proj
# are added alongside D3A's original o_proj entry so the whole-model TP
# contract validation (D4A) can map every per-layer linear operator family
# to its real Transformers module, using the exact same structural-regex,
# fail-closed mapping discipline as o_proj. No existing entry or behavior
# is changed.
_OP_TYPE_TO_MODULE_SUFFIX = {
    "llm.o_proj": "self_attn.o_proj",
    "llm.q_proj": "self_attn.q_proj",
    "llm.k_proj": "self_attn.k_proj",
    "llm.v_proj": "self_attn.v_proj",
    "llm.gate_proj": "mlp.gate_proj",
    "llm.up_proj": "mlp.up_proj",
    "llm.down_proj": "mlp.down_proj",
}
_OP_TYPE_TO_MODULE_PATTERN = {
    "llm.o_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.self_attn\.o_proj$"),
    "llm.q_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.self_attn\.q_proj$"),
    "llm.k_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.self_attn\.k_proj$"),
    "llm.v_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.self_attn\.v_proj$"),
    "llm.gate_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.mlp\.gate_proj$"),
    "llm.up_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.mlp\.up_proj$"),
    "llm.down_proj": re.compile(r"^model\.layers\.(?P<layer_index>\d+)\.mlp\.down_proj$"),
}

# o_proj/down_proj are square-shaped ([hidden,hidden] or check against plan);
# q/k/v/gate/up are rectangular ([out,in] with out != in in general). The
# original o_proj-only mapping enforced a square-weight check
# unconditionally; D4A widens that check to apply only to operators that are
# actually expected to be square, so the new rectangular families are not
# rejected by an o_proj-specific assumption.
_SQUARE_WEIGHT_OP_TYPES = {"llm.o_proj"}


class OperatorMappingError(ValueError):
    """Fail-closed: the compiler operator ID could not be uniquely and
    verifiably mapped to a real Transformers module."""


@dataclass(frozen=True)
class OperatorMappingResult:
    operator_id: str
    function_name: str
    op_type: str
    layer_index: int
    module_path: str
    module_class: str
    weight_shape: tuple[int, ...]
    bias_present: bool
    checks: dict[str, Any] = field(default_factory=dict)


def parse_operator_id(operator_id: str) -> tuple[str, str, int]:
    m = _OPERATOR_ID_RE.match(operator_id)
    if not m:
        raise OperatorMappingError(
            f"compiler operator ID {operator_id!r} does not match the expected "
            "'<function_name>::<op_type>::layer_<n>' format"
        )
    return m.group("function_name"), m.group("op_type"), int(m.group("layer_index"))


def map_compiler_operator_to_module(
    operator_id: str, model: Any, *, expected_hidden_size: int | None = None,
) -> OperatorMappingResult:
    """Fail-closed compiler-operator-ID -> Transformers-module mapping.

    Raises OperatorMappingError for: unknown operator kind, no matching
    module, ambiguous match (more than one module structurally matches this
    op_type+layer), incorrect layer number, incorrect module class, or a
    weight shape that disagrees with the plan's declared hidden dimension.
    """
    function_name, op_type, layer_index = parse_operator_id(operator_id)

    if op_type not in _OP_TYPE_TO_MODULE_SUFFIX:
        raise OperatorMappingError(
            f"unknown operator kind {op_type!r}; no compiler-operator-to-Transformers-"
            f"module mapping is declared for it (known kinds: {sorted(_OP_TYPE_TO_MODULE_SUFFIX)})"
        )
    pattern = _OP_TYPE_TO_MODULE_PATTERN[op_type]

    named = dict(model.named_modules())
    structural_matches = []
    for name in named:
        m = pattern.match(name)
        if m:
            structural_matches.append((name, int(m.group("layer_index"))))

    if not structural_matches:
        raise OperatorMappingError(
            f"compiler operator ID {operator_id!r} maps to no Transformers module: "
            f"no module name in this model matches the structural pattern {pattern.pattern!r}"
        )

    same_layer = [(n, li) for n, li in structural_matches if li == layer_index]
    if not same_layer:
        available_layers = sorted({li for _, li in structural_matches})
        raise OperatorMappingError(
            f"compiler operator ID {operator_id!r} declares layer_index={layer_index}, "
            f"but no matching module exists at that layer (available layers: {available_layers})"
        )
    if len(same_layer) > 1:
        raise OperatorMappingError(
            f"compiler operator ID {operator_id!r} maps ambiguously: {len(same_layer)} "
            f"modules match layer {layer_index}: {[n for n, _ in same_layer]}"
        )

    module_path = same_layer[0][0]
    module = named[module_path]

    module_class = f"{type(module).__module__}.{type(module).__name__}"
    if type(module).__name__ != "Linear":
        raise OperatorMappingError(
            f"module {module_path!r} has unexpected class {module_class!r}; "
            "expected torch.nn.modules.linear.Linear for operator kind 'llm.o_proj'"
        )

    weight = getattr(module, "weight", None)
    if weight is None:
        raise OperatorMappingError(f"module {module_path!r} has no 'weight' attribute")
    weight_shape = tuple(weight.shape)
    if op_type in _SQUARE_WEIGHT_OP_TYPES and weight_shape[0] != weight_shape[1]:
        raise OperatorMappingError(
            f"module {module_path!r} weight shape {weight_shape} is not square; "
            f"{op_type} is expected to be a hidden_size x hidden_size projection"
        )
    if expected_hidden_size is not None and weight_shape[0] != expected_hidden_size:
        raise OperatorMappingError(
            f"module {module_path!r} weight shape {weight_shape} does not match the "
            f"compiler plan's declared hidden dimension {expected_hidden_size}"
        )

    bias = getattr(module, "bias", None)

    checks = {
        "layer_number_matches": True,
        "operator_kind_matches": True,
        "weight_shape_matches_plan": expected_hidden_size is None or weight_shape[0] == expected_hidden_size,
        "module_appears_exactly_once": len(same_layer) == 1,
        "structural_match_count_all_layers": len(structural_matches),
    }

    return OperatorMappingResult(
        operator_id=operator_id,
        function_name=function_name,
        op_type=op_type,
        layer_index=layer_index,
        module_path=module_path,
        module_class=module_class,
        weight_shape=weight_shape,
        bias_present=bias is not None,
        checks=checks,
    )
