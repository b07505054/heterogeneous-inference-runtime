"""Registry for neutral model adapters."""

from __future__ import annotations

from typing import TypeVar

from deployment.model_adapter.base import ModelAdapter
from deployment.model_adapter.mock_adapter import MockModelAdapter


AdapterT = TypeVar("AdapterT", bound=ModelAdapter)

_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(kind: str, adapter_cls: type[AdapterT]) -> None:
    """Register an adapter class for a neutral artifact kind."""
    normalized = _normalize_kind(kind)
    if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, ModelAdapter):
        raise TypeError("adapter_cls must be a ModelAdapter subclass")
    _ADAPTERS[normalized] = adapter_cls


def create_adapter(kind: str, **kwargs) -> ModelAdapter:
    """Instantiate an adapter for a registered artifact kind."""
    normalized = _normalize_kind(kind)
    adapter_cls = _ADAPTERS.get(normalized)
    if adapter_cls is None:
        available = ", ".join(list_adapters()) or "none"
        raise ValueError(
            f"unknown model adapter kind '{kind}'. Registered adapters: {available}"
        )
    return adapter_cls(**kwargs)


def list_adapters() -> list[str]:
    """Return registered adapter kinds in stable order."""
    return sorted(_ADAPTERS)


def _normalize_kind(kind: str) -> str:
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("adapter kind must be a non-empty string")
    return kind.strip().lower()


register_adapter("mock", MockModelAdapter)
