"""Base interface for model artifact adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from deployment.model_adapter.neutral_runtime_graph import NeutralRuntimeGraph


class ModelAdapter(ABC):
    """Translate one model artifact source into a neutral runtime graph."""

    @abstractmethod
    def load(self) -> NeutralRuntimeGraph:
        """Return a neutral graph without changing runtime or benchmark state."""

    def validate(self) -> list[str]:
        """Validate adapter output and return symbolic errors."""
        try:
            return self.load().validate()
        except Exception as exc:
            return [f"adapter_load_failed:{type(exc).__name__}"]
