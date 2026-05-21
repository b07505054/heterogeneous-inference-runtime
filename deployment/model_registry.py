import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(
        self,
        registry_path: str = "configs/model_registry.json",
    ):
        self.registry_path = Path(registry_path)
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.registry_path}"
            )

        return json.loads(
            self.registry_path.read_text(encoding="utf-8")
        )

    def active_model_name(self) -> str:
        return self.data["active_model"]

    def active_model_config(self) -> dict[str, Any]:
        return self.get_model(self.active_model_name())

    def get_model(self, name: str) -> dict[str, Any]:
        models = self.data.get("models", {})

        if name not in models:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(models.keys())}"
            )

        return models[name]

    def list_models(self) -> list[str]:
        return list(self.data.get("models", {}).keys())