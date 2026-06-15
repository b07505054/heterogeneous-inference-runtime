from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    description: str


class ArtifactAccessError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path, allowed_artifacts: list[Artifact]) -> None:
        self.root = root
        self._allowed = {artifact.path: artifact for artifact in allowed_artifacts}

    def list_artifacts(self) -> list[dict]:
        return [
            {
                "path": artifact.path,
                "kind": artifact.kind,
                "description": artifact.description,
            }
            for artifact in self._allowed.values()
        ]

    def read_artifact(self, path: str) -> str:
        if path not in self._allowed:
            raise ArtifactAccessError(f"artifact is not allowlisted: {path}")

        artifact_path = (self.root / path).resolve()
        root = self.root.resolve()
        if root not in artifact_path.parents and artifact_path != root:
            raise ArtifactAccessError(f"artifact escapes root: {path}")

        return artifact_path.read_text(encoding="utf-8")

