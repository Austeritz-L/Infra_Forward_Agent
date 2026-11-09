from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from ..contracts import read_json


@dataclass(frozen=True)
class ArchitectureProfile:
    id: str
    vendor: str
    family: str
    runtime: str
    status: str
    capabilities: dict
    resources: dict
    verification_required: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> ArchitectureProfile:
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Architecture profile must be an object with schema_version=1")
        fields = ("id", "vendor", "family", "runtime", "status")
        if any(not isinstance(data.get(key), str) or not data[key].strip() for key in fields):
            raise ValueError("Architecture identity and status must be nonempty strings")
        if data["status"] not in {"planned", "experimental", "qualified"}:
            raise ValueError("Architecture status must be planned, experimental or qualified")
        for key in ("capabilities", "resources"):
            if not isinstance(data.get(key), dict):
                raise ValueError(f"Architecture {key} must be an object")
        checks = data.get("verification_required")
        if (
            not isinstance(checks, list)
            or not checks
            or not all(isinstance(x, str) for x in checks)
        ):
            raise ValueError("Architecture must list verification_required checks")
        return cls(
            **{
                key: data[key]
                for key in (*fields, "capabilities", "resources", "verification_required")
            }
        )

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}


class ArchitectureRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ArchitectureProfile] = {}

    @classmethod
    def builtin(cls) -> ArchitectureRegistry:
        registry = cls()
        directory = files("infra_forward_agent.architectures").joinpath("profiles")
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.name.endswith(".json"):
                registry.register(
                    ArchitectureProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
                )
        return registry

    def register(self, profile: ArchitectureProfile) -> None:
        if profile.id in self._profiles:
            raise ValueError(f"Duplicate architecture id: {profile.id}")
        self._profiles[profile.id] = profile

    def load(self, path: Path) -> None:
        self.register(ArchitectureProfile.from_dict(read_json(path)))

    def get(self, architecture: str) -> ArchitectureProfile:
        if architecture not in self._profiles:
            raise ValueError(f"Unknown architecture: {architecture}; register a profile first")
        return self._profiles[architecture]

    def all(self) -> list[ArchitectureProfile]:
        return [self._profiles[key] for key in sorted(self._profiles)]
