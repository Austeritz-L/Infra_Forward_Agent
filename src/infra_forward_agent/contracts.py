"""Versioned, JSON-compatible evidence contracts shared by all adapters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class Evidence:
    root: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Evidence:
        data = read_json(path)
        if data.get("schema_version") != 1:
            raise ValueError("Evidence schema_version must be 1")
        if data.get("kind") not in {"synthetic", "measured"}:
            raise ValueError("Evidence kind must be synthetic or measured")
        for field in (
            "id",
            "revision",
            "environment",
            "workload",
            "traces",
            "benchmark",
            "precision",
        ):
            if not data.get(field):
                raise ValueError(f"Missing evidence field: {field}")
        env = data["environment"]
        for field in ("image_digest", "hardware", "runtime", "topology", "profiler"):
            if not env.get(field):
                raise ValueError(f"Missing environment.{field}")
        if not isinstance(data["traces"], list):
            raise ValueError("traces must be a list")
        return cls(path.parent.resolve(), data)

    def artifact(self, relative: str) -> Path:
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("Artifact paths must be relative without parent traversal")
        if relative in {"evidence.json", "manifest.json"}:
            raise ValueError("Artifact path is reserved for evidence metadata")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Artifact must be an existing file under evidence root: {relative}")
        return path

    @property
    def id(self) -> str:
        return self.data["id"]


@dataclass(frozen=True)
class Policy:
    objective: str = "latency"
    min_gain: float = 0.03
    max_regression: float = 0.01
    min_samples: int = 5
    max_cv: float = 0.05

    def __post_init__(self) -> None:
        if self.objective not in {"latency", "throughput"}:
            raise ValueError("objective must be latency or throughput")
        for field in ("min_gain", "max_regression", "max_cv"):
            value = number(getattr(self, field), field)
            if value >= 1:
                raise ValueError(f"{field} must be below 1")
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reasons: list[str]
    metrics: dict[str, Any]
