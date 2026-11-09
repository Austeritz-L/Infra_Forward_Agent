from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..architectures import ArchitectureProfile


@dataclass(frozen=True)
class PlanningContext:
    architecture: ArchitectureProfile
    scenarios: list[dict]
    deployment: dict
    traces: dict
    knowledge: list[str]
    testing_contract: dict = field(default_factory=dict)
    performance_contract: dict = field(default_factory=dict)
    deployment_contract: dict = field(default_factory=dict)
    kernel_contract: dict = field(default_factory=dict)
    trace_contract: dict = field(default_factory=dict)

    def task_id(self, stage: str) -> str:
        return f"{self.architecture.id}:{stage}"


@dataclass(frozen=True)
class StagePlan:
    id: str
    module: str
    architecture: str
    depends_on: list[str]
    scope: str
    required_inputs: list[str]
    expected_outputs: list[str]
    status: str = "needs_adapter"
    options: dict = field(default_factory=dict)


class PlanningAgent(Protocol):
    def plan(self, context: PlanningContext) -> StagePlan: ...
