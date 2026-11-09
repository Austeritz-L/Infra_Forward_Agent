"""Future plugins must implement these interfaces; importing this module launches nothing."""

from pathlib import Path
from typing import Protocol

from ..contracts import Evidence


class Profiler(Protocol):
    def capture(self, task: dict, destination: Path) -> list[Path]:
        """Capture an explicitly selected forward on all owned ranks; return raw artifact paths."""
        ...


class Optimizer(Protocol):
    def propose(self, task: dict, analysis: dict, workspace: Path) -> dict:
        """Return a candidate plan, parent revision, patch path and expected validation commands."""
        ...


class Executor(Protocol):
    def preflight(self, task: dict) -> dict:
        """Verify image, weights, runtime, ownership, topology and budget on the live host."""
        ...

    def evaluate(self, candidate: dict, destination: Path) -> Evidence:
        """Apply in isolation; precision, unprofiled benchmark, precision, then capture a trace."""
        ...
