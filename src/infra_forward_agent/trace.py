"""A deliberately small Chrome complete-event adapter, with explicit boundaries."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import number, read_json

GPU_CATEGORIES = {"kernel", "gpu_kernel", "musa_kernel", "gpu_memcpy", "gpu_memset"}


@dataclass(frozen=True)
class Slice:
    name: str
    rank: int
    device: str
    stream: str
    start_us: float
    end_us: float
    kind: str
    classification: str
    operator: str | None = None
    compute_type: str = "unknown"
    signature: str = "unknown"
    invocation: str | None = None
    layer: str = "unknown"


@dataclass(frozen=True)
class Trace:
    slices: list[Slice]
    windows: dict[int, tuple[float, float]]
    ignored_events: int
    clipped_events: int = 0


def load_chrome(path: Path, *, rank: int = 0, marker: str = "forward") -> Trace:
    """Import one forward/rank. Chrome ts/dur are in us regardless of displayTimeUnit."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            # Validate numeric values below; gzip JSON has the same contract as plain JSON.
            data = json.load(handle)
    else:
        data = read_json(path)
    events = data["traceEvents"] if isinstance(data, dict) else data
    if not isinstance(events, list):
        raise ValueError("traceEvents must be a list")
    windows: dict[int, tuple[float, float]] = {}
    raw: list[Slice] = []
    ignored = 0
    for event in events:
        if event.get("ph") != "X":
            ignored += 1
            continue
        name = str(event.get("name", ""))
        categories = set(str(event.get("cat", "")).lower().split(","))
        if name != marker and not categories.intersection(GPU_CATEGORIES):
            ignored += 1
            continue
        args = event.get("args", {})
        event_rank = args.get("rank", rank)
        if type(event_rank) is not int or event_rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        start = number(event.get("ts"), "event.ts")
        end = number(start + number(event.get("dur"), "event.dur", positive=True), "event.end")
        if name == marker:
            if event_rank in windows:
                raise ValueError(f"Multiple forward windows for rank {event_rank}; select one step")
            windows[event_rank] = (start, end)
            continue
        hint = args.get("ifa_kind")
        if hint is not None and hint not in {"compute", "communication", "memory"}:
            raise ValueError("ifa_kind must be compute, communication, or memory")
        lower = name.lower()
        if hint:
            kind, source = hint, "explicit"
        elif "gpu_memcpy" in categories or "gpu_memset" in categories:
            kind, source = "memory", "category"
        elif any(token in lower for token in ("nccl", "mccl", "rccl")):
            kind, source = "communication", "name_heuristic"
        else:
            kind, source = "compute", "category_default"
        device = str(args.get("device", event.get("pid", "unknown")))
        metadata = [
            args.get("ifa_operator"),
            args.get("ifa_compute_type", "unknown"),
            args.get("ifa_signature", "unknown"),
            args.get("ifa_invocation"),
            str(args.get("ifa_layer", "unknown")),
        ]
        if any(value is not None and not isinstance(value, str) for value in metadata):
            raise ValueError("ifa_operator/compute_type/signature/invocation must be strings")
        raw.append(
            Slice(
                name,
                event_rank,
                device,
                str(args.get("stream", event.get("tid", "unknown"))),
                start,
                end,
                kind,
                source,
                *metadata,
            )
        )
    clipped = []
    clipping_count = 0
    for item in raw:
        if item.rank not in windows:
            raise ValueError(f"Missing {marker!r} window for rank {item.rank}")
        start, end = windows[item.rank]
        begin, finish = max(start, item.start_us), min(end, item.end_us)
        if finish > begin:
            clipping_count += begin != item.start_us or finish != item.end_us
            clipped.append(
                Slice(
                    item.name,
                    item.rank,
                    item.device,
                    item.stream,
                    begin,
                    finish,
                    item.kind,
                    item.classification,
                    item.operator,
                    item.compute_type,
                    item.signature,
                    item.invocation,
                    item.layer,
                )
            )
        else:
            ignored += 1
    if not clipped:
        raise ValueError("No GPU complete events inside a forward window")
    return Trace(clipped, windows, ignored, clipping_count)
