"""Each run owns immutable evidence copies and an append-only streaming event log."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .contracts import Evidence


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class RunStore:
    def __init__(self, root: Path, sink: Callable[[dict], None] | None = None):
        # Never overwrite a previous experiment, including interrupted runs.
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self.sink = sink
        self.sequence = 0

    def emit(self, stage: str, **payload: object) -> None:
        self.sequence += 1
        event = {
            "schema_version": 1,
            "seq": self.sequence,
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **payload,
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        if self.sink:
            self.sink(event)

    def snapshot(self, evidence: Evidence, name: str) -> Evidence:
        root = self.root / "evidence" / name
        root.mkdir(parents=True)
        paths = [evidence.data["benchmark"], *evidence.data["precision"].values()]
        paths += [spec["path"] for spec in evidence.data["traces"]]
        manifest = {}
        for relative in sorted(set(paths)):
            source = evidence.artifact(relative)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(root / "evidence.json", evidence.data)
        manifest["evidence.json"] = hashlib.sha256(
            (root / "evidence.json").read_bytes()
        ).hexdigest()
        write_json(root / "manifest.json", manifest)
        return Evidence.load(root / "evidence.json")
