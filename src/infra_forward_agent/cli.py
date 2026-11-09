from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .analysis import analyze
from .architectures import ArchitectureRegistry
from .campaign import write_plan
from .contracts import Evidence, Policy
from .demo import make_demo
from .engine import replay
from .evaluation import compare
from .performance.demo import make_performance_demo
from .performance.report import write_report
from .trace import load_chrome


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infra Forward Agent: trace/evidence workflow baseline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("architectures", help="List architecture targets and unverified capabilities")
    plan = sub.add_parser(
        "plan", help="Plan five domains across architectures; launches no GPU jobs"
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    performance = sub.add_parser(
        "performance", help="Analyze imported prefill statistics, resource ceilings and overlap"
    )
    performance.add_argument("bundle", type=Path)
    performance.add_argument("--output", type=Path, required=True)
    perf_demo = sub.add_parser(
        "performance-demo", help="Synthetic prefill bottleneck investigation; no GPU"
    )
    perf_demo.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Replay synthetic evidence; requires no GPU or model API")
    demo.add_argument("--output", type=Path, required=True)
    fixtures = sub.add_parser("fixtures", help="Write editable synthetic input evidence")
    fixtures.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Run a bounded artifact replay with JSONL streaming events")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    trace = sub.add_parser("analyze", help="Analyze one selected forward in Chrome JSON[.gz]")
    trace.add_argument("trace", type=Path)
    trace.add_argument("--rank", type=int, default=0)
    trace.add_argument("--marker", default="forward")
    comparison = sub.add_parser(
        "compare", help="Check precision and forward metrics in two bundles"
    )
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--objective", choices=["latency", "throughput"], default="latency")
    args = parser.parse_args(argv)
    try:
        if args.command == "architectures":
            emit(
                {
                    "architectures": [
                        profile.to_dict() for profile in ArchitectureRegistry.builtin().all()
                    ]
                }
            )
        elif args.command == "plan":
            write_plan(args.config, args.output, emit)
        elif args.command == "performance":
            write_report(args.bundle, args.output, emit)
        elif args.command == "performance-demo":
            with tempfile.TemporaryDirectory(prefix="ifa-performance-") as directory:
                write_report(make_performance_demo(Path(directory)), args.output, emit)
        elif args.command == "demo":
            with tempfile.TemporaryDirectory(prefix="ifa-demo-") as directory:
                replay(make_demo(Path(directory)), args.output, emit)
        elif args.command == "fixtures":
            if args.output.exists():
                raise ValueError("Fixture destination already exists")
            print(make_demo(args.output))
        elif args.command == "run":
            replay(args.config, args.output, emit)
        elif args.command == "analyze":
            emit(analyze(load_chrome(args.trace, rank=args.rank, marker=args.marker)))
        elif args.command == "compare":
            from dataclasses import asdict

            result = compare(
                Evidence.load(args.baseline),
                Evidence.load(args.candidate),
                Policy(objective=args.objective),
            )
            emit(asdict(result))
            return 0 if result.accepted else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ifa: {error}", file=sys.stderr)
        return 1
    return 0
