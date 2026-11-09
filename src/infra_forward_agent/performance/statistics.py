from __future__ import annotations

import statistics

from ..contracts import number


def summarize(values: list[float]) -> dict:
    if not isinstance(values, list) or not values:
        raise ValueError("Samples must be a nonempty list")
    ordered = sorted(number(value, "sample", positive=True) for value in values)

    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "cv": statistics.stdev(ordered) / statistics.mean(ordered) if len(ordered) > 1 else None,
        "p95_sample_eligible": len(ordered) >= 200,
        "p99_sample_eligible": len(ordered) >= 1000,
    }


def service_statistics(report: dict) -> dict:
    """One matched condition, including all attempts and a full timed-load horizon if applicable."""
    if report.get("profiled") is not False or report.get("cohort") not in {
        "fixed_work_drained",
        "timed_offer_drained",
    }:
        raise ValueError("Service statistics require an unprofiled drained cohort")
    if not report.get("clock_id"):
        raise ValueError("Service cohort must identify its monotonic clock")
    rows = report.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Service cohort must contain requests")
    seen = set()
    valid = []
    starts, ends = [], []
    failures = []
    for row in rows:
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Duplicate service request id")
        if row.get("clock_id") != report["clock_id"]:
            raise ValueError("Normalize or separate worker clock domains before pooling a cohort")
        seen.add(identifier)
        start = number(row["sent_us"], "sent_us")
        end = number(row["terminal_us"], "terminal_us")
        if end <= start:
            raise ValueError("Every submitted request needs a terminal or cutoff time after send")
        starts.append(start)
        ends.append(end)
        if (
            row.get("status") != "completed"
            or row.get("done") is not True
            or row.get("errors") != []
        ):
            failures.append(identifier)
            continue
        first = number(row["first_token_us"], "first_token_us")
        if (
            not start < first <= end
            or type(row.get("output_tokens")) is not int
            or row["output_tokens"] != 1
        ):
            failures.append(identifier)
            continue
        for field in ("input_tokens", "cached_tokens", "new_tokens"):
            if type(row[field]) is not int:
                raise ValueError(f"{field} must be an integer token count")
        logical = number(row["input_tokens"], "input_tokens", positive=True)
        cached = number(row["cached_tokens"], "cached_tokens")
        new = number(row["new_tokens"], "new_tokens")
        if cached + new != logical:
            raise ValueError("Normalized owner token accounting must satisfy cached + new = input")
        valid.append(
            {
                "ttft_us": first - start,
                "e2e_us": end - start,
                "logical": logical,
                "new": new,
                "cached": cached,
            }
        )
    raw_cohort_s = (max(ends) - min(starts)) / 1e6
    elapsed_s = raw_cohort_s
    offer = None
    if report["cohort"] == "timed_offer_drained":
        window = report["offered_window"]
        begin = number(window["started_us"], "offered_window.started_us")
        finish = number(window["finished_us"], "offered_window.finished_us")
        if finish <= begin or min(starts) < begin or max(starts) >= finish:
            raise ValueError(
                "Timed cohort must contain exactly the requests sent inside the offered window"
            )
        elapsed_s = (max(max(ends), finish) - begin) / 1e6
        offer = {
            "seconds": (finish - begin) / 1e6,
            "submitted_per_s": len(rows) / ((finish - begin) / 1e6),
            "window_completed": window.get("completed") is True,
            "formal_plus_drain_seconds": elapsed_s,
        }
    changes = {}
    for start, end in zip(starts, ends):
        changes[start] = changes.get(start, 0) + 1
        changes[end] = changes.get(end, 0) - 1
    active, peak = 0, 0
    for _, delta in sorted(changes.items()):
        active += delta
        peak = max(peak, active)
    return {
        "submitted": len(rows),
        "completed": len(valid),
        "failed_or_incomplete": failures,
        "cohort_seconds": elapsed_s,
        "raw_first_send_to_last_terminal_seconds": raw_cohort_s,
        "timed_offered_window": offer,
        "client_observed_peak_inflight": peak,
        "request_integrity_valid": not failures,
        "ttft_us": summarize([row["ttft_us"] for row in valid]) if valid else None,
        "e2e_us": summarize([row["e2e_us"] for row in valid]) if valid else None,
        "logical_input_tokens_per_s": sum(row["logical"] for row in valid) / elapsed_s,
        "new_prefill_tokens_per_s": sum(row["new"] for row in valid) / elapsed_s,
        "requests_per_s": len(valid) / elapsed_s,
        "cached_token_fraction": (
            sum(row["cached"] for row in valid) / sum(row["logical"] for row in valid)
        )
        if valid
        else None,
        "limitations": [
            "Throughput numerators use complete successes; time includes all attempts.",
            "A failed cohort remains invalid; statistics do not make it a passing result.",
            "TTFT includes service overhead, not just model forward.",
        ],
    }
