from ..contracts import number


def estimate_communication(phases: list[dict]) -> list[dict]:
    """Per-rank phase estimates only. Never add overlap or infer TTFT from wire volume."""
    rows = []
    for phase in phases:
        if not phase.get("source") or phase.get("byte_convention") != "tx_wire_bytes_per_rank":
            raise ValueError(
                "Communication estimates require a source and explicit per-rank wire units"
            )
        wire = number(phase["wire_bytes"], "wire bytes")
        messages = phase["messages"]
        if type(messages) is not int or messages < 0:
            raise ValueError("Communication message count must be nonnegative")
        latency, bandwidth = phase.get("startup_us"), phase.get("bandwidth_bytes_per_s")
        estimate = None
        if latency is not None and bandwidth is not None:
            estimate = (
                messages * number(latency, "startup latency")
                + wire / number(bandwidth, "reference bandwidth", positive=True) * 1e6
            )
        rows.append(
            {
                **phase,
                "reference_phase_us": estimate,
                "is_measured_end_to_end_latency": False,
                "limitations": "Match size/group/algorithm/link; waits and contention omitted",
            }
        )
    return rows
