#!/usr/bin/env python3
"""Summarize first occurrences of host spans in sequential bench requests.

These are inclusive host durations, not CUDA kernel timings. Nested spans must
not be added together. Traced runs are diagnostics, never speedup measurements.
"""
import argparse
import collections
import json
import statistics


def summarize(events):
    stacks = collections.defaultdict(list)
    spans = []
    markers = []
    for event in events:
        phase = event.get("ph")
        if phase == "B":
            stacks[event["tid"]].append(event)
        elif phase == "E":
            start = stacks[event["tid"]].pop()
            assert start["name"] == event["name"]
            spans.append((start["name"], start["ts"], event["ts"]))
        elif event.get("args", {}).get("message") == "first_audio_received":
            markers.append(event["ts"])
    assert not any(stacks.values()), "incomplete trace"
    requests = sorted(span for span in spans if span[0] == "request")
    stages = collections.defaultdict(list)
    for _, start, end in requests:
        first = next(t for t in markers if start <= t <= end)
        stages["host_ttfa"].append((first - start) / 1000)
        seen = set()
        for name, a, b in sorted(spans, key=lambda span: span[1]):
            if name in {"request", "warmup"} or name in seen:
                continue
            if start <= a < first and b <= first:
                seen.add(name)
                stages[name].append((b - a) / 1000)
    return {
        "note": "Inclusive host spans; nested durations overlap. Diagnostic only.",
        "measured_requests": len(requests),
        "stages": {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in stages.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    with open(args.trace) as source:
        result = summarize(json.load(source))
    print(json.dumps(result, indent=2))
