#!/usr/bin/env python3
"""Summarize an xctrace time-profile XML export; CPU samples are not TTFA fractions."""
import argparse
import collections
import json
import xml.etree.ElementTree as ET


def summarize(path):
    root = ET.parse(path).getroot()
    ids = {element.get("id"): element for element in root.iter() if element.get("id")}

    def resolve(element):
        return ids[element.get("ref")] if element.get("ref") else element

    rows = []
    for row in root.iter("row"):
        backtrace = row.find("backtrace")
        if backtrace is None:
            continue
        frames = [resolve(frame) for frame in resolve(backtrace)]
        if not frames:
            continue
        binary = frames[0].find("binary")
        binary = resolve(binary).get("name") if binary is not None else "unknown"
        rows.append((int(resolve(row.find("sample-time")).text),
                     int(resolve(row.find("weight")).text),
                     [frame.get("name", "unknown") for frame in frames], binary))
    start = min(t for t, _, frames, _ in rows
                if any("run_backbone" in f or "run_inline" in f for f in frames))
    leaves, inclusive, binaries = (collections.Counter() for _ in range(3))
    total = 0
    for time, weight, frames, binary in rows:
        if time < start:
            continue
        total += weight
        leaves[frames[0]] += weight
        binaries[binary] += weight
        for frame in set(frames):
            inclusive[frame] += weight

    def percentages(counter):
        return [{"symbol": name, "percent": weight/total*100} for name, weight in counter.most_common()]

    return {"note": "CPU sample weights after first generation stack, including warmup; not wall-time or TTFA attribution. Inclusive stacks overlap.",
            "first_generation_seconds": start/1e9, "total_sample_weight_ms": total/1e6,
            "leaf": percentages(leaves)[:30], "binaries": percentages(binaries),
            "ptts_inclusive": [row for row in percentages(inclusive) if "ptts::" in row["symbol"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml")
    print(json.dumps(summarize(parser.parse_args().xml), indent=2))
