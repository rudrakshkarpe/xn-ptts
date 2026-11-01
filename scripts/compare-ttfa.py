#!/usr/bin/env python3
"""Paired, alternating native Synth benchmarks; no trace or audio capture in timed runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess

TEXTS = {
    "short": "Hello, how are you?",
    "release": "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "long": "Thank you for calling. I have checked the details of your reservation, and everything is ready for your arrival tomorrow afternoon. Please bring your confirmation number and a photo identification document. If your travel plans change, you can contact us to arrange a different arrival time.",
}


def run(args):
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    model = root / "models/phonon-31m-en"
    shared = ["--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
              "--voice", str(model / "default-voice.safetensors"), "--quant", "q8", "--device", "cpu",
              "--threads", str(args.threads), "--lang", "en", "--seed", "42", "--warmup", "5",
              "--iters", str(args.iters), "--per-iter"]
    metadata = {"platform": platform.platform(), "threads": args.threads, "seed": 42,
                "XN_KAI": os.environ.get("XN_KAI", "default"), "texts": TEXTS,
                "binaries": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                             for k, v in binaries.items()}, "runs": []}
    for trial in range(args.trials):
        names = list(TEXTS)
        names = names[trial % len(names):] + names[:trial % len(names)]
        for name in names:
            order = list(binaries) if trial % 2 == 0 else list(reversed(binaries))
            for variant in order:
                command = [str(binaries[variant]), *shared, "--input", TEXTS[name]]
                proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stem = f"{name}-{trial}-{variant}"
                (args.out / f"{stem}.txt").write_text(proc.stdout)
                if proc.returncode:
                    raise RuntimeError(f"{stem} failed: {proc.stdout}")
                rows = re.findall(r"iter\s+\d+: total\s+([\d.]+)ms\s+ttfa\s+([\d.]+)ms\s+frames\s+(\d+)", proc.stdout)
                assert len(rows) == args.iters, (stem, rows)
                values = [{"total_ms": float(t), "ttfa_ms": float(f), "frames": int(n)} for t, f, n in rows]
                metadata["runs"].append({"text": name, "trial": trial, "variant": variant,
                                          "command": command, "values": values})
                print(stem, "ttfa", statistics.median(v["ttfa_ms"] for v in values), flush=True)
                (args.out / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {}
    for name in TEXTS:
        summary[name] = {}
        for variant in binaries:
            runs = [r for r in metadata["runs"] if r["text"] == name and r["variant"] == variant]
            trial_medians = [statistics.median(v["ttfa_ms"] for v in r["values"]) for r in runs]
            summary[name][variant] = {"trial_ttfa_p50_ms": trial_medians,
                                      "median_trial_ttfa_p50_ms": statistics.median(trial_medians),
                                      "median_trial_total_p50_ms": statistics.median(
                                          statistics.median(v["total_ms"] for v in r["values"]) for r in runs)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "baseline", "candidate", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    run(parser.parse_args())
