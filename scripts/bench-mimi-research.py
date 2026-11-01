#!/usr/bin/env python3
"""Controlled delivery/kernel/scheduler ablations, with per-request arrival records."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

TEXTS = {
    "short": "Hello, how are you?",
    "release": "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "long": "Thank you for calling. I have checked the details of your reservation, and everything is ready for your arrival tomorrow afternoon. Please bring your confirmation number and a photo identification document. If your travel plans change, you can contact us to arrange a different arrival time.",
}
PACK = {"PTTS_EXPERIMENT_PACKED_MIMI": "1"}
SPLIT = {"PTTS_EXPERIMENT_FIRST_STEPS": "4"}
WAVE = {**SPLIT, "PTTS_EXPERIMENT_WAVEFORM_ONLY": "1"}
VARIANTS = {
    "packed-conv-small": {**PACK, "PTTS_EXPERIMENT_PACKED_CONV": "initial-small"},
    "conv-initial": {"PTTS_EXPERIMENT_PACKED_CONV": "initial"},
    "conv-all": {"PTTS_EXPERIMENT_PACKED_CONV": "all"},
    "packed-conv-initial": {**PACK, "PTTS_EXPERIMENT_PACKED_CONV": "initial"},
    "packed-conv-all": {**PACK, "PTTS_EXPERIMENT_PACKED_CONV": "all"},
    "baseline": {}, "packed": PACK, "split20": SPLIT, "wave20": WAVE,
    "packed-wave20": {**PACK, **WAVE},
    "packed-wave20-early": {**PACK, **WAVE, "PTTS_EXPERIMENT_EARLY_RELEASE": "1"},
    "inline": {"PTTS_EXPERIMENT_INLINE": "1"}, "spin0": {"XN_SPIN_BUDGET": "0"},
    "spin1000": {"XN_SPIN_BUDGET": "1000"}, "qos": {"PTTS_EXPERIMENT_QOS": "initiated"},
    "packed-qos": {**PACK, "PTTS_EXPERIMENT_QOS": "initiated"},
}
METRICS = ["ttfa_ms", "t20_ms", "t80_ms", "amplitude_onset_ms", "total_ms", "rtfx", "simulated_underrun_ms"]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    root, binary = args.root.resolve(), args.binary.resolve()
    model = root / "models/phonon-31m-en"
    selected = args.variants.split(",")
    texts = args.texts.split(",")
    for v in selected:
        assert v in VARIANTS, v
    for name in texts:
        assert name in TEXTS, name
    common = ["--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
              "--voice", str(model / "default-voice.safetensors"), "--quant", "q8", "--device", "cpu",
              "--threads", str(args.threads), "--lang", "en", "--seed", "42", "--warmup", "5", "--iters", str(args.iters)]
    if args.concurrency != 1:
        common += ["--concurrency", str(args.concurrency)]
    if args.runtime_diagnostics:
        common += ["--runtime-diagnostics"]
    metadata = {"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "platform": platform.platform(), "threads": args.threads, "trials": args.trials,
                "iterations": args.iters, "concurrency": args.concurrency, "runtime_diagnostics": args.runtime_diagnostics, "warmups_per_process": 5*args.concurrency, "texts": {t: TEXTS[t] for t in texts},
                "variants": {v: VARIANTS[v] for v in selected}, "runs": []}
    rows = []
    for trial in range(args.trials):
        for ti, name in enumerate(texts[trial % len(texts):] + texts[:trial % len(texts)]):
            offset = (trial + ti) % len(selected)
            order = selected[offset:] + selected[:offset]
            if trial % 2:
                order = list(reversed(order))
            for variant in order:
                stem = f"{name}-{trial}-{variant}"
                path = (args.out / (stem + ".json")).resolve()
                env = {k: v for k, v in os.environ.items() if not k.startswith("PTTS_EXPERIMENT_") and k != "XN_SPIN_BUDGET"}
                env.update(VARIANTS[variant])
                command = [str(binary), *common, "--input", TEXTS[name], "--metrics-json", str(path)]
                p = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (args.out / (stem + ".txt")).write_text(p.stdout)
                if p.returncode:
                    raise RuntimeError(stem + ": " + p.stdout)
                result = json.loads(path.read_text())
                raw = result["runs"]
                assert len(raw) == args.iters*args.concurrency
                for index, row in enumerate(raw):
                    rows.append({"text": name, "variant": variant, "trial": trial, "request": index, **row})
                metadata["runs"].append({"text": name, "variant": variant, "trial": trial, "command": command, "rounds": result.get("rounds", [])})
                (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
                (args.out / "observations.jsonl").write_text("".join(json.dumps(r, separators=(",", ":"))+"\n" for r in rows))
                print(stem, {k: round(statistics.median(r[k] for r in raw), 3) for k in ["t20_ms", "t80_ms", "total_ms"]}, flush=True)
    summary = {}
    for name in texts:
        summary[name] = {}
        for variant in selected:
            summary[name][variant] = {}
            for metric in METRICS:
                medians, tails = [], []
                for trial in range(args.trials):
                    values = sorted(r[metric] for r in rows if r["text"] == name and r["variant"] == variant and r["trial"] == trial and r[metric] is not None)
                    assert values, (name, variant, metric)
                    medians.append(statistics.median(values))
                    tails.append(values[math.ceil(len(values)*.95)-1])
                summary[name][variant][metric] = {"p50": statistics.median(medians), "p95": statistics.median(tails), "trial_p50": medians, "trial_p95": tails}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "binary", "out"]:
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--variants", default="baseline,packed,split20,wave20,packed-wave20,packed-wave20-early")
    parser.add_argument("--texts", default="short,release,long")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--runtime-diagnostics", action="store_true")
    run(parser.parse_args())
