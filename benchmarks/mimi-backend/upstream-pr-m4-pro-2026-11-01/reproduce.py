#!/usr/bin/env python3
"""Validate the focused upstream PR using bench's saved arrival/PCM outputs."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import wave

TEXTS = {
    "short": "Hello, how are you?",
    "release": "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "long": "Thank you for calling. I have checked the details of your reservation, and everything is ready for your arrival tomorrow afternoon. Please bring your confirmation number and a photo identification document. If your travel plans change, you can contact us to arrange a different arrival time.",
}
QUALITY = [
    "Hey Gradium team, this is Rudraksh loving the phonon so far!",
    TEXTS["short"], TEXTS["release"],
    "Your appointment is at 10:30 tomorrow morning, and the total comes to 42 dollars.",
    "Please confirm your delivery address. I will then check the available delivery dates for you.",
]
VARIANTS = {
    "baseline": {},
    "packed": {"PTTS_EXPERIMENT_PACKED_MIMI": "1"},
    "combined": {"PTTS_EXPERIMENT_PACKED_MIMI": "1", "PTTS_EXPERIMENT_PACKED_CONV": "initial-small"},
}

def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def compare(a, b):
    if len(a) != len(b):
        return {"same_length": False, "reference_samples": len(a), "candidate_samples": len(b)}
    assert a and all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
    errors = [float(x) - float(y) for x, y in zip(a, b)]
    mse = sum(x*x for x in errors) / len(a)
    power = sum(float(x)*float(x) for x in a) / len(a)
    return {"same_length": True, "samples": len(a), "max_abs_error": max(map(abs, errors)),
            "rmse": math.sqrt(mse), "snr_db": 10 * math.log10(power/mse) if mse else None}

def main(args):
    root, binary, source = args.root.resolve(), args.binary.resolve(), args.source.resolve()
    model = root / "models/phonon-31m-en"
    voices = [model / "default-voice.safetensors", *[v.resolve() for v in args.voice]]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "binary": str(binary), "binary_sha256": sha(binary), "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "platform": platform.platform(), "threads": 4, "seed": 42, "mode": args.mode,
        "variants": VARIANTS, "assets": {str(p): sha(p) for p in [model / "model.q8.gguf", model / "config.json", model / "tokenizer.json", *voices]},
        "commands": [],
    }
    def invoke(label, text, voice, stem, warmup, iters, export):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PTTS_EXPERIMENT_") and k != "XN_SPIN_BUDGET"}
        env.update({"XN_KAI": "1", "RAYON_NUM_THREADS": "4", **VARIANTS.get(label, {})})
        path = out / (stem + (".json" if export == "--metrics-json" else ".f32"))
        cmd = [str(binary), "--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
               "--voice", str(voice), "--quant", "q8", "--cpu", "--threads", "4", "--lang", "en",
               "--seed", "42", "--warmup", str(warmup), "--iters", str(iters), "--input", text, export, str(path)]
        manifest["commands"].append({"variant": label, "command": cmd, "env_overrides": VARIANTS.get(label, {})})
        write(out / "manifest.json", manifest)
        p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        (out / (stem + ".log")).write_text(p.stdout)
        return path
    if args.mode == "timing":
        observations, trial_stats = [], []
        labels = list(VARIANTS)
        texts = list(TEXTS)
        for trial in range(3):
            for name in texts[trial:] + texts[:trial]:
                order = labels[trial:] + labels[:trial]
                if trial % 2:
                    order = order[::-1]
                for label in order:
                    p = invoke(label, TEXTS[name], voices[0], f"{name}-{trial}-{label}", 5, 20, "--metrics-json")
                    data = json.loads(p.read_text())
                    ttfas, speeds = [], []
                    for index, r in enumerate(data["runs"]):
                        assert len(data["runs"]) == 20 and r["samples"] > 0
                        assert r["arrivals"][0][1] >= data["sample_rate"] * 0.08
                        assert r["arrivals"][0][0] == r["ttfa_ms"]
                        assert sum(n for _, n in r["arrivals"]) == r["samples"]
                        speed = r["samples"] / data["sample_rate"] * 1000 / r["total_ms"]
                        observations.append({"text": name, "trial": trial, "variant": label, "iteration": index,
                                             **r, "rtfx": speed})
                        ttfas.append(r["ttfa_ms"]); speeds.append(speed)
                    trial_stats.append({"text": name, "trial": trial, "variant": label,
                        "ttfa_p50_ms": statistics.median(ttfas), "ttfa_p95_ms": sorted(ttfas)[round(0.95 * (len(ttfas)-1))],
                        "ttfa_max_ms": max(ttfas), "rtfx_p50": statistics.median(speeds)})
                    print(name, trial, label, trial_stats[-1], flush=True)
        summary = []
        for name in TEXTS:
            for label in VARIANTS:
                rows = [r for r in trial_stats if r["text"] == name and r["variant"] == label]
                summary.append({"text": name, "variant": label, **{k: statistics.median(r[k] for r in rows)
                    for k in ["ttfa_p50_ms", "ttfa_p95_ms", "rtfx_p50"]},
                    "worst_ttfa_ms": max(r["ttfa_max_ms"] for r in rows)})
        write(out / "summary.json", summary)
        write(out / "trial-stats.json", trial_stats)
        (out / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in observations))
        manifest.update(measured_requests=len(observations), warmups=135)
    else:
        report = []
        for vi, voice in enumerate(voices):
            for ti, text in enumerate(QUALITY):
                audio = {}
                for label in ["baseline", "repeat", "packed", "combined"]:
                    p = invoke(label, text, voice, f"voice{vi}-text{ti}-{label}", 2, 1, "--pcm-output")
                    values = array.array("f", p.read_bytes())
                    if sys.byteorder != "little":
                        values.byteswap()
                    audio[label] = values
                    if vi == 0 and ti == 0:
                        pcm = array.array("h", [max(-32768, min(32767, round(x*32767))) for x in values])
                        if sys.byteorder != "little":
                            pcm.byteswap()
                        with wave.open(str(out / f"sample-{label}.wav"), "wb") as output:
                            output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                            output.writeframes(pcm.tobytes())
                row = {"voice": str(voice), "text": text, **{v: compare(audio["baseline"], audio[v]) for v in ["repeat", "packed", "combined"]}}
                report.append(row)
                print(vi, ti, {v: row[v].get("max_abs_error") for v in ["repeat", "packed", "combined"]}, flush=True)
        write(out / "report.json", report)
        assert all(r[v]["same_length"] and r[v]["max_abs_error"] <= 1e-5 for r in report for v in ["repeat", "packed", "combined"])
    write(out / "manifest.json", manifest)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "binary", "source", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--voice", type=Path, action="append", default=[])
    p.add_argument("--mode", choices=["timing", "quality"], required=True)
    main(p.parse_args())
