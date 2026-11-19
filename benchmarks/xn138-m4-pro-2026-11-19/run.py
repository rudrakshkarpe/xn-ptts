#!/usr/bin/env python3
"""Compare the ptts base, xn #138, and ptts #322 on one host."""

import argparse
import array
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


TEXTS = {
    "short": "Hello, how are you?",
    "release": "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "long": "Thank you for calling. I have checked the details of your reservation, and everything is ready for your arrival tomorrow afternoon. Please bring your confirmation number and a photo identification document. If your travel plans change, you can contact us to arrange a different arrival time.",
}
VARIANTS = ("baseline", "xn138")
CONFIGS = {"acckai": "hf,accelerate,kai", "kai": "hf,kai"}
THREADS = (1, 4)
TRIALS = 3
ITERS = 20
FRAME_SAMPLES = 1920


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def binary_path(root: Path, config: str, variant: str) -> Path:
    return root / "binaries" / f"ptts-{variant}-{config}"


def invoke(
    root: Path,
    model: Path,
    config: str,
    threads: int,
    variant: str,
    text: str,
    stem: str,
    capture: str,
    warmup: int,
    iters: int,
    manifest: dict,
) -> Path:
    output = root / capture / f"{stem}{'.json' if capture == 'timing' else '.f32'}"
    log = root / capture / f"{stem}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    actual_variant = {"baseline-repeat": "baseline", "xn138-cache-off": "xn138"}.get(variant, variant)
    command = [
        str(binary_path(root, config, actual_variant)),
        "--model", str(model / "model.q8.gguf"),
        "--config", str(model / "config.json"),
        "--voice", str(model / "default-voice.safetensors"),
        "--quant", "q8", "--cpu", "--threads", str(threads),
        "--lang", "en", "--seed", "42", "--warmup", str(warmup),
        "--iters", str(iters), "--input", text,
        "--metrics-json" if capture == "timing" else "--pcm-output", str(output),
    ]
    overrides = {"XN_KAI": "1", "RAYON_NUM_THREADS": str(threads)}
    if variant == "xn138-cache-off":
        overrides["GEMM_PACKED_LHS_CACHE_MB"] = "0"
    if actual_variant == "pr322":
        overrides.update(
            PTTS_EXPERIMENT_PACKED_MIMI="1",
            PTTS_EXPERIMENT_PACKED_CONV="initial-small",
        )
    manifest["commands"].append({
        "config": config, "threads": threads, "variant": variant,
        "command": command, "env_overrides": overrides,
    })
    save_json(root / "manifest.json", manifest)
    if not (output.exists() and log.exists()):
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("PTTS_EXPERIMENT_")
            and key not in ("XN_SPIN_BUDGET", "GEMM_PACKED_LHS_CACHE_MB")
        }
        environment.update(overrides)
        completed = subprocess.run(
            command, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        )
        log.write_text(completed.stdout)
    return output


def timing(root: Path, model: Path, manifest: dict) -> None:
    observations = []
    trials = []
    for config in CONFIGS:
        for threads in THREADS:
            for trial in range(TRIALS):
                names = list(TEXTS)
                names = names[trial:] + names[:trial]
                for name in names:
                    order = list(VARIANTS)
                    if (trial + threads + (config == "kai")) % 2:
                        order.reverse()
                    for variant in order:
                        stem = f"{config}-t{threads}-{name}-r{trial}-{variant}"
                        path = invoke(
                            root, model, config, threads, variant, TEXTS[name],
                            stem, "timing", 0, ITERS, manifest,
                        )
                        data = json.loads(path.read_text())
                        assert len(data["runs"]) == ITERS
                        records = []
                        for index, run in enumerate(data["runs"]):
                            assert run["samples"] > 0 and run["arrivals"]
                            assert run["arrivals"][0][1] >= FRAME_SAMPLES
                            assert run["arrivals"][0][0] == run["ttfa_ms"]
                            assert sum(size for _, size in run["arrivals"]) == run["samples"]
                            frame_count = run["samples"] / FRAME_SAMPLES
                            record = {
                                "config": config, "threads": threads, "text": name,
                                "trial": trial, "variant": variant, "iteration": index,
                                "phase": "first" if index == 0 else "steady",
                                "ttfa_ms": run["ttfa_ms"],
                                "total_ms": run["total_ms"],
                                "per_frame_ms": run["total_ms"] / frame_count,
                                "rtfx": run["samples"] / data["sample_rate"] * 1000 / run["total_ms"],
                                "samples": run["samples"],
                                "first_chunk_samples": run["arrivals"][0][1],
                            }
                            records.append(record)
                            observations.append(record)
                        steady = records[1:]
                        row = {
                            "config": config, "threads": threads, "text": name,
                            "trial": trial, "variant": variant,
                            "first_ttfa_ms": records[0]["ttfa_ms"],
                            "steady_ttfa_p50_ms": statistics.median(r["ttfa_ms"] for r in steady),
                            "steady_ttfa_p95_ms": sorted(r["ttfa_ms"] for r in steady)[round(.95 * (len(steady) - 1))],
                            "steady_per_frame_p50_ms": statistics.median(r["per_frame_ms"] for r in steady),
                            "steady_rtfx_p50": statistics.median(r["rtfx"] for r in steady),
                        }
                        trials.append(row)
                        save_json(root / "timing" / "trial-stats.json", trials)
                        print(config, threads, name, trial, variant, row["steady_ttfa_p50_ms"], flush=True)
    summary = []
    for config in CONFIGS:
        for threads in THREADS:
            for name in TEXTS:
                for variant in VARIANTS:
                    rows = [
                        row for row in trials
                        if (row["config"], row["threads"], row["text"], row["variant"])
                        == (config, threads, name, variant)
                    ]
                    summary.append({
                        "config": config, "threads": threads, "text": name, "variant": variant,
                        **{key: statistics.median(row[key] for row in rows) for key in (
                            "first_ttfa_ms", "steady_ttfa_p50_ms", "steady_ttfa_p95_ms",
                            "steady_per_frame_p50_ms", "steady_rtfx_p50",
                        )},
                    })
    save_json(root / "timing" / "summary.json", summary)
    with (root / "timing" / "observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row) + "\n")
    manifest["measured_requests"] = len(observations)
    save_json(root / "manifest.json", manifest)


def load_pcm(path: Path) -> array.array:
    pcm = array.array("f")
    pcm.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm


def compare(reference: array.array, candidate: array.array) -> dict:
    if len(reference) != len(candidate):
        return {"same_length": False, "reference_samples": len(reference),
                "candidate_samples": len(candidate)}
    assert reference and all(math.isfinite(x) for x in reference)
    assert all(math.isfinite(x) for x in candidate)
    differences = [float(x) - float(y) for x, y in zip(reference, candidate)]
    mse = sum(value * value for value in differences) / len(reference)
    power = sum(float(value) ** 2 for value in reference) / len(reference)
    return {
        "same_length": True, "samples": len(reference),
        "max_abs_error": max(abs(value) for value in differences),
        "rmse": math.sqrt(mse),
        "snr_db": 10 * math.log10(power / mse) if mse else None,
        "bit_identical": mse == 0,
    }


def parity(root: Path, model: Path, manifest: dict) -> None:
    rows = []
    for config in CONFIGS:
        for threads in THREADS:
            for name, text in TEXTS.items():
                pcm = {}
                for variant in ("baseline", "baseline-repeat", "xn138", "pr322", "xn138-cache-off"):
                    stem = f"{config}-t{threads}-{name}-{variant}"
                    path = invoke(root, model, config, threads, variant, text,
                                  stem, "parity", 2, 1, manifest)
                    pcm[variant] = load_pcm(path)
                row = {
                    "config": config, "threads": threads, "text": name,
                    **{variant: compare(pcm["baseline"], pcm[variant]) for variant in (
                        "baseline-repeat", "xn138", "pr322",
                    )},
                    "xn138-vs-cache-off": compare(pcm["xn138-cache-off"], pcm["xn138"]),
                }
                rows.append(row)
                save_json(root / "parity" / "report.json", rows)
                print("parity", config, threads, name,
                      row["xn138"].get("max_abs_error"), flush=True)
    manifest["parity_cases"] = len(rows)
    save_json(root / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "parity"), required=True)
    args = parser.parse_args()
    root, model = args.root.resolve(), args.model.resolve()
    assets = [model / name for name in (
        "model.q8.gguf", "config.json", "tokenizer.json", "default-voice.safetensors",
    )]
    binaries = {
        f"{config}-{variant}": binary_path(root, config, variant)
        for config in CONFIGS for variant in VARIANTS
    }
    manifest = {
        "platform": platform.platform(), "cpu": "Apple M4 Pro",
        "ptts_base": "e5110afd3ead725c940b6411eec4985fb8b111aa",
        "xn138": "7e9ffab9b82c7a45a953140a4f2b2578cf356b0a",
        "ptts322": "70d748173cfe0250a9e116e9706eb96809e1630b",
        "xn134_fork": "7363979dd2c8ff1c0beb3a023bd176901872922f",
        "features": CONFIGS, "threads": THREADS, "seed": 42,
        "iterations_per_process": ITERS, "trials": TRIALS,
        "assets": {str(path): sha256(path) for path in assets},
        "binaries": {name: sha256(path) for name, path in binaries.items()},
        "commands": [],
    }
    previous = root / "manifest.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        for key in ("ptts_base", "xn138", "ptts322", "assets", "binaries"):
            assert saved[key] == manifest[key], f"benchmark input changed: {key}"
        manifest = saved
    if args.mode == "timing":
        timing(root, model, manifest)
    else:
        parity(root, model, manifest)


if __name__ == "__main__":
    main()
