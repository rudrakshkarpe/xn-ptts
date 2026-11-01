#!/usr/bin/env python3
"""Compare saved float32 output, with a repeated baseline to expose runtime variation."""
import argparse
import array
import json
import math
from pathlib import Path
import subprocess
import sys
import wave

TEXTS = [
    "Hey Gradium team, this is Rudraksh loving the phonon so far!",
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "Your appointment is at 10:30 tomorrow morning, and the total comes to 42 dollars.",
    "Please confirm your delivery address. I will then check the available delivery dates for you.",
]


def compare(a, b):
    if len(a) != len(b):
        return {"same_length": False, "reference_samples": len(a), "candidate_samples": len(b)}
    assert a and all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
    errors = [float(x) - float(y) for x, y in zip(a, b)]
    mse = sum(x*x for x in errors) / len(a)
    power = sum(float(x)*float(x) for x in a) / len(a)
    return {"same_length": True, "samples": len(a), "different_samples": sum(x != 0 for x in errors),
            "max_abs_error": max(map(abs, errors)), "rmse": math.sqrt(mse),
            "snr_db": 10 * math.log10(power/mse) if mse else None, "bit_identical": mse == 0}


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model = args.root.resolve() / "models/phonon-31m-en"
    shared = ["--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
              "--voice", str(model / "default-voice.safetensors"), "--quant", "q8", "--device", "cpu",
              "--threads", str(args.threads), "--lang", "en", "--seed", "42", "--warmup", "2", "--iters", "1"]
    report = []
    for index, text in enumerate(TEXTS):
        audio = {}
        for label, binary in [("baseline", args.baseline), ("repeat", args.baseline), ("candidate", args.candidate)]:
            path = (args.out / f"{index}-{label}.f32").resolve()
            proc = subprocess.run([str(binary.resolve()), *shared, "--input", text, "--pcm-output", str(path)],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            (args.out / f"{index}-{label}.txt").write_text(proc.stdout)
            values = array.array("f", path.read_bytes())
            if sys.byteorder != "little":
                values.byteswap()
            audio[label] = values
            if index == 0:
                pcm = array.array("h", [max(-32768, min(32767, round(x*32767))) for x in values])
                if sys.byteorder != "little":
                    pcm.byteswap()
                with wave.open(str(args.out / f"sample-{label}.wav"), "wb") as output:
                    output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    output.writeframes(pcm.tobytes())
        report.append({"text": text, "candidate": compare(audio["baseline"], audio["candidate"]),
                       "baseline_repeat": compare(audio["baseline"], audio["repeat"])})
    (args.out / "report.json").write_text(json.dumps({"threads": args.threads, "seed": 42, "results": report}, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if any(not row["candidate"]["same_length"] or row["candidate"].get("max_abs_error", math.inf) > args.max_abs_error for row in report):
        raise SystemExit("candidate exceeded the waveform tolerance; inspect report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "baseline", "candidate", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-abs-error", type=float, default=1e-5)
    run(parser.parse_args())
