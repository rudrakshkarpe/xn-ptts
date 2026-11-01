#!/usr/bin/env python3
"""Check experimental Mimi paths against a repeated baseline, without timing claims."""
import argparse
import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

def module(filename):
    spec = importlib.util.spec_from_file_location(filename, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def run(args):
    reference = module("check-ttfa-audio.py")
    variants = module("bench-mimi-research.py").VARIANTS
    args.out.mkdir(parents=True, exist_ok=False)
    root, binary = args.root.resolve(), args.binary.resolve()
    model = root / "models/phonon-31m-en"
    voices = [model / "default-voice.safetensors"] + args.voice
    report = []
    for vi, voice in enumerate(voices):
        for ti, text in enumerate(reference.TEXTS):
            audio = {}
            for label in ["baseline", "repeat", "packed", "packed-wave20"]:
                env = {k: v for k, v in os.environ.items() if not k.startswith("PTTS_EXPERIMENT_") and k != "XN_SPIN_BUDGET"}
                env.update(variants.get(label, {}))
                stem = f"voice{vi}-text{ti}-{label}"
                path = (args.out / (stem + ".f32")).resolve()
                command = [str(binary), "--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
                           "--voice", str(voice.resolve()), "--quant", "q8", "--device", "cpu", "--threads", str(args.threads),
                           "--lang", "en", "--seed", "42", "--warmup", "2", "--iters", "1", "--input", text,
                           "--pcm-output", str(path)]
                p = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
                (args.out / (stem + ".txt")).write_text(p.stdout)
                values = array.array("f", path.read_bytes())
                if sys.byteorder != "little":
                    values.byteswap()
                audio[label] = values
                if vi == 0 and ti == 0:
                    pcm = array.array("h", [max(-32768, min(32767, round(x * 32767))) for x in values])
                    if sys.byteorder != "little":
                        pcm.byteswap()
                    with wave.open(str(args.out / f"sample-{label}.wav"), "wb") as output:
                        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                        output.writeframes(pcm.tobytes())
            row = {"voice": str(voice), "text": text, **{v: reference.compare(audio["baseline"], audio[v]) for v in ["repeat", "packed", "packed-wave20"]}}
            report.append(row)
            print(vi, ti, {v: row[v].get("snr_db") for v in ["repeat", "packed", "packed-wave20"]}, flush=True)
    result = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "seed": 42, "threads": args.threads, "results": report}
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    if any(not r[v]["same_length"] or r[v].get("max_abs_error", math.inf) > 1e-5 for r in report for v in ["repeat", "packed", "packed-wave20"]):
        raise SystemExit("waveform check failed; inspect report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "binary", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--voice", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
