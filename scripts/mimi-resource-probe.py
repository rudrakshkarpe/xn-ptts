#!/usr/bin/env python3
"""Fresh-process macOS peak RSS and load-time comparison; not a TTFA benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model = args.root.resolve() / "models/phonon-31m-en"
    binary = args.binary.resolve()
    rows = []
    for trial in range(3):
        for variant in (["baseline", "packed"] if trial % 2 == 0 else ["packed", "baseline"]):
            stem = args.out / f"{trial}-{variant}"
            env = {k: v for k, v in os.environ.items() if not k.startswith("PTTS_EXPERIMENT_") and k != "XN_SPIN_BUDGET"}
            if variant == "packed":
                env["PTTS_EXPERIMENT_PACKED_MIMI"] = "1"
            command = ["/usr/bin/time", "-l", "-o", str(stem.with_suffix(".time")), str(binary),
                       "--model", str(model / "model.q8.gguf"), "--config", str(model / "config.json"),
                       "--voice", str(model / "default-voice.safetensors"), "--quant", "q8", "--device", "cpu",
                       "--threads", "4", "--lang", "en", "--seed", "42", "--warmup", "5", "--iters", "10",
                       "--input", "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun."]
            p = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            stem.with_suffix(".txt").write_text(p.stdout)
            usage = stem.with_suffix(".time").read_text()
            rows.append({"trial": trial, "variant": variant, "command": command,
                         "load_ms": float(re.search(r"load ([0-9.]+)ms", p.stdout)[1]),
                         "peak_rss_bytes": int(re.search(r"(\d+)\s+maximum resident set size", usage)[1]),
                         "packed_weight_bytes": sum(map(int, re.findall(r"experimental packed Mimi matrix:.*bytes=(\d+)", p.stdout)))})
    (args.out / "results.json").write_text(json.dumps({"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "runs": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "binary", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
