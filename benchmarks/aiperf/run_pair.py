#!/usr/bin/env python3
"""Run paired upstream/fork AIPerf profiles against fresh local servers."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def wait_ready(base_url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("server did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--fork-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voice-catalog", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=18880)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workload = [
        json.loads(line)
        for line in args.workload.read_text().splitlines()
        if line.strip()
    ]
    warmup = [row for row in workload if row["phase"] == "warmup"]
    measured = [row for row in workload if row["phase"] == "measured"]
    dataset = output / "input.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"text": row["input"], "extra": {"voice": row["voice"]}}) + "\n"
            for row in measured
        )
    )
    base_url = f"http://127.0.0.1:{args.port}"
    results = []
    for trial in range(args.trials):
        for variant in ("upstream", "fork") if trial % 2 == 0 else ("fork", "upstream"):
            binary = (
                args.baseline_binary if variant == "upstream" else args.fork_binary
            ).resolve()
            run_dir = output / f"trial-{trial + 1}-{variant}"
            run_dir.mkdir(exist_ok=True)
            command = [
                str(binary),
                "--model",
                str(args.model.resolve()),
                "--quant",
                "q8",
                "--device",
                "cpu",
                "--lang",
                "en",
                "--voice-catalog",
                str(args.voice_catalog.resolve()),
                "--addr",
                f"127.0.0.1:{args.port}",
                "--max-concurrent-requests",
                str(args.concurrency),
                "--temperature",
                "0.3",
                "--seed",
                "42",
            ]
            if variant == "fork":
                command += [
                    "--voice-cache-entries",
                    "16",
                    "--voice-cache-bytes",
                    "67108864",
                ]
            environment = dict(os.environ, RAYON_NUM_THREADS="4", XN_KAI="1")
            with (run_dir / "server.log").open("w") as server_log:
                server = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    wait_ready(base_url, server)
                    for row in warmup:
                        payload = json.dumps(
                            {
                                "input": row["input"],
                                "voice": row["voice"],
                                "response_format": "pcm",
                            }
                        ).encode()
                        request = urllib.request.Request(
                            base_url + "/v1/audio/speech",
                            payload,
                            {"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=30) as response:
                            if response.status != 200 or not response.read():
                                raise RuntimeError("warmup did not return audio")
                    aiperf = [
                        "uvx",
                        "--from",
                        "aiperf==0.13.0",
                        "--with",
                        "./benchmarks/aiperf",
                        "aiperf",
                        "profile",
                        "--endpoint-type",
                        "phonon-tts",
                        "--transport",
                        "phonon-http",
                        "--model",
                        "phonon-31m-en",
                        "--url",
                        base_url,
                        "--input-file",
                        str(dataset),
                        "--custom-dataset-type",
                        "single-turn",
                        "--dataset-sampling-strategy",
                        "sequential",
                        "--request-count",
                        str(len(measured)),
                        "--concurrency",
                        str(args.concurrency),
                        "--output-artifact-dir",
                        str(run_dir / "aiperf"),
                        "--no-gpu-telemetry",
                    ]
                    with (run_dir / "aiperf-console.log").open("w") as aiperf_log:
                        subprocess.run(
                            aiperf,
                            cwd=repo,
                            env=environment,
                            stdout=aiperf_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    report = json.loads(
                        (run_dir / "aiperf" / "profile_export_aiperf.json").read_text()
                    )
                    if (
                        report["request_count"]["avg"] != len(measured)
                        or report["request_latency"]["count"] != len(measured)
                        or report["request_error_rate"]["avg"] != 0
                        or report.get("error_summary")
                    ):
                        raise RuntimeError(
                            "AIPerf did not return valid PCM for every request"
                        )
                    row = {
                        "trial": trial + 1,
                        "variant": variant,
                        "concurrency": args.concurrency,
                        "requests": len(measured),
                        "binary_sha256": hashlib.sha256(
                            binary.read_bytes()
                        ).hexdigest(),
                        "request_latency_ms": report["request_latency"],
                        "request_throughput_rps": report["request_throughput"]["avg"],
                        "benchmark_duration_s": report["benchmark_duration"]["avg"],
                        "error_summary": report.get("error_summary", []),
                        "run_dir": run_dir.name,
                    }
                    results.append(row)
                    (output / "summary.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
                    print(
                        f"{row['trial']} {variant}: p50={row['request_latency_ms']['p50']:.2f}ms p95={row['request_latency_ms']['p95']:.2f}ms {row['request_throughput_rps']:.2f}rps",
                        flush=True,
                    )
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()


if __name__ == "__main__":
    main()
