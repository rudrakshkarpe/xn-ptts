"""Create paired Phonon speech samples with identical requests and startup settings."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()  # experiment folder containing binaries/ and models/
OUTPUT = Path(sys.argv[2]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
TEXT = "Hey Gradium team, this is Rudraksh loving the phonon so far!"
PORT = 18880
BASE_URL = f"http://127.0.0.1:{PORT}"
SAMPLE_RATE = 24000
VOICES = [
    "Zoey", "Juniper", "Everly", "Harper", "Hadley", "Delia", "Easton",
    "Micah", "Trevor", "Russell", "Preston", "Miles", "Maeve", "Freya",
    "Pippa", "Elodie-Rose",
]
WARM_TEXT = "Your order is ready. Thank you for waiting."


def request_audio(voice: str, text: str) -> tuple[bytes, float]:
    payload = json.dumps({"input": text, "voice": voice, "response_format": "pcm"}).encode()
    request = urllib.request.Request(
        BASE_URL + "/v1/audio/speech",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200 or response.headers.get_content_type() != "audio/pcm":
            raise RuntimeError(f"unexpected response: {response.status} {response.headers}")
        first_sample = response.read(2)
        first_audio_ms = (time.perf_counter_ns() - start) / 1e6
        body = first_sample + response.read()
    if not body or len(body) % 2:
        raise RuntimeError("missing or partial PCM sample")
    return body, first_audio_ms


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


def wait_ready(server: subprocess.Popen[bytes]) -> None:
    for _ in range(300):
        if server.poll() is not None:
            raise RuntimeError(f"server exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("server did not become healthy")


def main() -> None:
    records: dict[str, dict] = {}
    for variant, binary_name in [("original", "upstream-server"), ("optimized", "optimized-server")]:
        binary = ROOT / "binaries" / binary_name
        command = [
            str(binary), "--model", str(ROOT / "models/phonon-31m-en"),
            "--quant", "q8", "--device", "cpu", "--lang", "en",
            "--voice-catalog", str(ROOT / "models/voices/voices.jsonl"),
            "--addr", f"127.0.0.1:{PORT}", "--max-concurrent-requests", "1",
            "--temperature", "0.3", "--seed", "42",
        ]
        if variant == "optimized":
            command += ["--voice-cache-entries", "16", "--voice-cache-bytes", "67108864"]
        env = dict(os.environ, RAYON_NUM_THREADS="4", XN_KAI="1", HF_HUB_OFFLINE="1")
        with (OUTPUT / f"{variant}-server.log").open("wb") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                wait_ready(server)
                pcm, ttfa = request_audio("default", TEXT)
                write_wav(OUTPUT / f"{variant}-default.wav", pcm)
                records[f"{variant}-default"] = {"voice": "default", "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "samples": len(pcm)//2, "first_audio_ms": ttfa}
                # The seventeenth catalog voice clears the original cache, while
                # the optimized cache retains recently used Pippa.
                for voice in VOICES:
                    request_audio(voice, WARM_TEXT)
                request_audio("Imogen", WARM_TEXT)
                pcm, ttfa = request_audio("Pippa", TEXT)
                write_wav(OUTPUT / f"{variant}-cache-pressure.wav", pcm)
                records[f"{variant}-cache-pressure"] = {"voice": "Pippa", "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "samples": len(pcm)//2, "first_audio_ms": ttfa}
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    comparisons = {}
    for scenario in ["default", "cache-pressure"]:
        def read_samples(variant: str) -> np.ndarray:
            with wave.open(str(OUTPUT / f"{variant}-{scenario}.wav"), "rb") as wav:
                assert wav.getnchannels() == 1 and wav.getframerate() == SAMPLE_RATE and wav.getsampwidth() == 2
                return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.int32)
        original = read_samples("original")
        optimized = read_samples("optimized")
        result = {"same_sample_count": bool(len(original) == len(optimized))}
        if len(original) == len(optimized):
            delta = optimized - original
            result.update({
                "max_abs_pcm_steps": int(np.max(np.abs(delta))),
                "rmse_pcm_steps": float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
                "normalized_rmse": float(np.sqrt(np.mean((delta / 32768.0)**2))),
                "nonzero_sample_fraction": float(np.count_nonzero(delta)/len(delta)),
                "original_rms": float(np.sqrt(np.mean(original.astype(np.float64)**2))),
                "optimized_rms": float(np.sqrt(np.mean(optimized.astype(np.float64)**2))),
                "correlation": float(np.corrcoef(original, optimized)[0,1]),
            })
        comparisons[scenario] = result
    report = {
        "text": TEXT,
        "sample_rate_hz": SAMPLE_RATE,
        "format": "mono signed 16-bit PCM WAV",
        "model_revision": "92741c5b07d085ff840e26fea8ce43110b87852f",
        "original_source": "31e4e9f80275e432b570e17f5ccbefa895eff109",
        "optimized_source": "df6aeaa68d318c9ec5684dc6658eb8c0b5304ece",
        "warmup_for_pressure": "16 catalog voices, followed by Imogen, then the sample in Pippa",
        "records": records,
        "comparisons": comparisons,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
