# AIPerf profiling for Phonon

This adapter lets NVIDIA AIPerf 0.13.0 profile the existing
`POST /v1/audio/speech` PCM endpoint without changing the server. It reports
completed-request latency, requests/second and request errors. AIPerf's HTTP
client buffers non-SSE binary responses, so these are **full response** timings,
not time to first audio (TTFA) or chunk cadence. Use
[`scripts/bench-serving.py`](../../scripts/bench-serving.py) for those metrics.

AIPerf 0.13.0 does not provide a TTS endpoint type. Its default HTTP transport
also tries to serialize binary PCM into its JSON record channel, which fails on
non-UTF-8 samples. The `phonon-tts` endpoint formats the speech request, while
`phonon-http` uses AIPerf's normal aiohttp request path and replaces the complete
PCM byte buffer with its validated byte count **after** the response is read and
timed. This keeps binary audio out of the record channel. Request-level metrics
are comparable between variants using this same adapter; token metrics are not
meaningful for TTS.

## Reproduce

Install `uv` and prepare both release binaries and model assets as described in
the [voice cache benchmark](../voice-cache/README.md). From the repository root:

```sh
python3 benchmarks/aiperf/run_pair.py \
  --baseline-binary ../binaries/upstream-server \
  --fork-binary ../binaries/optimized-server \
  --model ../models/phonon-31m-en \
  --voice-catalog ../models/voices/voices.jsonl \
  --workload benchmarks/voice-cache/workloads/voice-pressure.jsonl \
  --output ../results/aiperf/pressure-c1 --trials 3 --concurrency 1
```

The runner starts a fresh server for every variant and trial, performs the
workload's explicit warm-up, and runs AIPerf on measured rows in sequential
order. It alternates variant order across trials and writes AIPerf's original
artifacts, console logs, server logs, input JSONL and a compact summary under
the output directory. Change `--concurrency` to test the server with the same
`--max-concurrent-requests` limit. The limit is not a queue: excess requests
receive HTTP 429 and must be included in error-rate analysis.

The workload is an intentionally constructed 16-entry cache boundary case. It
does not represent a production voice distribution. The same sentence is used
throughout, so these measurements do not establish longer-text behavior,
quality, speaker similarity, or GPU performance. The earlier PCM parity check
only covered numerical output similarity. Use a separate ASR, speaker-similarity
and listening evaluation before asserting voice quality parity.
