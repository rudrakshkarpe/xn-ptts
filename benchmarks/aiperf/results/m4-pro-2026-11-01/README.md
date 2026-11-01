# AIPerf 0.13.0 local CPU comparison — 1 November 2026

On the constructed 17-voice cache-pressure workload, the fork reduced median
**complete-response** latency from **54.18 to 48.90 ms (−9.8%)** and increased
throughput from **17.62 to 19.40 requests/s (+10.1%)** at one concurrent request.
All three paired trials favored the fork. This is a targeted cache-policy result,
not a general inference-kernel speedup.

| Workload / concurrency | Complete-response p50, upstream → fork | p95, upstream → fork | Requests/s, upstream → fork |
|---|---:|---:|---:|
| One warm voice / 1 | 49.81 → 49.97 ms | 72.39 → 58.11 ms | 18.15 → 19.10 |
| Cache pressure / 1 | **54.18 → 48.90 ms (−9.8%)** | 64.39 → 58.47 ms | **17.62 → 19.40 (+10.1%)** |
| Round-robin 17 voices / 1 | 51.56 → 52.17 ms | 74.26 → 60.55 ms | 17.97 → 18.23 |
| Cache pressure / 2 | 77.05 → 71.01 ms (−7.8%) | 99.82 → 83.39 ms | 24.23 → 26.60 (+9.8%) |

Each cell is the median of **three per-trial statistics**, not a pooled
percentile. The completed-request p95 values are descriptive, not production
tail-latency estimates. In the two-concurrent-request case, the third pair
reversed direction (p50 76.66 → 78.23 ms; throughput 22.72 → 21.93 requests/s),
so the apparent median advantage at concurrency two needs a longer, isolated
capacity run. The one-voice and round-robin p50 controls do not show a clear
benefit. Their p95 variation should not be attributed to the cache change.

## Trial-level results

| Workload / concurrency | Trial | p50 ms, upstream → fork | Requests/s, upstream → fork |
|---|---:|---:|---:|
| Cache pressure / 1 | 1 | 53.53 → 49.05 | 18.03 → 19.40 |
| Cache pressure / 1 | 2 | 54.18 → 48.84 | 17.62 → 19.21 |
| Cache pressure / 1 | 3 | 54.61 → 48.90 | 17.32 → 19.45 |
| Cache pressure / 2 | 1 | 77.05 → 69.38 | 24.23 → 27.29 |
| Cache pressure / 2 | 2 | 77.69 → 71.01 | 24.70 → 26.60 |
| Cache pressure / 2 | 3 | 76.66 → 78.23 | 22.72 → 21.93 |

## Method and scope

- Apple M4 Pro, 12 logical CPUs, 24 GiB RAM; macOS 27.0.1; CPU Q8;
  `accelerate,kai`; `RAYON_NUM_THREADS=4`; `XN_KAI=1`; temperature 0.3;
  seed 42; 24 kHz mono signed 16-bit PCM. Same saved binaries and assets as the
  [earlier PCM benchmark](../../../voice-cache/results/m4-pro-2026-11-01/README.md).
- Upstream source `31e4e9f80275e432b570e17f5ccbefa895eff109`, binary SHA-256
  `d8bf9d24075720ba8eb25a37292c5f3440e9e47d45434e07d90f20c2d1d718bf`;
  fork runtime `df6aeaa68d318c9ec5684dc6658eb8c0b5304ece`, binary SHA-256
  `d491397620bec684d3eb5af9e9b7e6e081c8b8362ba7c58cec3694f022f5754d`.
- AIPerf `0.13.0`, local HTTP transport with the
  [Phonon adapter](../../README.md), closed-loop concurrency 1 or 2, one server
  process at a time, max concurrent server requests equal to offered concurrency.
  Each server started fresh and received the workload's explicit warm-up before
  AIPerf began. Variant order alternated by trial.
- **2,340 measured requests** across 24 runs and **300 warm-up requests**. All
  measured requests returned validated nonempty, even-length PCM; AIPerf reported
  zero request errors. The checked-in input JSONL, per-run AIPerf JSON exports
  and compact summaries are under each workload directory.
- AIPerf's complete-response latency is different from the PCM harness's
  first-sample TTFA. Its token metrics are inapplicable to TTS. The previous
  pressure TTFA comparison was 9.856 → 4.479 ms at p50, measured by the separate
  streaming PCM harness. The same source workload is used here, but the client
  and metric differ, so those timings should not be combined.
- The voice-pressure trace deliberately exercises a 16-entry boundary with
  17 voices; it is not a production frequency distribution. These runs were on
  a shared desktop without clock, thermal or power isolation. No new quality,
  long-text, cancellation, GPU, ASR or listening evaluation was performed.
