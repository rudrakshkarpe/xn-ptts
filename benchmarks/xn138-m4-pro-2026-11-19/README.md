# M4 Pro validation of xn #138

Independent local comparison of released xn 0.2.13 with xn #138 at `7e9ffab9b82c7a45a953140a4f2b2578cf356b0a`, through ptts `Synth::stream_with`.

Steady-state gains are visible, especially with KleidiAI alone. First synthesis TTFA increases in all 36 paired process trials; the median increase across the twelve workload configurations is 1.60 to 2.98 ms. Cache initialization is a possible explanation, but this run does not profile its cause. This first-request tradeoff should be considered alongside the steady-state improvement.

## Conditions and limits

Apple M4 Pro (12 cores), macOS 27.2.1, AC power. Heavy user applications and iCloud sync were temporarily suspended during this run and resumed afterward. macOS background services remained active. This is a desktop benchmark, not an isolated performance lab. See `environment.json` for the recorded conditions. An earlier battery-powered run is excluded from this report.

Both variants use ptts `e5110afd3ead725c940b6411eec4985fb8b111aa`, identical benchmark instrumentation, the same Q8 model, default voice, seed 42, and CPU inference. Baseline xn 0.2.13 was published from `4242d4445ffafd8ed77380d30890db057102cb0c`. The five commits between that revision and #138's parent `135b5fdbb3b8c9690b0207269454315e10cafdf2` change CUDA code and version metadata, not the CPU paths exercised here. This remains a released-baseline comparison, not a build of the exact parent revision.

## Method

- 1,440 measured synthesis requests: 2 backends x 2 thread counts x 3 texts x 2 variants x 3 fresh-process trials x 20 requests.
- Text order rotates and variant order reverses between trials. Each cell below is a median of per-process statistics, not a pooled percentile.
- First TTFA is the first synthesis request in each process after model loading and voice conditioning. It is not application cold-start latency. Steady TTFA uses the remaining 19 requests.
- TTFA starts before `Synth::stream_with` and ends at the first nonempty chunk. Every timing record is checked to contain at least 1,920 samples in that chunk: one complete 80 ms frame at 24 kHz.
- Per-frame time is total synthesis duration divided by the number of 80 ms audio frames. It includes token generation and synthesis overhead; it is not a decoder-kernel-only measurement.
- JSON output is written after measurements. PCM capture is performed in a separate parity run. P95 values from 19 requests are included in raw statistics but are not treated as stable tail-latency estimates.

## Timing results

All times are milliseconds. Arrows mean baseline to #138. Positive improvement means lower steady TTFA.

| Backend | Threads | Text | First TTFA | Steady TTFA p50 | Improvement | Per-frame p50 |
|---|---:|---|---:|---:|---:|---:|
| accelerate,kai | 1 | short | 5.676 → 8.458 | 3.744 → 3.489 | 6.8% | 1.462 → 1.463 |
| accelerate,kai | 1 | release | 7.052 → 9.584 | 4.796 → 4.807 | -0.2% | 1.338 → 1.353 |
| accelerate,kai | 1 | long | 7.888 → 10.744 | 5.687 → 5.481 | 3.6% | 1.291 → 1.273 |
| accelerate,kai | 4 | short | 5.707 → 8.363 | 3.634 → 3.349 | 7.8% | 1.220 → 1.236 |
| accelerate,kai | 4 | release | 5.955 → 8.933 | 4.479 → 4.212 | 6.0% | 1.067 → 1.061 |
| accelerate,kai | 4 | long | 7.774 → 10.737 | 5.293 → 4.953 | 6.4% | 1.003 → 0.996 |
| kai | 1 | short | 7.743 → 9.899 | 5.593 → 4.487 | 19.8% | 2.338 → 2.190 |
| kai | 1 | release | 9.278 → 11.442 | 6.803 → 5.633 | 17.2% | 2.160 → 2.102 |
| kai | 1 | long | 10.372 → 12.019 | 7.756 → 6.551 | 15.5% | 2.132 → 2.097 |
| kai | 4 | short | 6.410 → 8.364 | 4.615 → 3.566 | 22.7% | 1.638 → 1.472 |
| kai | 4 | release | 7.073 → 8.676 | 5.511 → 4.370 | 20.7% | 1.461 → 1.367 |
| kai | 4 | long | 8.478 → 10.194 | 6.279 → 5.165 | 17.7% | 1.401 → 1.348 |

## Trial variability

Ranges below are observed ranges across three processes, not confidence intervals. Positive paired improvements favor #138. Small differences overlapping trial variability should be treated cautiously.

| Backend | Threads | Text | Paired steady TTFA improvement range | Baseline p50 range (ms) | #138 p50 range (ms) |
|---|---:|---|---:|---:|---:|
| acckai | 1 | short | 4.9% to 7.8% | 3.670 to 3.808 | 3.475 to 3.513 |
| acckai | 1 | release | -0.2% to 4.8% | 4.724 to 5.011 | 4.499 to 4.808 |
| acckai | 1 | long | 3.0% to 4.1% | 5.653 to 5.776 | 5.454 to 5.571 |
| acckai | 4 | short | 7.4% to 8.3% | 3.616 to 3.651 | 3.334 to 3.364 |
| acckai | 4 | release | 5.3% to 6.9% | 4.478 to 4.522 | 4.179 to 4.242 |
| acckai | 4 | long | 5.1% to 6.4% | 5.213 to 5.306 | 4.922 to 5.034 |
| kai | 1 | short | 13.7% to 21.9% | 5.525 to 5.747 | 4.429 to 4.770 |
| kai | 1 | release | 16.1% to 20.1% | 6.761 to 6.933 | 5.538 to 5.672 |
| kai | 1 | long | 13.6% to 15.6% | 7.581 to 7.841 | 6.543 to 6.634 |
| kai | 4 | short | 21.5% to 22.8% | 4.509 to 4.655 | 3.538 to 3.595 |
| kai | 4 | release | 19.8% to 21.2% | 5.441 to 5.576 | 4.365 to 4.394 |
| kai | 4 | long | 17.0% to 17.8% | 6.272 to 6.280 | 5.161 to 5.207 |

## Output comparison

Three texts x two backends x two thread counts. Each process performs two unmeasured warmups before capturing one output. Baseline repeats estimate run-to-run numerical variation. The cache-off ablation uses the same #138 binary with `GEMM_PACKED_LHS_CACHE_MB=0`.

| Comparison | Matching lengths | Bit-identical cases | Maximum absolute sample difference | Minimum finite SNR |
|---|---:|---:|---:|---:|
| Baseline repeat vs baseline | 12/12 | 1/12 | 1.84029341e-06 | 125.70 dB |
| #138 vs baseline | 12/12 | 0/12 | 1.48452818e-06 | 120.81 dB |
| Original #322 vs baseline | 12/12 | 0/12 | 1.84029341e-06 | 126.41 dB |
| #138 cache on vs cache off | 12/12 | 0/12 | 1.25914812e-06 | 125.71 dB |

All captured outputs were checked for finite samples. Numerical closeness is not a perceptual quality or intelligibility evaluation. This run does not measure cache memory overhead, cache eviction under pressure, concurrent-client throughput, or other hardware.

## Artifacts

- [Reproduction instructions](REPRODUCE.md), [runner](run.py), and [report generator](summarize.py).
- `manifest.json`: source revisions, model/binary hashes, and executed commands. Local path prefixes are normalized to `$RESULTS` and `$MODEL`.
- `timing/`: per-process raw JSON and logs, all observations, and summary statistics.
- `parity/report.json`: every numerical comparison; `parity/pcm-sha256.json`: captured PCM hashes. Model weights and native binaries are not included.
