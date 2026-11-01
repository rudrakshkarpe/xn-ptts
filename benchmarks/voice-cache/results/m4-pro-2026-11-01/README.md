# M4 Pro comparison — 1 November 2026

The controlled voice-pressure diagnostic improved median client TTFA by **54.6%**
and aggregate audio throughput by **11.9%** against the pinned upstream binary.
The single-voice control had effectively identical median TTFA. The adverse
round-robin control was slightly slower. This is evidence for a targeted cache-policy
change, not a general kernel speedup or a win over published M5 benchmarks.

[Graphs and paired audio samples](showcase/README.md) include all workload controls,
per-trial dots, a request trace and sample-level audio differences.

## Paired results

Each cell is the **median of three per-trial statistics**, not a pooled percentile.
Concurrency is one. There were **1,764 measured requests and 204 warm-up requests**,
all completed with no HTTP errors or rejections. First audio means the first complete
PCM sample received by the localhost client, not HTTP headers or audible playback.

| Workload | TTFA p50, upstream → fork | TTFA p95, upstream → fork | Audio seconds / wall second, upstream → fork |
|---|---:|---:|---:|
| One warm voice | 4.445 → 4.445 ms | 4.897 → 4.695 ms | 63.52 → 64.34 (+1.3%) |
| Controlled voice pressure | 9.856 → 4.479 ms (−54.6%) | 10.424 → 9.887 ms (−5.2%) | 56.65 → 63.40 (+11.9%) |
| Round-robin 17 voices | 9.882 → 10.057 ms (+1.8%) | 10.611 → 10.692 ms | 56.90 → 56.01 (−1.6%) |

Small differences in the controls should not be treated as statistically established
gains or regressions on this shared desktop. The pressure TTFA reduction occurs in
all three trials. The first warm-voice pair has more throughput variation than the
other two; it is retained rather than discarded.

The pressure workload deliberately alternates two occasional voices around 15 hot
voices, returning to hot voices in reverse order. It exposes upstream's clear-all
behavior at its 16-entry boundary. It is **not a representative production trace**.
The fork records 89 hits, 7 misses and 7 evictions during each 96-request measured
pressure run. Its retained KV payload stays at 49,152,000 bytes (46.875 MiB), within
the common 16-entry capacity and the fork's 64 MiB byte budget. Baseline has no cache
counter endpoint; its internal hit/miss count was not instrumented.

Round-robin accesses all 17 voices in order with only 16 retained entries. The fork
records 102 misses and 102 evictions per measured trial. LRU cannot help this pattern;
a larger cache would be a separate configuration experiment with a different memory
budget. RSS and energy were not measured.

## Every trial

Links open each run's report; its adjacent `requests.jsonl` contains all raw records.
TTFA columns are p50 / p95 in milliseconds. Throughput is audio seconds per wall second.

| Workload / trial | Upstream TTFA | Fork TTFA | Upstream throughput | Fork throughput |
|---|---:|---:|---:|---:|
| Warm / 1 | [4.483 / 5.523](warm-one-upstream-1/report.json) | [4.489 / 5.250](warm-one-optimized-1/report.json) | 62.072 | 58.692 |
| Warm / 2 | [4.445 / 4.782](warm-one-upstream-2/report.json) | [4.445 / 4.695](warm-one-optimized-2/report.json) | 64.356 | 64.340 |
| Warm / 3 | [4.441 / 4.897](warm-one-upstream-3/report.json) | [4.404 / 4.691](warm-one-optimized-3/report.json) | 63.524 | 64.598 |
| Pressure / 1 | [10.216 / 13.652](voice-pressure-upstream-1/report.json) | [4.479 / 9.854](voice-pressure-optimized-1/report.json) | 54.349 | 63.396 |
| Pressure / 2 | [9.856 / 10.424](voice-pressure-upstream-2/report.json) | [4.444 / 9.966](voice-pressure-optimized-2/report.json) | 56.648 | 63.673 |
| Pressure / 3 | [9.794 / 10.423](voice-pressure-upstream-3/report.json) | [4.481 / 9.887](voice-pressure-optimized-3/report.json) | 57.019 | 61.681 |
| Round-robin / 1 | [9.788 / 10.690](round-robin-17-upstream-1/report.json) | [10.057 / 11.259](round-robin-17-optimized-1/report.json) | 56.903 | 55.322 |
| Round-robin / 2 | [9.891 / 10.611](round-robin-17-upstream-2/report.json) | [10.095 / 10.692](round-robin-17-optimized-2/report.json) | 57.411 | 56.014 |
| Round-robin / 3 | [9.882 / 10.445](round-robin-17-upstream-3/report.json) | [9.969 / 10.635](round-robin-17-optimized-3/report.json) | 56.649 | 56.670 |

## Numerical checks

All **984 paired response lengths** match, including warm-up. Only 55 paired PCM
hashes match exactly. Repeat runs of upstream also differ, so hash inequality alone
is not evidence of a fork regression. The upstream CPU pipeline opportunistically
batches already queued Mimi frames; its batch shapes depend on worker timing. This
is a plausible source of rounding variation, not an experimentally isolated cause.

A separate check retained PCM from the first 48 pressure-workload requests, with
two fresh-server runs of each binary using the same four-thread/KleidiAI settings.
All compared lengths match. In both upstream repeat comparisons and fork/upstream
comparisons, the maximum absolute difference is **one signed 16-bit PCM step**
(`1 / 32768` of normalized full scale). Worst per-response normalized RMSE across
the cross-version comparisons is `1.006e-6` rounded upward.
[Per-request numerical evidence](pcm-parity.json) includes hashes, lengths, maximum
errors and RMSE. No ASR, speaker-similarity or listening evaluation was run.

The opt-in Rust checkpoint tests independently passed with `RAYON_NUM_THREADS=1`
and `XN_KAI=0`: cached versus disabled/evicted prefixes produced finite float audio
of equal length with maximum error below `1e-5`, and separately owned concurrent
sessions remained isolated. GPU behavior has not been validated.

## Provenance and validation

- Machine: Apple M4 Pro, 12 logical CPUs, 24 GiB RAM; macOS 27.0.1 (26A434).
- Rust 1.92.0, release profile, `accelerate,kai`, `target-cpu=native`;
  `RAYON_NUM_THREADS=4`, `XN_KAI=1`, CPU Q8, temperature 0.3, seed 42.
- Upstream runtime: `31e4e9f80275e432b570e17f5ccbefa895eff109`.
- Fork runtime: `df6aeaa68d318c9ec5684dc6658eb8c0b5304ece`. Later commits add
  documentation and evidence without changing the measured runtime or harness.
- The upstream binary was built and preserved before source edits. Both use the
  same Cargo lockfile, compiler, features and assets. Binary hashes, exact arguments,
  run order and dates are in [environment.json](environment.json); model/voice pins
  and file hashes are in [assets.json](assets.json).
- Every scenario starts a fresh server and uses the checked-in warm-up order. Trials
  alternate upstream/fork order. No compiler or other task-owned inference ran during
  measurements. The machine was not isolated from ordinary desktop activity; clocks,
  thermal state and power were not controlled.
- `cargo check --locked --workspace`, formatting and `git diff --check` passed.
- Core/server tests: 170 passed; the two new opt-in checkpoint tests were then run
  explicitly and passed. Four Python harness tests passed, covering first-sample
  timing, failures/partial responses, throughput accounting and workload validation.
- Clippy for core/server/all targets passed with
  `-D warnings -A clippy::nonminimal_bool`. The exception is for the existing upstream
  expression at `ptts/src/preprocess.rs:1252` under local Rust 1.92.0. An unrestricted
  warnings-as-errors run fails there; that unrelated expression was not changed.

[summary.json](summary.json) contains the computed medians, per-trial values and
cache deltas. These short trials do not establish production p99, sustainable QPS,
behavior under concurrent load, playback underruns, GPU speed, or quality across
texts and languages. See the [reproduction guide and next layers](../../README.md)
for the next experiments.
