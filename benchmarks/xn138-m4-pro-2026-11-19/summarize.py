import json, statistics
from pathlib import Path
p=Path(__file__).resolve().parent
rows=json.loads((p/'timing/summary.json').read_text())
trials=json.loads((p/'timing/trial-stats.json').read_text())
parity=json.loads((p/'parity/report.json').read_text())
lookup={(r['config'],r['threads'],r['text'],r['variant']):r for r in rows}
lines=['# M4 Pro validation of xn #138', '',
'Independent local comparison of released xn 0.2.13 with xn #138 at `7e9ffab9b82c7a45a953140a4f2b2578cf356b0a`, through ptts `Synth::stream_with`.', '',
'Steady-state gains are visible, especially with KleidiAI alone. First synthesis TTFA increases in all 36 paired process trials; the median increase across the twelve workload configurations is 1.60 to 2.98 ms. Cache initialization is a possible explanation, but this run does not profile its cause. This first-request tradeoff should be considered alongside the steady-state improvement.', '', '## Conditions and limits', '',
'Apple M4 Pro (12 cores), macOS 27.2.1, AC power. Heavy user applications and iCloud sync were temporarily suspended during this run and resumed afterward. macOS background services remained active. This is a desktop benchmark, not an isolated performance lab. See `environment.json` for the recorded conditions. An earlier battery-powered run is excluded from this report.', '',
'Both variants use ptts `e5110afd3ead725c940b6411eec4985fb8b111aa`, identical benchmark instrumentation, the same Q8 model, default voice, seed 42, and CPU inference. Baseline xn 0.2.13 was published from `4242d4445ffafd8ed77380d30890db057102cb0c`. The five commits between that revision and #138\'s parent `135b5fdbb3b8c9690b0207269454315e10cafdf2` change CUDA code and version metadata, not the CPU paths exercised here. This remains a released-baseline comparison, not a build of the exact parent revision.', '',
'## Method', '',
'- 1,440 measured synthesis requests: 2 backends x 2 thread counts x 3 texts x 2 variants x 3 fresh-process trials x 20 requests.',
'- Text order rotates and variant order reverses between trials. Each cell below is a median of per-process statistics, not a pooled percentile.',
'- First TTFA is the first synthesis request in each process after model loading and voice conditioning. It is not application cold-start latency. Steady TTFA uses the remaining 19 requests.',
'- TTFA starts before `Synth::stream_with` and ends at the first nonempty chunk. Every timing record is checked to contain at least 1,920 samples in that chunk: one complete 80 ms frame at 24 kHz.',
'- Per-frame time is total synthesis duration divided by the number of 80 ms audio frames. It includes token generation and synthesis overhead; it is not a decoder-kernel-only measurement.',
'- JSON output is written after measurements. PCM capture is performed in a separate parity run. P95 values from 19 requests are included in raw statistics but are not treated as stable tail-latency estimates.', '',
'## Timing results', '',
'All times are milliseconds. Arrows mean baseline to #138. Positive improvement means lower steady TTFA.', '',
'| Backend | Threads | Text | First TTFA | Steady TTFA p50 | Improvement | Per-frame p50 |',
'|---|---:|---|---:|---:|---:|---:|']
for config in ('acckai','kai'):
 for threads in (1,4):
  for text in ('short','release','long'):
   a=lookup[config,threads,text,'baseline']; b=lookup[config,threads,text,'xn138']
   pair=lambda k:f"{a[k]:.3f} → {b[k]:.3f}"
   gain=100*(1-b['steady_ttfa_p50_ms']/a['steady_ttfa_p50_ms'])
   lines.append(f"| {'accelerate,kai' if config=='acckai' else 'kai'} | {threads} | {text} | {pair('first_ttfa_ms')} | {pair('steady_ttfa_p50_ms')} | {gain:.1f}% | {pair('steady_per_frame_p50_ms')} |")
lines += ['', '## Trial variability', '', 'Ranges below are observed ranges across three processes, not confidence intervals. Positive paired improvements favor #138. Small differences overlapping trial variability should be treated cautiously.', '', '| Backend | Threads | Text | Paired steady TTFA improvement range | Baseline p50 range (ms) | #138 p50 range (ms) |', '|---|---:|---|---:|---:|---:|']
for config in ('acckai','kai'):
 for threads in (1,4):
  for text in ('short','release','long'):
   vals={v:[next(r['steady_ttfa_p50_ms'] for r in trials if (r['config'],r['threads'],r['text'],r['variant'],r['trial'])==(config,threads,text,v,i)) for i in range(3)] for v in ('baseline','xn138')}
   gains=[100*(1-b/a) for a,b in zip(vals['baseline'],vals['xn138'])]
   ranges=[f"{min(vals[v]):.3f} to {max(vals[v]):.3f}" for v in ('baseline','xn138')]
   lines.append(f"| {config} | {threads} | {text} | {min(gains):.1f}% to {max(gains):.1f}% | {ranges[0]} | {ranges[1]} |")
lines += ['', '## Output comparison', '', 'Three texts x two backends x two thread counts. Each process performs two unmeasured warmups before capturing one output. Baseline repeats estimate run-to-run numerical variation. The cache-off ablation uses the same #138 binary with `GEMM_PACKED_LHS_CACHE_MB=0`.', '', '| Comparison | Matching lengths | Bit-identical cases | Maximum absolute sample difference | Minimum finite SNR |', '|---|---:|---:|---:|---:|']
for v,label in [('baseline-repeat','Baseline repeat vs baseline'),('xn138','#138 vs baseline'),('pr322','Original #322 vs baseline'),('xn138-vs-cache-off','#138 cache on vs cache off')]:
 records=[r[v] for r in parity]
 assert all(r['same_length'] for r in records)
 finite=[r['snr_db'] for r in records if r['snr_db'] is not None]
 lines.append(f"| {label} | {len(records)}/{len(records)} | {sum(r['bit_identical'] for r in records)}/{len(records)} | {max(r['max_abs_error'] for r in records):.9g} | {min(finite):.2f} dB |" if finite else f"| {label} | {len(records)}/{len(records)} | {len(records)}/{len(records)} | 0 | Infinite |")
lines += ['', 'All captured outputs were checked for finite samples. Numerical closeness is not a perceptual quality or intelligibility evaluation. This run does not measure cache memory overhead, cache eviction under pressure, concurrent-client throughput, or other hardware.', '', '## Artifacts', '', '- [Reproduction instructions](REPRODUCE.md), [runner](run.py), and [report generator](summarize.py).', '- `manifest.json`: source revisions, model/binary hashes, and executed commands. Local path prefixes are normalized to `$RESULTS` and `$MODEL`.', '- `timing/`: per-process raw JSON and logs, all observations, and summary statistics.', '- `parity/report.json`: every numerical comparison; `parity/pcm-sha256.json`: captured PCM hashes. Model weights and native binaries are not included.', '']
(p/'README.md').write_text('\n'.join(lines))
