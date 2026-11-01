"""Render the checked-in evidence; no inference, filtering, or synthesized timings.

Run with numpy and matplotlib installed: python generate_figures.py
"""
from pathlib import Path
import json
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
COLORS = {"upstream": "#3976bc", "optimized": "#d77836"}
WORKLOADS = [("warm-one", "One warm voice"), ("voice-pressure", "Voice pressure"), ("round-robin-17", "Round-robin 17")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelcolor": "#263247", "text.color": "#263247", "axes.edgecolor": "#b9c2ce", "figure.facecolor": "white", "savefig.facecolor": "white"})


def report(workload, variant, trial):
    return json.loads((RESULTS / f"{workload}-{variant}-{trial}" / "report.json").read_text())["summary"]


def save(fig, name, caption):
    fig.text(.055, .015, caption, fontsize=9, color="#566379")
    fig.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def paired_bars(ax, metric, stat=None):
    for offset, variant, label in [(-.18, "upstream", "Original"), (.18, "optimized", "Optimized")]:
        vals = []
        for workload, _ in WORKLOADS:
            rows = [report(workload, variant, n)[metric] for n in (1, 2, 3)]
            vals.append([r[stat] for r in rows] if stat else rows)
        medians = [np.median(v) for v in vals]
        positions = np.arange(3) + offset
        ax.bar(positions, medians, width=.32, color=COLORS[variant], alpha=.8, label=label, zorder=2)
        for x, median, trials in zip(positions, medians, vals):
            ax.scatter(x + np.array([-.055, 0, .055]), trials, s=28, color=COLORS[variant], edgecolor="white", linewidth=.7, zorder=3)
            ax.text(x, max(trials) + ax.get_ylim()[1] * .025, f"{median:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(3), [name for _, name in WORKLOADS])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.17, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
for ax, stat, title in zip(axes, ("p50", "p95"), ("Median first-audio latency", "95th-percentile first-audio latency")):
    ax.set_ylim(0, 16)
    paired_bars(ax, "client_ttfa_ms_completed", stat)
    ax.set_title(title, loc="left", pad=16)
    ax.set_ylabel("Milliseconds · lower is better")
    ax.legend(frameon=False, loc="upper left")
fig.suptitle("Voice-cache eviction affects startup latency under pressure", x=.06, ha="left", fontsize=17, weight="bold")
fig.tight_layout(rect=(0, .075, 1, .94))
save(fig, "latency", "M4 Pro · Q8 · 4 threads · concurrency 1. Bars: median of 3 trial statistics; dots: individual trials. First complete PCM sample over localhost HTTP.")

fig, ax = plt.subplots(figsize=(10, 4.9))
ax.set_ylim(0, 76)
paired_bars(ax, "audio_seconds_per_wall_second")
ax.set_title("Audio throughput: the benefit depends on voice reuse", loc="left", fontsize=17, pad=18)
ax.set_ylabel("Audio seconds / wall second\n(higher is better)")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout(rect=(0, .09, 1, 1))
save(fig, "throughput", "Same paired runs as latency. Successful audio / scenario wall time, including client overhead. These are not Gradium's model-only benchmark figures.")

fig, ax = plt.subplots(figsize=(12, 4.8))
for variant, label in [("upstream", "Original"), ("optimized", "Optimized")]:
    records = [json.loads(line) for line in (RESULTS / f"voice-pressure-{variant}-1" / "requests.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 96 and all(r["outcome"] == "completed" for r in measured)
    ax.plot(np.arange(1, 97), [r["ttfa_ms"] for r in measured], ".-", linewidth=1, markersize=4, color=COLORS[variant], label=label)
    if variant == "optimized":
        arrivals = [i + 1 for i, r in enumerate(measured) if r["voice"] in {"Imogen", "Elodie-Rose"}]
for i, x in enumerate(arrivals):
    ax.axvline(x, color="#9aa2ad", alpha=.35, linestyle=":", label="Occasional voice" if i == 0 else None)
ax.set_title("Cache-pressure trace · trial 1, all 96 measured requests", loc="left", fontsize=17, pad=18)
ax.set_xlabel("Request number (identical voice order in both runs)")
ax.set_ylabel("Time to first complete PCM sample (ms)")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, ncol=3)
ax.grid(axis="y", alpha=.17)
fig.tight_layout(rect=(0, .09, 1, 1))
save(fig, "request-trace", "One illustrative trial, not a pooled result. Dotted lines mark occasional voices from the input trace; per-request cache outcomes were not recorded.")

fig, axes = plt.subplots(2, 1, figsize=(12, 5.7), sharey=True)
audio_report = json.loads((HERE / "audio/report.json").read_text())
for ax, scenario, label in zip(axes, ("default", "cache-pressure"), ("Default voice · 4.24 seconds", "Pippa after cache pressure · 4.40 seconds")):
    samples = []
    for variant in ("original", "optimized"):
        with wave.open(str(HERE / "audio" / f"{variant}-{scenario}.wav"), "rb") as wav:
            assert wav.getframerate() == 24000 and wav.getsampwidth() == 2
            samples.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.int32))
    assert len(samples[0]) == len(samples[1])
    difference = samples[1] - samples[0]
    changed = np.flatnonzero(difference)
    ax.axhline(0, color="#b9c2ce", linewidth=1)
    ax.scatter(changed / 24000, difference[changed], s=12, color=COLORS["optimized"])
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(0, len(difference) / 24000)
    ax.set_ylabel("Difference (PCM steps)")
    ax.set_title(f"{label} · {len(changed):,}/{len(difference):,} samples differ", loc="left", fontsize=12)
    ax.set_xlabel("Time in audio (seconds)")
    assert int(np.abs(difference).max()) == audio_report["comparisons"][scenario]["max_abs_pcm_steps"]
fig.suptitle("Paired audio: differences are at most one 16-bit PCM step", x=.06, ha="left", fontsize=17, weight="bold")
fig.tight_layout(rect=(0, .08, 1, .95))
save(fig, "audio-difference", "Optimized minus original, with no alignment, trimming, or normalization. All omitted samples have zero difference. This is numerical validation, not a listening score.")
print("Rendered four figures as PNG and SVG from the saved records and WAV files.")
