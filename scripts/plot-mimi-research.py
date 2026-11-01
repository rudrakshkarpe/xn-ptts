#!/usr/bin/env python3
"""Render the committed Mimi measurements; no inferred or projected speedups."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def run(root):
    summary = json.loads((root / "final-confirmation/summary.json").read_text())
    variants = ["baseline", "packed", "packed-wave20"]
    labels = ["Baseline", "Packed SME2", "Packed + 20 ms delivery"]
    colors = ["#64748b", "#0891b2", "#7c3aed"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in zip(axes, ["t20_ms", "t80_ms"], ["Time to receive at least 20 ms of audio", "Time to receive at least 80 ms of audio"]):
        x = np.arange(2)
        for i, (v, label, color) in enumerate(zip(variants, labels, colors)):
            values = [summary[t][v][metric]["p50"] for t in ["short", "release"]]
            bars = ax.bar(x + (i-1)*.24, values, .22, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            for j, text in enumerate(["short", "release"]):
                ax.scatter([x[j]+(i-1)*.24]*3, summary[text][v][metric]["trial_p50"], s=12, color="#111827", zorder=5)
        ax.set_xticks(x, ["Short greeting", "Release sentence"])
        ax.set_ylabel("Milliseconds · lower is better")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 5.6)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Mimi first-frame experiments · Apple M4 Pro · 4 CPU threads", fontsize=14)
    fig.text(.5, .02, "Rebuilt candidate: 3 trials × 30 requests per cell. Bars: median of trial p50s; dots: individual trial p50s.\nPacked weights speed up the whole frame; splitting trades earlier 20 ms for later 80 ms.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.94])
    save(fig, root / "delivery-comparison")

    raw = json.loads((root / "kernel-probe.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    shapes = ["mimi_qkv", "mimi_ffn_expand", "mimi_ffn_contract"]
    x = np.arange(3)
    for i, (variant, label, color) in enumerate(zip(["accelerate_fortran", "sme2_repack", "sme2_prepacked"], ["Accelerate SGEMM", "SME2 · repack each call", "SME2 · weights already packed"], colors)):
        values = [statistics.median(r["mean_us"] for r in raw if r["shape"] == s and r["m"] == 16 and r["variant"] == variant) for s in shapes]
        bars = ax.bar(x+(i-1)*.24, values, .22, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, ["QKV projection", "FFN expansion", "FFN contraction"])
    ax.set_ylabel("Microseconds per call · lower is better")
    ax.set_title("Kernel probe: 16 rows · same f32 inputs and weights")
    ax.set_ylim(0, 122)
    ax.legend(fontsize=9)
    fig.text(.5,.02,"Median of 5 trial means; 200 calls/trial. Input packing included. Weight packing retained only in purple.\nThis is a kernel microbenchmark, not a 2× end-to-end TTS speedup.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.1,1,1])
    save(fig, root / "kernel-comparison")

def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".svg"))
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
