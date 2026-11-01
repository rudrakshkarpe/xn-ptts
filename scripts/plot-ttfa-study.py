#!/usr/bin/env python3
"""Render the recorded study; never generates or invents benchmark observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, stem):
    for ext in ["png", "svg"]:
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=180, metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def render(root):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "phonon-ttfa-study"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, name, title in zip(axes, ["final-4threads", "final-1thread"], ["Four CPU threads", "One CPU thread"]):
        summary = json.loads((root / name / "summary.json").read_text())
        x = np.arange(3)
        for variant, offset, color, label in [("baseline", -.18, "#8795a5", "Upstream"),
                                             ("candidate", .18, "#168080", "Skip unused outputs")]:
            vals = [summary[k][variant]["median_trial_ttfa_p50_ms"] for k in ["short", "release", "long"]]
            ax.bar(x+offset, vals, width=.33, color=color, label=label)
            for i, k in enumerate(["short", "release", "long"]):
                trials = summary[k][variant]["trial_ttfa_p50_ms"]
                ax.scatter(np.full(3, i+offset), trials, s=17, facecolors="white", edgecolors="#263849", zorder=3)
                ax.text(i+offset, max(trials)+.13, f"{vals[i]:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x, ["Short greeting", "Release sentence", "Long message"])
        ax.set_title(title)
        ax.set_ylim(0, 7)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Time to first audio (ms; lower is better)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Ordinary warm-voice synthesis: removing discarded prefill work", weight="bold")
    fig.text(.5, .02, "Bars: median of three trial medians. Dots: every trial. 30 requests per trial; no profiling during timing.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, root / "ttfa-comparison")
    plt.close(fig)

    stages = json.loads((root / "baseline-stages.json").read_text())["stages"]
    selected = [("normalize_tokenize_plan", "Normalize, tokenize, plan"), ("session_setup", "Session + KV setup"),
                ("mimi_state_init", "Mimi state setup"), ("text_prefill", "Text prefill"),
                ("flow_transformer", "First flow transformer step"), ("flow_sampling", "First flow sampling"),
                ("mimi_decode", "First Mimi decode"), ("audio_readback", "Audio readback")]
    values = [stages[k]["median_ms"] for k, _ in selected]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh([label for _, label in selected], values, color=["#168080" if k in ["text_prefill", "mimi_decode"] else "#8795a5" for k, _ in selected])
    for i, v in enumerate(values):
        ax.text(v+.025, i, f"{v:.3f} ms", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.22)
    ax.set_xlabel("Median host duration of the first occurrence (ms)")
    ax.set_title("Where first-audio time goes on this Mac", weight="bold")
    fig.text(.5, .02, "Upstream diagnostic trace: 10 measured requests, four threads. Excludes HTTP; not a kernel-level latency breakdown.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, root / "first-audio-stages")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    render(parser.parse_args().results)
