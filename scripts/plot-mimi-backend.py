#!/usr/bin/env python3
"""Plot the final owned-backend cohort; bars are medians of trial p50s."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def run(root):
    data = json.loads((root / "final/summary.json").read_text())
    variants = ["baseline", "packed", "packed-conv-small"]
    labels = ["Baseline", "Packed linears", "+ Selective convolution"]
    colors = ["#64748b", "#0891b2", "#7c3aed"]
    texts = ["short", "release", "long"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric,title,unit,margin in zip(axes,["t80_ms","rtfx"],["Time to receive the complete first 80 ms","Full-generation speed"],["Milliseconds · lower is better","Audio seconds per wall second · higher is better"],[.12,1.5]):
        x = np.arange(3)
        maximum = 0
        for i,(v,label,color) in enumerate(zip(variants,labels,colors)):
            values = [data[t][v][metric]["p50"] for t in texts]
            pos = x+(i-1)*.24
            ax.bar(pos,values,.22,label=label,color=color)
            for p,t,value in zip(pos,texts,values):
                trials = data[t][v][metric]["trial_p50"]
                ax.scatter([p-.025,p,p+.025],trials,color="#111827",s=12,zorder=5)
                top = max(trials+[value])+margin
                ax.text(p,top,f"{value:.2f}" if metric=="t80_ms" else f"{value:.1f}",ha="center",fontsize=8)
                maximum = max(maximum,top)
        ax.set_xticks(x,["Short","Release","Long"])
        ax.set_title(title,fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0,maximum*1.13)
    axes[0].legend(loc="lower left",fontsize=9)
    fig.suptitle("Owned weight packing + selective convolution · Apple M4 Pro",fontsize=14)
    fig.text(.5,.02,"Four CPU threads · 3 trials × 20 requests per cell · dots show individual trial medians.\nAll variants deliver the same full first frame; no partial-frame delivery or reduced model precision.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.94])
    fig.savefig(root/"results.png",dpi=170)
    fig.savefig(root/"results.svg")
    p=root/"results.svg";p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    run(parser.parse_args().root)
