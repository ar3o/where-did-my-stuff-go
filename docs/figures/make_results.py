"""Result graphs on a white background: calibration, edit recall by type, and false alarms against the baselines.

    python docs/figures/make_results.py --calibration <runs>/<scene>/checker/conf_calibration.json
The comparison numbers are the ones in docs/results.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "assets"
BLUE, RED, GREY, GREEN = "#3b7fc4", "#c4506f", "#9a9a9a", "#58a97a"
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}
)


def calibration(path: Path) -> None:
    types = json.loads(path.read_text())["types"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, t) in zip(axs, types.items()):
        pts = [(c["mean_p"], c["frac_pos"], c["n"]) for c in t["curve"]]
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=BLUE)
        ax.set_title(f"{name}: ECE {t['ece_out_of_fold']:.3f}  (n+ {t['n_pos']}, n- {t['n_neg']})", fontsize=11)
        ax.set_xlabel("predicted probability (out of fold)")
        ax.set_ylabel("observed fraction of true edits")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "result_calibration.png", dpi=130)
    plt.close(fig)


def comparison() -> None:
    methods = [
        "This pipeline",
        "Learned checker\n+ SAM masks",
        "SAM mask\nmatching",
        "Depth blobs",
        "Zero-shot rules\n(Pi3X)",
        "SceneDiff",
        "VLM judge",
    ]
    false_alarms = [0, 0, 0, 0, 0, 16, 21]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    cols = [GREEN, BLUE, BLUE, BLUE, BLUE, RED, RED]
    ax.bar(range(len(methods)), false_alarms, color=cols)
    for i, v in enumerate(false_alarms):
        ax.text(i, v + 0.4, str(v), ha="center")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("changes reported on the 5 untouched photos")
    ax.set_title("False alarms on the untouched room (no change is real)", fontsize=11)
    ax.set_ylim(0, 25)
    fig.tight_layout()
    fig.savefig(OUT / "result_false_alarms.png", dpi=130)
    plt.close(fig)


def recall_by_edit() -> None:
    rows = [
        ("remove", 0.981),
        ("add", 0.979),
        ("move 15-60 cm", 0.964),
        ("move 2-10 cm", 0.857),
        ("rotate", 0.793),
        ("move 2-5 cm", 0.75),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=BLUE)
    for i, r in enumerate(rows[::-1]):
        ax.text(r[1] + 0.01, i, f"{r[1]:.2f}", va="center")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("pixel-level recall of held-out edits (191 edits)")
    ax.set_title("Learned checker on its own pixel maps", fontsize=11)
    ax = axs[1]
    ax.bar(["hand-written\nrules", "learned\nchecker"], [0.29, 0.49], color=[GREY, BLUE])
    for i, v in enumerate([0.29, 0.49]):
        ax.text(i, v + 0.015, f"{v:.2f}   ({[16, 0][i]} false on\nunedited pairs)", ha="center", fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("object recall")
    ax.set_title("Same 48 pairs, same object step", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "result_recall.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, required=True)
    args = ap.parse_args()
    calibration(args.calibration)
    comparison()
    recall_by_edit()
    print("wrote result graphs")


if __name__ == "__main__":
    main()
