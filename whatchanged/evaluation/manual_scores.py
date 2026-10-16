"""Scores of a hand review of reported changes: per-method tables, a per-group CSV and a bar chart.

    python -m whatchanged.evaluation.manual_scores --scores examples/manual_review/scores.json \
        --key examples/manual_review/key.json --groups 1-13 --out results/manual_scores

Review protocol. For each change set the top events of every method (at most 4 per method, by the method's own score)
are mixed in one anonymous list. Events of the same type whose 3D boxes overlap (IoU >= 0.3, or one inside the other)
are merged automatically with the same rule for every method. The scorer gives each event one verdict:
  c  correct object, change type and name
  t  correct object, wrong change type
  n  correct object, wrong name
  d  duplicate of another listed event (dup_of), possibly an event of another method that reports the same change
  w  nothing like that changed
  u  unsure
Definitions used here. A correct detection (TP) is c, t or n. A duplicate of an event that was itself scored c, t or n
is the same real change reported by a second method, so it counts as a correct detection for its own method but is left
out of the type and name accuracy, because only the original card was judged for those. A duplicate of an event of the
same method is redundancy and is left out of precision. Unsure counts as not correct.
  precision        correct detections / (correct detections + w + u)
  recall (pooled)  distinct real changes a method found / distinct real changes found by any method. Duplicate links
                   join cards into one real change; a change that two methods both scored c without a link counts twice.
  type accuracy    (c + n) / (c + t + n)         name accuracy  (c + t) / (c + t + n)
  redundancy       same-method duplicates / scored events
Wilson 95 % intervals; events are clustered in change sets, so the intervals are narrower than they should be.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METHODS = ["ours", "scenediff", "rules"]
LABEL = {"ours": "Full pipeline", "scenediff": "Official SceneDiff", "rules": "Rules (phrase pairing on the inventory)"}
COLOR = {"ours": "#2a78d6", "scenediff": "#eb6834", "rules": "#1baf7a"}
TP_VERDICTS = ("c", "t", "n")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(t: tuple[float, float, float]) -> str:
    if t[0] != t[0]:
        return "n/a"
    return f"{100 * t[0]:.0f} % ({100 * t[1]:.0f}-{100 * t[2]:.0f})"


def parse_groups(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def normalise(score: dict) -> str:
    """Verdict with the older 'correct, type not ok' form read as wrong type."""
    v = score["verdict"]
    return "t" if v == "c" and score.get("type_ok") is False else v


def components(events: dict, scores: dict) -> dict[str, str]:
    """Event id -> id of the first event of its real change, following duplicate links."""
    root = {}
    for eid in events:
        cur, seen = eid, set()
        while scores.get(cur, {}).get("verdict") == "d" and scores[cur].get("dup_of") in events and cur not in seen:
            seen.add(cur)
            cur = scores[cur]["dup_of"]
        root[eid] = cur
    return root


def analyse(scores: dict, key: dict, groups: set[int]) -> dict:
    events = {e: k for e, k in key["events"].items() if k["group"] in groups and e in scores}
    root = components(events, scores)
    stats = {m: dict(c=0, t=0, n=0, w=0, u=0, dup_same=0, dup_other=0, found=set(), scored=0) for m in METHODS}
    per_group = {}
    pooled = set()
    for eid, k in events.items():
        m, g = k["method"], k["group"]
        v = normalise(scores[eid])
        s = stats[m]
        s["scored"] += 1
        pg = per_group.setdefault(g, {mm: dict(tp=0, wrong=0, unsure=0, dup=0, scored=0) for mm in METHODS})[m]
        pg["scored"] += 1
        if v in TP_VERDICTS:
            s[v] += 1
            s["found"].add(root[eid])
            pooled.add(root[eid])
            pg["tp"] += 1
        elif v == "d":
            target = scores[eid].get("dup_of")
            tv = normalise(scores[target]) if target in scores else None
            if target in events and events[target]["method"] != m and tv in TP_VERDICTS:
                s["dup_other"] += 1
                s["found"].add(root[eid])
                pg["tp"] += 1
            else:
                s["dup_same"] += 1
                pg["dup"] += 1
        elif v == "w":
            s["w"] += 1
            pg["wrong"] += 1
        else:
            s["u"] += 1
            pg["unsure"] += 1
    res = {}
    for m in METHODS:
        s = stats[m]
        direct = s["c"] + s["t"] + s["n"]
        tp = direct + s["dup_other"]
        res[m] = dict(
            scored=s["scored"],
            correct=s["c"],
            wrong_type=s["t"],
            wrong_name=s["n"],
            duplicate_of_other_method=s["dup_other"],
            duplicate_same_method=s["dup_same"],
            wrong=s["w"],
            unsure=s["u"],
            found=len(s["found"]),
            precision=wilson(tp, tp + s["w"] + s["u"]),
            recall=wilson(len(s["found"]), len(pooled)),
            type_accuracy=wilson(s["c"] + s["n"], direct),
            name_accuracy=wilson(s["c"] + s["t"], direct),
            redundancy=wilson(s["dup_same"], s["scored"]),
        )
    return dict(
        methods=res, pooled_changes=len(pooled), per_group=per_group, n_events=len(events), groups=sorted(groups)
    )


def tables(r: dict) -> str:
    rows = [
        "| Method | Events scored | Precision (95 % CI) | Recall, pooled (95 % CI) | Type accuracy (95 % CI) |"
        " Name accuracy (95 % CI) | Redundancy (95 % CI) |",
        "|:--|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for m in METHODS:
        x = r["methods"][m]
        rows.append(
            f"| {LABEL[m]} | {x['scored']} | {fmt(x['precision'])} | "
            f"{fmt(x['recall'])} ({x['found']} of {r['pooled_changes']}) | "
            f"{fmt(x['type_accuracy'])} | {fmt(x['name_accuracy'])} | {fmt(x['redundancy'])} |"
        )
    rows += [
        "",
        "| Method | Correct | Wrong type | Wrong name | Duplicate of another method's correct event | "
        "Duplicate within the method | "
        "Nothing like that changed | Unsure |",
        "|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for m in METHODS:
        x = r["methods"][m]
        rows.append(
            f"| {LABEL[m]} | {x['correct']} | {x['wrong_type']} | {x['wrong_name']} | "
            f"{x['duplicate_of_other_method']} | "
            f"{x['duplicate_same_method']} | {x['wrong']} | {x['unsure']} |"
        )
    return "\n".join(rows)


def chart(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall (pooled)"),
        ("type_accuracy", "Change-type accuracy"),
        ("name_accuracy", "Name accuracy"),
        ("redundancy", "Redundancy"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.4), facecolor="white")
    w = 0.26
    for j, m in enumerate(METHODS):
        for i, (key, _) in enumerate(metrics):
            p, lo, hi = r["methods"][m][key]
            if p != p:
                continue
            x = i + (j - 1) * (w + 0.03)
            ax.bar(x, p, w, color=COLOR[m], label=LABEL[m] if i == 0 else None, zorder=2)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color="#333333", capsize=3, lw=1, zorder=3)
            ax.text(x, min(hi, 1.0) + 0.02, f"{100 * p:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([t for _, t in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#e5e5e0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--groups", default="1-13", help="change sets to include, e.g. 1-13")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chart", type=Path, help="where to write the bar chart (default: <out>/manual_scores.png)")
    args = ap.parse_args()
    scores = json.loads(args.scores.read_text())["events"]
    key = json.loads(args.key.read_text())
    r = analyse(scores, key, parse_groups(args.groups))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text(tables(r) + "\n")
    (args.out / "results.json").write_text(json.dumps(r, indent=1, default=list))
    with open(args.out / "per_group.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["set"]
            + [f"{m}_{c}" for m in METHODS for c in ("correct_detections", "wrong", "unsure", "duplicates", "scored")]
        )
        for g in sorted(r["per_group"]):
            w.writerow(
                [g] + [r["per_group"][g][m][c] for m in METHODS for c in ("tp", "wrong", "unsure", "dup", "scored")]
            )
    chart(r, args.chart or args.out / "manual_scores.png")
    print(tables(r))
    print(
        f"{r['n_events']} events in sets {r['groups'][0]}-{r['groups'][-1]}, "
        f"{r['pooled_changes']} distinct real changes"
    )


if __name__ == "__main__":
    main()
