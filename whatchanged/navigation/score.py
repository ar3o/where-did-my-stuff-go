"""Score navigation verdicts against by-eye labels (<runs>/navigation/truth/truth.json, never read by the robot).

    python -m whatchanged.navigation.score --results runs/navigation/RESULTS_config1.json [--name label]

truth.json maps each target to {episode number: label}; the run of target X on episode N is named epN_X. Each verdict is
mapped to the state it claims about the target's remembered place:
    here     confirmed / ready to pick up
    gone     not here / moved / not at its place
    covered  something new stands in front of its place
    abstain  could not see it / no path / not visible
A verdict is correct when its state equals the label, wrong when it names a different state, and an abstention
otherwise. Saying 'here' about something that is gone is the costly error for a robot, so it is counted separately.
Baseline 'memory only': the robot trusts its memory and answers 'here' every time (a robot without the episode view).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whatchanged.navigation.common import RUNS

TRUTH = RUNS / "truth" / "truth.json"

CLAIM = {"confirmed": "here", "reached": "here", "not_here": "gone", "moved": "gone", "reached_moved": "gone"}


def claim(v: dict) -> str:
    """State a verdict (a run row or a verdict dict) claims about the target's remembered place."""
    k = v.get("verdict") or v.get("kind")
    if k in CLAIM:
        return CLAIM[k]
    if k == "unverified" and (
        "in front of its place" in v.get("text", "") or "covered by something new" in v.get("text", "")
    ):
        return "covered"
    return "abstain"


def score(rows: dict[str, str], truth: dict) -> dict:
    """rows: run name -> claimed state. Returns counts of correct / wrong / abstain / false 'here' and per-run
    results."""
    out = dict(correct=0, wrong=0, abstain=0, false_here=0, n=0, per_run={})
    for tgt, eps in truth.items():
        if tgt.startswith("_") or tgt == "details":
            continue
        for ep, lab in eps.items():
            run = f"ep{ep}_{tgt}"
            c = rows.get(run, "abstain")
            if c == "covered" and lab != "covered":  # 'something new is in front of its place' says it could not verify
                res = "abstain"
            else:
                res = "abstain" if c == "abstain" else ("correct" if c == lab else "wrong")
            out[res] += 1
            out["n"] += 1
            if c == "here" and lab != "here":
                out["false_here"] += 1
            out["per_run"][run] = dict(truth=lab, claim=c, result=res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="RESULTS_<scene>.json written by collect.py")
    ap.add_argument("--truth", default=str(TRUTH))
    ap.add_argument("--name", default="current")
    ap.add_argument("--out", default=str(RUNS / "score.json"))
    a = ap.parse_args()
    truth = json.loads(Path(a.truth).read_text())
    res = json.loads(Path(a.results).read_text())
    rows = {r["run"]: claim(r) for r in res}
    allres = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    allres[a.name] = score(rows, truth)
    allres["memory only"] = score({r: "here" for r in rows}, truth)
    Path(a.out).write_text(json.dumps(allres, indent=1))
    for k, v in allres.items():
        print(
            f"{k:28s} correct {v['correct']:2d}/{v['n']}  wrong {v['wrong']:2d}  abstain {v['abstain']:2d}  "
            f"'here' when not here {v['false_here']}"
        )


if __name__ == "__main__":
    main()
