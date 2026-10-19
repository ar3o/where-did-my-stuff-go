"""Task parsing: the prompt's object words -> inventory name(s) by SigLIP 2 text-embedding similarity (no word lists of
objects), and the intent (check / go) from generic command words.

Generic English command words are stripped first; the rest is the object phrase. The name is the inventory name with the
highest cosine("a photo of a {phrase}.", "a photo of a {name}.") over the distinct names of movable inventory objects
("(uncertain)" names count under their base name; "object (uncertain)" never), restricted to names with the same English
head noun (last word) when any exists, because SigLIP alone matched "blue horse" to "blue broom". The candidates are
every inventory object with that name; fragments of one physical thing (box centres within 10 cm) are merged into one
target instance group. Output: <run>/task.json.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.workers import run_worker
from whatchanged.navigation.common import inventory_sha, memory_paths

COMMAND_WORDS = [
    "please", "can you", "could you", "pick up", "pick", "bring me", "bring", "fetch", "grab", "get", "find", "go to",
    "walk to", "take", "look for", "search for", "where is", "show me", "me", "the", "a", "an", "my", "some", "to",
    "up",
]  # fmt: skip
CHECK_WORDS = ("check", "verify", "is here", "is there", "still", "make sure", "confirm")
CHECK_PHRASES = (
    r"\b(is here|is there|is still here|still here|here|there|ready to|get ready|make sure|check|verify|confirm|that)\b"
)


def object_phrase(prompt: str) -> str:
    s = re.sub(r"[^a-z0-9 \-]", " ", prompt.lower())
    s = " " + re.sub(r"\s+", " ", s).strip() + " "
    for w in sorted(COMMAND_WORDS, key=len, reverse=True):
        s = s.replace(f" {w} ", " ")
    return re.sub(r"\s+", " ", s).strip()


def base_name(n: str) -> str:
    return n.replace(" (uncertain)", "").strip()


def target_groups(objects_F: list[dict], cands: list[str], merge_m: float = 0.10) -> list[dict]:
    """Instances of the target: candidate inventory objects (memory-map objects, F frame) merged when their box centres
    are within merge_m (fragments of one physical thing). weight = share of Gaussians (tiny fragments carry little
    evidence)."""
    byid = {o["id"]: o for o in objects_F}
    cs = [byid[c] for c in cands if c in byid]
    groups = []
    for o in sorted(cs, key=lambda o: -o["n_gauss"]):
        c = np.array(o["box"]["centre"])
        for g in groups:
            if np.linalg.norm(np.array(g["centre"]) - c) < merge_m:
                g["ids"].append(o["id"])
                g["n_gauss"] += o["n_gauss"]
                break
        else:
            groups.append(
                dict(
                    ids=[o["id"]],
                    centre=o["box"]["centre"],
                    size=o["box"]["size"],
                    yaw=o["box"]["yaw"],
                    n_gauss=o["n_gauss"],
                    name=base_name(o["name"]),
                )
            )
    tot = sum(g["n_gauss"] for g in groups) or 1
    for k, g in enumerate(groups):
        g["key"] = f"T{k}"
        g["weight"] = round(g["n_gauss"] / tot, 3)
        g["gids"] = [int(i[1:]) for i in g["ids"]]
    return groups


def text_embeddings(texts: list[str], rd: Path) -> np.ndarray:
    """Unit SigLIP 2 text embeddings (vision env worker, CPU), cached in rd while the text list is unchanged."""
    jobs = rd / "text_jobs.json"
    out = rd / "text_emb.npz"
    if not out.exists() or not jobs.exists() or read_json(jobs).get("texts") != texts:
        run_worker("vision", "whatchanged.navigation.text_worker", dict(texts=texts, out=str(out)), jobs, need_mib=0)
    return np.load(out)["emb"]


def parse_prompt(scene: str, prompt: str, rd: Path) -> dict:
    """Intent and target of a prompt; raises when the target cannot be resolved to inventory objects."""
    from whatchanged.navigation.memory_map import load_map

    mem = memory_paths(scene)
    inv = read_json(mem["inv"])["objects"]
    names = sorted({base_name(o["name"]) for o in inv if o["cls"] != "static" and not o["name"].startswith("object")})
    pl = prompt.lower()
    intent = "check" if any(w in pl for w in CHECK_WORDS) else "go"
    phrase = object_phrase(re.sub(CHECK_PHRASES, " ", pl))
    rd.mkdir(parents=True, exist_ok=True)
    E = text_embeddings([phrase] + names, rd)
    sim = E[1:] @ E[0]
    head = phrase.split()[-1] if phrase.split() else ""
    same_head = np.array(
        [n.split()[-1] == head for n in names]
    )  # English head noun = last word ("blue horse" -> horse)
    rank = sim + (2.0 * same_head if same_head.any() else 0.0)
    order = np.argsort(-rank)
    name = names[order[0]]
    cands = [o["id"] for o in inv if o["cls"] != "static" and base_name(o["name"]) == name]
    src = "SigLIP 2 text + head noun over the current inventory names"
    tp = mem["inv"].parent / "targets.json"
    if tp.exists():  # the inventory builder's own target list (same inventory), if any
        T = read_json(tp)
        items = T.get("targets", T) if isinstance(T, dict) else T
        items = list(items.values()) if isinstance(items, dict) else items
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("name", "")).lower()
            ids = it.get("ids") or ([it["id"]] if "id" in it else [])
            if nm and (nm.split()[-1] == head or base_name(nm) == name) and ids:
                cands, name, src = list(ids), base_name(nm), f"targets.json ({tp})"
                break
    _, info = load_map(scene, ensure_current=True)
    groups = target_groups(info["objects"], cands)
    byid = {o["id"]: o for o in inv}
    bad = [
        c
        for c in cands
        if c not in byid
        or (head and base_name(byid[c]["name"]).split()[-1] != head and base_name(byid[c]["name"]) != name)
    ]
    bad += [g["key"] for g in groups if base_name(g["name"]) != base_name(byid[g["ids"][0]]["name"])]
    if not cands or not groups or bad:
        raise RuntimeError(
            f"prompt '{prompt}': target resolution failed (name '{name}', head '{head}', candidates {cands}, "
            f"groups {[(g['ids'], g['name']) for g in groups]}, mismatching {bad}); aborting the run"
        )
    t = dict(
        prompt=prompt,
        intent=intent,
        phrase=phrase,
        name=name,
        resolved_by=src,
        inventory=str(mem["inv"]),
        inventory_sha=inventory_sha(scene),
        cosine=round(float(sim[order[0]]), 4),
        runner_up=names[order[1]],
        margin=round(float(sim[order[0]] - sim[order[1]]), 4),
        head_noun_match=bool(same_head.any()),
        candidates=cands,
        groups=groups,
    )
    write_json(t, rd / "task.json")
    log(
        f"task: '{prompt}' -> {intent} '{name}' (phrase '{phrase}', cos {t['cosine']}, margin {t['margin']}); "
        f"{len(groups)} instance(s)"
    )
    return t
