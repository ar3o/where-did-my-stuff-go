import json

import numpy as np
import pytest

from whatchanged import config
from whatchanged.changes import language, reid
from whatchanged.changes.infer import noisy_or
from whatchanged.common.io import pack_masks, save_npz, unpack_masks
from whatchanged.ingest import run as ingest_run
from whatchanged.ingest import video
from whatchanged.inventory import detect, naming, voting


def test_frame_selection_keeps_the_sharpest_frames_and_holds_out_extras():
    times = np.arange(200) / 10.0
    scores = np.full(200, 10.0)
    scores[::10] = 50.0
    stds = np.full(200, 30.0)
    stds[55] = 1.0  # a hand over the lens
    rows, n_ok = video.select_frames(
        times, scores, stds, n_keyframes=4, n_extra=16, fps=10.0, blur_frac=0.35, test_every=8
    )
    keys = [i for i, kf, _ in rows if kf >= 0]
    assert len(keys) == 4 and all(scores[i] == 50.0 for i in keys)
    extras = [i for i, kf, _ in rows if kf < 0]
    assert all(min(abs(times[i] - times[k]) for k in keys) >= 0.15 for i in extras)
    assert sum(split == "test" for _, _, split in rows) == 2
    assert n_ok == 199


def test_tone_mapping_only_for_hdr():
    assert "tonemap=hable" in video.tone_chain("arib-std-b67", 960, 540)
    assert video.tone_chain("bt709", 960, 540) == "scale=960:540:flags=area"


def test_groups_accept_both_formats(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"groups": [{"group": 0, "photos": ["IMG_1.HEIC", "IMG_2.HEIC"]}]}))
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"groups": {"3": ["IMG_7"]}}))
    assert ingest_run.read_groups(a, []) == {"0": ["IMG_1", "IMG_2"]}
    assert ingest_run.read_groups(b, []) == {"3": ["IMG_7"]}
    assert ingest_run.read_groups(None, ["x", "y"]) == {"0": ["x", "y"]}


def test_masks_pack_round_trip():
    rng = np.random.default_rng(0)
    m = rng.random((3, 7, 9)) > 0.5
    bits, shape = pack_masks(m)
    assert (unpack_masks(bits, shape) == m).all()


def write_dets(path, dets, H, W):
    prompts = sorted({p for p, _, _ in dets})
    bits, shape = pack_masks([m for _, _, m in dets]) if dets else (np.zeros(0, np.uint8), np.array([0, H, W]))
    save_npz(
        path,
        prompt=np.array([prompts.index(p) for p, _, _ in dets], np.int16),
        score=np.array([s for _, s, _ in dets], np.float32),
        masks=bits,
        shape=shape,
        prompts=np.array(prompts),
    )


def test_tile_detections_land_on_their_window_and_duplicates_go(tmp_path):
    W, H = 100, 60
    full = np.zeros((H, W), bool)
    full[50:58, 80:95] = True
    write_dets(tmp_path / "frame_a.npz", [("cup", 0.6, full)], H, W)
    x, y, tw, th = detect.tile_windows(W, H)[3]
    tile = np.zeros((th, tw), bool)
    tile[50 - y : 58 - y, 80 - x : 95 - x] = True
    other = np.zeros((th, tw), bool)
    other[:5, :5] = True
    write_dets(tmp_path / "frame_a_t3.npz", [("cup", 0.9, tile), ("bowl", 0.5, other)], th, tw)
    out = detect.merged(tmp_path / "frame_a", W, H)
    assert [(p, round(s, 2)) for p, s, _ in out] == [("cup", 0.9), ("bowl", 0.5)]
    assert out[1][2][y : y + 5, x : x + 5].all()


def test_name_votes_and_margin():
    assert voting.decide({"mug": 1.0, "cup": 0.9}, 0.25) == ("mug (uncertain)", pytest.approx(0.1))
    assert voting.decide({"mug": 1.0, "cup": 0.5}, 0.25)[0] == "mug"
    phrases = ["mug", "cup", "bowl"]
    r = naming.combine_probs({"mug": 2.0, "cup": 1.0}, np.array([0.1, 0.8, 0.1]), phrases, 0.25)
    assert r["name"] == "cup" and r["runner_up"] == "mug"
    assert naming.combine_probs({}, None, phrases, 0.25)["name"] == "object (uncertain)"


def test_logistic_fit_separates_and_ranks():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-1, 1, (200, 2)), rng.normal(1, 1, (200, 2))]
    y = np.r_[np.zeros(200), np.ones(200)]
    m = reid.fit_logit(X, y)
    p = reid.predict_logit(m, X)
    assert reid.auc(p, y) > 0.9
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([]) == 0.0


def test_text_comes_from_3d_facts():
    furniture = [
        dict(id="I1", name="dark wood coffee table", box=dict(centre=[2.0, 0, 0.2], size=[1.0, 0.6, 0.4], yaw=0))
    ]
    e = dict(
        kind="moved",
        object="I5",
        name="egg carton",
        change_conf=0.8,
        dist_m=0.4,
        centre_before=[0.0, 0, 0.1],
        centre_after=[1.6, 0, 0.45],
    )
    assert language.summary_line(e) == "egg carton moved 40 cm (80%)"
    s = language.sentence(e, furniture)
    assert s.startswith("The egg carton moved 40 cm, up 35 cm") and "coffee table" in s
    gone = dict(kind="removed", object="I6", name="banana (uncertain)", change_conf=0.99, centre_before=[9.0, 9, 0])
    assert language.sentence(gone, furniture) == "The banana? is gone (99 % sure)."


def test_config_reads_nested_settings_and_env(tmp_path, monkeypatch):
    from whatchanged.splat.settings import SplatSettings

    path = tmp_path / "splat.yaml"
    path.write_text("train:\n  iters: 10\nsupport:\n  min_views: $MIN_VIEWS\n")
    monkeypatch.setenv("MIN_VIEWS", "4")
    s = config.load(SplatSettings, "splat", path)
    assert s.train.iters == 10 and s.train.max_gaussians == 2_500_000 and s.support.min_views == 4
    monkeypatch.delenv("MIN_VIEWS")
    assert config.load(SplatSettings, "splat", path).support.min_views == 3


def test_repo_configs_match_code_defaults():
    from whatchanged.changes.run import ChangesConfig
    from whatchanged.checker.settings import CheckerSettings
    from whatchanged.ingest.run import IngestConfig
    from whatchanged.inventory.run import InventoryConfig
    from whatchanged.render.film import FilmConfig
    from whatchanged.sfm.run import SfmConfig
    from whatchanged.splat.settings import SplatSettings

    for cls, stage in (
        (IngestConfig, "ingest"),
        (SfmConfig, "sfm"),
        (SplatSettings, "splat"),
        (InventoryConfig, "inventory"),
        (CheckerSettings, "checker"),
        (ChangesConfig, "changes"),
        (FilmConfig, "render"),
    ):
        assert config.load(cls, stage) == cls(), stage
