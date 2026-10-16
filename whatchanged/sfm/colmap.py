"""pycolmap structure from motion of the reference frames, with the query photos registered in the same model.

Frames share one PINHOLE camera; photos get one camera per image size with a focal prior from the 35 mm-equivalent
focal length, refined by bundle adjustment. Matching uses chosen pairs instead of all pairs:
  * each frame with the next 12 frames, every keyframe pair (keyframes span the walk, so the loop closes), and each
    frame with its 6 nearest keyframes in time;
  * each photo with every keyframe and with the next 5 photos (photos of one group are taken close together);
  * after that pass, each photo with the frames around the 6 keyframes it matched best (+-4 frames).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SEQUENTIAL_WINDOW = 12
NEAREST_KEYFRAMES = 6
PHOTO_NEIGHBOURS = 5
PHOTO_BEST_KEYFRAMES = 6
PHOTO_FRAME_RADIUS = 4
MAX_FEATURES = 8192
FULL_FRAME_DIAGONAL_MM = 43.2666


def focal_from_35mm(f35: float, w: int, h: int) -> float:
    return f35 / FULL_FRAME_DIAGONAL_MM * float(np.hypot(w, h))


def frame_pairs(names: list[str], kf_names: list[str]) -> set[tuple[str, str]]:
    pairs = set()
    for i in range(len(names)):
        for j in range(i + 1, min(len(names), i + SEQUENTIAL_WINDOW + 1)):
            pairs.add((names[i], names[j]))
    for i in range(len(kf_names)):
        for j in range(i + 1, len(kf_names)):
            pairs.add((kf_names[i], kf_names[j]))
    kidx = [names.index(k) for k in kf_names]
    for i, n in enumerate(names):
        for k in sorted(kidx, key=lambda k: abs(k - i))[:NEAREST_KEYFRAMES]:
            if names[k] != n:
                pairs.add(tuple(sorted((n, names[k]))))
    return pairs


def photo_pairs(photos: list[str], kf_names: list[str]) -> set[tuple[str, str]]:
    pairs = {tuple(sorted((p, k))) for p in photos for k in kf_names}
    for i in range(len(photos)):
        for j in range(i + 1, min(len(photos), i + PHOTO_NEIGHBOURS + 1)):
            pairs.add(tuple(sorted((photos[i], photos[j]))))
    return pairs


def _match(db: Path, sdir: Path, pairs: set, tag: str) -> None:
    import pycolmap

    pf = sdir / f"pairs_{tag}.txt"
    pf.write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs)))
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(pf)
    pycolmap.match_image_pairs(db, pairing_options=po, device=pycolmap.Device.cpu)


def best_keyframes(db_path: Path, photos: list[str], kf_names: list[str]) -> dict[str, list[str]]:
    """For each photo, the keyframes with the most verified inlier matches."""
    import pycolmap

    db = pycolmap.Database.open(str(db_path))
    name_of = {im.image_id: im.name for im in db.read_all_images()}
    pair_ids, counts = db.read_two_view_geometry_num_inliers()
    db.close()
    kf, ph = set(kf_names), set(photos)
    score = {p: {} for p in photos}
    for pid, n in zip(pair_ids, counts):
        a, b = (name_of[i] for i in pycolmap.pair_id_to_image_pair(pid))
        if a in ph and b in kf:
            score[a][b] = n
        elif b in ph and a in kf:
            score[b][a] = n
    return {p: sorted(s, key=lambda k: -s[k])[:PHOTO_BEST_KEYFRAMES] for p, s in score.items()}


def run_sfm(
    image_root: Path, sdir: Path, names: list[str], kf_names: list[str], frame_K: np.ndarray, photo_groups: list[dict]
):
    """names / kf_names / photo names are paths relative to image_root. photo_groups = [{names, K (fx, fy, cx, cy)}].
    Returns the largest reconstruction and timings."""
    import pycolmap

    sdir.mkdir(parents=True, exist_ok=True)
    db = sdir / "database.db"
    if db.exists():
        db.unlink()
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = MAX_FEATURES
    t0 = time.time()
    cameras = [dict(names=names, K=(frame_K[0, 0], frame_K[1, 1], frame_K[0, 2], frame_K[1, 2]))] + photo_groups
    for group in cameras:
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = ",".join(f"{v:.3f}" for v in group["K"])
        pycolmap.extract_features(
            db,
            image_root,
            image_names=group["names"],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=ro,
            extraction_options=eo,
            device=pycolmap.Device.cpu,
        )
    t1 = time.time()
    photos = [n for g in photo_groups for n in g["names"]]
    _match(db, sdir, frame_pairs(names, kf_names) | photo_pairs(photos, kf_names), "first")
    if photos:
        extra = set()
        for p, kfs in best_keyframes(db, photos, kf_names).items():
            for k in kfs:
                i = names.index(k)
                for j in range(max(0, i - PHOTO_FRAME_RADIUS), min(len(names), i + PHOTO_FRAME_RADIUS + 1)):
                    extra.add(tuple(sorted((p, names[j]))))
        _match(db, sdir, extra, "photos")
    t2 = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_principal_point = False
    recs = pycolmap.incremental_mapping(db, image_root, sdir, opts)
    t3 = time.time()
    if not recs:
        raise RuntimeError("SfM produced no model")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    rec.write(str(sdir))
    return rec, dict(
        extract_s=round(t1 - t0, 1), match_s=round(t2 - t1, 1), map_s=round(t3 - t2, 1), n_models=len(recs)
    )


def cam_from_world(img) -> np.ndarray:
    M = np.eye(4)
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    M[:3, :4] = np.asarray(cfw.matrix())
    return M


def camera_K(cam) -> np.ndarray:
    fx, fy, cx, cy = cam.params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def reprojection_errors(rec, img) -> list[float]:
    K = camera_K(rec.cameras[img.camera_id])
    M = cam_from_world(img)
    errs = []
    for p2 in img.points2D:
        if not p2.has_point3D():
            continue
        Xc = M[:3, :3] @ rec.points3D[p2.point3D_id].xyz + M[:3, 3]
        uv = K[:2, :2] @ (Xc[:2] / Xc[2]) + K[:2, 2]
        errs.append(float(np.linalg.norm(uv - np.asarray(p2.xy))))
    return errs
