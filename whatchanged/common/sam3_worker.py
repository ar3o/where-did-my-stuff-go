"""SAM 3 image model (official facebook/sam3 checkpoint) with text prompts, one model load for all images; SAM 3 env.

    PYTHONPATH=<sam3 repo>:<repo> python -m whatchanged.common.sam3_worker jobs.json

jobs.json = {"prompts": [...], "min_p": 0.35, "topk": 6, "checkpoint": "sam3.pt", "need_mib": 5000,
             "items": [{"image": "a.jpg", "out": "a.npz", "mask_wh": [W, H]}]}

Per image: prompt index, score, box (xyxy, image px) and packed masks resized (area) to mask_wh. Items whose output
exists are skipped. Weights stay fp32 under bf16 autocast (a bf16 trunk returned no detections); the text encoder runs
on the CPU and every phrase is encoded once.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, save_npz


def load_model(checkpoint: str, min_p: float, device: str = "cuda"):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint, load_from_HF=False)
    language = model.backbone.language_backbone
    model.backbone.language_backbone = None
    model.to(device)
    model.backbone.language_backbone = language
    encode_text = model.backbone.forward_text
    target_device = device
    cache = {}

    def forward_text_cpu(captions, input_boxes=None, additional_text=None, device="cuda", **kw):
        key = tuple(captions)
        if key not in cache:
            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(device_type="cpu", enabled=False):
                out = encode_text(captions, input_boxes=input_boxes, additional_text=additional_text, device="cpu")
            cache[key] = {k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in out.items()}
        return dict(cache[key])

    model.backbone.forward_text = forward_text_cpu
    torch.cuda.empty_cache()
    return Sam3Processor(model, device=device, confidence_threshold=min_p)


def detect(processor, image: str, prompts: list[str], mask_wh, min_p: float, topk: int) -> dict:
    rgb = np.ascontiguousarray(cv2.imread(image)[..., ::-1])
    W, H = mask_wh
    P, S, B, M = [], [], [], []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(torch.from_numpy(rgb).permute(2, 0, 1))
        for p, phrase in enumerate(prompts):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(phrase, state)
            probs = state["scores"].float().cpu().numpy()
            if len(probs) == 0:
                continue
            masks = torch.nn.functional.interpolate(state["masks"].float(), (H, W), mode="area")[:, 0] > 0.5
            masks = masks.cpu().numpy()
            boxes = state["boxes"].float().cpu().numpy()
            for j in np.argsort(-probs)[:topk]:
                if probs[j] < min_p or not masks[j].any():
                    continue
                P.append(p)
                S.append(probs[j])
                B.append(boxes[j])
                M.append(masks[j])
    m = len(P)
    return dict(
        prompt=np.array(P, np.int16),
        score=np.array(S, np.float32),
        box=np.array(B, np.float32).reshape(m, 4),
        masks=np.packbits(np.array(M, bool).reshape(m, H, W)) if m else np.zeros(0, np.uint8),
        shape=np.array([m, H, W]),
        prompts=np.array(prompts),
    )


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    prompts, min_p, topk = jobs["prompts"], float(jobs["min_p"]), int(jobs["topk"])
    items = [it for it in jobs["items"] if not Path(it["out"]).exists()]
    if not items:
        return
    with gpu_stage("sam3", need_mib=int(jobs.get("need_mib", 5000))):
        t0 = time.time()
        processor = load_model(jobs["checkpoint"], min_p)
        log(f"sam3 loaded in {time.time() - t0:.0f} s: {len(items)} images x {len(prompts)} phrases")
        for it in items:
            t1 = time.time()
            out = detect(processor, it["image"], prompts, it["mask_wh"], min_p, topk)
            save_npz(it["out"], **out, seconds=time.time() - t1)
            log(f"  {Path(it['image']).name}: {int(out['shape'][0])} detections, {time.time() - t1:.1f} s")


if __name__ == "__main__":
    main()
