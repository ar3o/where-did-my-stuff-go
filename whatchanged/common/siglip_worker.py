"""SigLIP 2 (google/siglip2-so400m-patch14-384) image embeddings and zero-shot phrase probabilities for crops;
vision env.

    python -m whatchanged.common.siglip_worker jobs.json

jobs.json = {"siglip": <weights>, "phrases": [...], "crops": [{"key": ..., "path": ...}], "out": "x.npz",
             "need_mib": 2500}

Prompt "a photo of a {phrase}.", softmax over the phrase list. Output npz: keys, emb (n, d) unit float16,
probs (n, phrases) float16, phrases. Keys already in `out` with the same phrase list are kept, not recomputed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, save_npz

BATCH = 32
TEXT_LEN = 64


def _features(out):
    return out if torch.is_tensor(out) else out.pooler_output


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    phrases = jobs["phrases"]
    done = {}
    if Path(jobs["out"]).exists():
        z = np.load(jobs["out"])
        if [str(p) for p in z["phrases"]] == phrases:
            done = {str(k): (z["emb"][i], z["probs"][i]) for i, k in enumerate(z["keys"])}
    queue = [c for c in jobs["crops"] if c["key"] not in done]
    if queue:
        from transformers import AutoModel, AutoProcessor

        with gpu_stage("siglip", need_mib=int(jobs.get("need_mib", 2500))):
            model = AutoModel.from_pretrained(jobs["siglip"], torch_dtype=torch.float16).cuda().eval()
            proc = AutoProcessor.from_pretrained(jobs["siglip"])
            texts = [f"a photo of a {p}." for p in phrases]
            with torch.no_grad():
                ti = proc(text=texts, padding="max_length", max_length=TEXT_LEN, return_tensors="pt").to("cuda")
                tf = torch.nn.functional.normalize(_features(model.get_text_features(**ti)).float(), dim=-1)
            t0 = time.time()
            for i in range(0, len(queue), BATCH):
                batch = queue[i : i + BATCH]
                ims = [Image.open(c["path"]).convert("RGB") for c in batch]
                with torch.no_grad():
                    ii = proc(images=ims, return_tensors="pt").to("cuda")
                    imf = _features(model.get_image_features(pixel_values=ii["pixel_values"].half()))
                    imf = torch.nn.functional.normalize(imf.float(), dim=-1)
                    logits = imf @ tf.T * model.logit_scale.exp().float() + model.logit_bias.float()
                    p = torch.softmax(logits, -1)
                for c, e, pp in zip(batch, imf.cpu().numpy(), p.cpu().numpy()):
                    done[c["key"]] = (e.astype(np.float16), pp.astype(np.float16))
            log(f"siglip: {len(queue)} crops in {time.time() - t0:.0f} s")
    keys = sorted(done)
    save_npz(
        jobs["out"],
        compressed=True,
        keys=np.array(keys),
        emb=np.array([done[k][0] for k in keys]),
        probs=np.array([done[k][1] for k in keys]),
        phrases=np.array(phrases),
    )


if __name__ == "__main__":
    main()
