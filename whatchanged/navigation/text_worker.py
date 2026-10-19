"""SigLIP 2 text embeddings for task parsing (google/siglip2-so400m-patch14-384, the same weights as the inventory
naming); vision env, CPU only.

    python -m whatchanged.navigation.text_worker jobs.json

jobs.json = {"siglip": <weights>, "texts": [...], "out": "emb.npz"} -> unit text embeddings with the prompt template
"a photo of a {text}.", as in whatchanged/common/siglip_worker.py. The text tower alone is fast on the CPU, so this
worker never competes for the shared GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TEXT_LEN = 64


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    from transformers import AutoModel, AutoProcessor

    t0 = time.time()
    torch.set_num_threads(8)
    model = AutoModel.from_pretrained(jobs["siglip"], torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(jobs["siglip"])
    texts = [f"a photo of a {t}." for t in jobs["texts"]]
    with torch.no_grad():
        ti = proc(text=texts, padding="max_length", max_length=TEXT_LEN, return_tensors="pt")
        tf = model.get_text_features(**ti)
        tf = tf if torch.is_tensor(tf) else tf.pooler_output
        tf = torch.nn.functional.normalize(tf.float(), dim=-1).numpy()
    np.savez(jobs["out"], emb=tf.astype(np.float32), texts=np.array(jobs["texts"]), seconds=time.time() - t0)
    print("done", len(texts), round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
