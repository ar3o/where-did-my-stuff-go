"""SigLIP 2 zero-shot naming server (vision env, about 2.2 GB of GPU memory in fp16), driven by the robot run over
stdin / stdout to name newly observed obstacle blobs. The splat environment that runs the robot has no transformers,
and loading the model once per crop would be far too slow, so the model stays loaded for the whole run.

    python -m whatchanged.navigation.siglip_server vocab.json

vocab.json = {"siglip": <weights>, "phrases": [...]}; prompt "a photo of a {phrase}.", as in
whatchanged/common/siglip_worker.py. Protocol: one JSON line in, {"crops": ["a.jpg", ...]} or {"quit": true}, one JSON
line out, {"probs": [[softmax over phrases] per crop]}; the first output line is {"ready": true}. The server does not
take the pipeline's GPU lock: the robot run that starts it already holds it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

TEXT_LEN = 64
BATCH = 16


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    vocab = job["phrases"]
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(job["siglip"], dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(job["siglip"])
    with torch.no_grad():
        texts = [f"a photo of a {p}." for p in vocab]
        ti = proc(text=texts, padding="max_length", max_length=TEXT_LEN, return_tensors="pt").to("cuda")
        tf = model.get_text_features(**ti)
        tf = tf if torch.is_tensor(tf) else tf.pooler_output
        tf = torch.nn.functional.normalize(tf.float(), dim=-1)
    peak = int(torch.cuda.max_memory_reserved() / 2**20)
    print(json.dumps({"ready": True, "n": len(vocab), "torch_peak_mib": peak}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        ims = [Image.open(p).convert("RGB") for p in req["crops"]]
        out = []
        for i in range(0, len(ims), BATCH):
            with torch.no_grad():
                ii = proc(images=ims[i : i + BATCH], return_tensors="pt").to("cuda")
                f = model.get_image_features(pixel_values=ii["pixel_values"].half())
                f = f if torch.is_tensor(f) else f.pooler_output
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                lg = f @ tf.T * model.logit_scale.exp().float() + model.logit_bias.float()
                out += torch.softmax(lg, -1).cpu().numpy().round(5).tolist()
        print(json.dumps({"probs": out}), flush=True)


if __name__ == "__main__":
    main()
