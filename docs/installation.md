# Where did my stuff go: installation

The pipeline uses three Python environments, because gsplat, SAM 3 and the vision models pin different library versions:

| Environment | Used for | Variable |
|:--|:--|:--|
| `splat` | the pipeline itself: ingest, SfM, splat, inventory, checker, changes, film (PyTorch, gsplat, pycolmap) | `SPLAT_PY` |
| `vision` | MoGe-3 depth, DINOv2 features, SigLIP 2 (PyTorch, Transformers, MoGe) | `VISION_PY` |
| `sam3` | SAM 3 detections from text prompts | `SAM3_PY`, `SAM3_REPO`, `SAM3_CHECKPOINT` |

The pipeline runs in `splat` and starts the vision and SAM 3 models as subprocesses with the interpreters named in
`configs/environment.yaml` (read from the variables above).

Developed on Ubuntu 22.04 with one RTX 4090 (24 GB), CUDA 12.9 and ffmpeg 6 with the zscale filter (needed for the HDR
tone mapping). The splat training peaks at about 7.4 GB of GPU memory; every other step stays under 5 GB.

## 1. This repository

```bash
git clone https://github.com/ar3o/where-did-my-stuff-go.git
cd where-did-my-stuff-go
export PYTHONPATH=$PWD
```

The code is used from the checkout, so the same tree serves all three environments. Data, runs and weights go to
`data/`, `runs/` and `weights/` in the checkout unless `WHATCHANGED_DATA`, `WHATCHANGED_RUNS` and `WHATCHANGED_WEIGHTS`
point elsewhere.

## 2. Splat environment

```bash
python3.11 -m venv envs/splat && source envs/splat/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/splat.txt
deactivate
export SPLAT_PY=$PWD/envs/splat/bin/python
```

gsplat compiles its CUDA kernels on first use; set `TORCH_EXTENSIONS_DIR` to keep them between runs. `ffmpeg -filters | grep
zscale` should list the filter.

## 3. Vision environment

```bash
python3.12 -m venv envs/vision && source envs/vision/bin/activate
pip install -r requirements/vision.txt
git clone https://github.com/microsoft/MoGe.git third_party/MoGe && pip install -e third_party/MoGe
git clone https://github.com/facebookresearch/dinov2.git weights/torch_hub/facebookresearch_dinov2_main
deactivate
export VISION_PY=$PWD/envs/vision/bin/python
```

Weights download from Hugging Face on first use: MoGe-3 (`Ruicheng/moge-3-vitl`), SigLIP 2
(`google/siglip2-so400m-patch14-384`, or set `SIGLIP_DIR` to a local copy) and the DINOv2 ViT-L/14 checkpoint with
registers (`dinov2_vitl14_reg4_pretrain.pth`, fetched by torch hub into `$TORCH_HOME/hub/checkpoints`).
`DINO_HUB` points to the dinov2 checkout. Offline, set `MOGE_CHECKPOINT` to a local MoGe-3 `model.pt` and `SIGLIP_DIR`
to a local SigLIP 2 folder. `VISION_PYTHONPATH` adds entries to the vision workers' path, for example a MoGe checkout
that is not pip-installed.

## 4. SAM 3 environment

Follow the setup in [facebookresearch/sam3](https://github.com/facebookresearch/sam3), then:

```bash
export SAM3_PY=/path/to/sam3/env/bin/python
export SAM3_REPO=/path/to/sam3                     # the checkout, added to PYTHONPATH of the worker
export SAM3_CHECKPOINT=$PWD/weights/sam3/sam3.pt   # the official facebook/sam3 release
```

The worker keeps the weights in fp32 under bf16 autocast: a bf16 copy of the vision trunk returned no detections. The
text encoder runs on the CPU.

## 5. Check

```bash
$VISION_PY -m pytest -q tests        # fast, no GPU
$SPLAT_PY -m whatchanged.run --help
```

## Docker

`docker/Dockerfile.splat` and `docker/Dockerfile.vision` describe the splat and vision environments as containers:

```bash
docker build -f docker/Dockerfile.splat -t whatchanged-splat .
docker build -f docker/Dockerfile.vision -t whatchanged-vision .
docker run --gpus all -it --rm -v $PWD/data:/workspace/whatchanged/data -v $PWD/runs:/workspace/whatchanged/runs whatchanged-splat
```

Set `VISION_PY` and `SAM3_PY` in the splat container to interpreters that can run the workers, or run those stages in
the vision container. The images were written to mirror the environments above and have not been built on every GPU.
