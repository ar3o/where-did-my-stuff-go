# Where did my stuff go: attribution: what is ours and what isn't

## Third-party components

| Component | Role here | Licence |
|---|---|---|
| gsplat 1.5.3 (nerfstudio) | Gaussian splat training and rendering | Apache-2.0 |
| COLMAP / pycolmap | Structure from motion | BSD-3-Clause |
| SAM 3 (Meta), official `facebook/sam3` | Detections from text prompts | SAM License; access is granted on the model page |
| DINOv2 ViT-L/14 with registers (Meta) | Frozen features for the checker and fingerprints | Apache-2.0 |
| SigLIP 2 so400m (Google) | Zero-shot names, re-identification embeddings | Apache-2.0 |
| MoGe-3 (Microsoft) | Metric monocular depth, metric scale | see the upstream repository |
| PyTorch, OpenCV, SciPy, ffmpeg | everything else | BSD, Apache-2.0 / BSD, BSD, LGPL / GPL depending on the build |

Not used by the code in this repository, but named in the README or `docs/results.md`:

| Component | Where it appears | Licence note |
|---|---|---|
| Pi3X | Gave the metric scale and table plane in the published config1 runs; replaced here by MoGe-3 and an SfM plane fit | Weights are CC BY-NC, so non-commercial use only |
| Qwen3-VL-4B | The VLM judge baseline | see the upstream repository |
| SceneDiff, 3DGS-CD | Baseline and related work | see the upstream repositories |
| SAM v1 ViT-H | SAM automatic-mask matching baseline | Apache-2.0 |

The SAM 3 and MoGe weights are not redistributed here. Check each model's licence before using it commercially.

## Ours

The checker is ours: the generator that edits the reference splat and renders it at the pose of a real walk frame, the labels that fall out of those edits, the decoder that reads the pair, and the validation by walk time. The same goes for predicting a photo's view by render-and-compare against the splat and for the per-pixel count of walk cameras that saw each point, which marks what the walk never filmed. The inventory is ours as well, meaning the voting of SAM 3 masks onto Gaussians with region growing for thin objects, the merge by geometry before naming, and names from SAM 3 votes multiplied with SigLIP 2. For the decisions per set of photos, the re-identification with calibrated probabilities, the counting against a render of the reference, the one-to-one assignment, the rule for when the answer is "can't verify" and the text made from 3D facts are ours too, and so is the film with its hologram, glide and rise animations and its label placement.

## Hand-chosen settings that remain

They are listed with their sources in [method.md](method.md), and the README has a section on what I fixed by hand and what was tuned on what. The name list is the only scene-specific input.
