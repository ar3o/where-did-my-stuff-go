# Where did my stuff go: results

Scene config1: a living room, a three-minute 4K HLG reference walk (276 frames used), 71 photos in 23 groups. Group 0
is 5 photos of the untouched room; groups 1 to 22 are 3 photos after each of 22 changes, and the room was reset in
between. The only scene input is a list of 61 names (10 furniture, 43 items, 8 static parts of the room).

There is no pixel-level or object-level ground truth for the 22 real changes. The numbers below are wrongly reported changes on the
untouched room, counts of what each method reported, and tests on synthetic edits held out from training. Where a
number comes from a different room or a different input, the row says so.

## Reconstruction

| | |
|:--|:--|
| Frames registered by SfM | 271 / 276, reprojection 0.62 px, 101,519 points |
| Photos registered in the same model | 70 / 71, median reprojection 0.82 px |
| Splat | 246 training frames, 2,526,856 Gaussians, 30k steps, 13.3 min, 7.4 GB peak |
| Held-out frames (25), with the nearest frame's colour correction | PSNR 26.3 dB, SSIM 0.878 |
| Same, without colour correction | PSNR 23.9 dB |
| Photo poses kept after render-and-compare refinement | 65 / 71 (the others keep the SfM pose) |
| Photo residual after refinement (median photo) | 10.9 / 255 on inlier pixels, inlier PSNR 23.5 dB |
| Known pixels per photo (seen by >= 3 reference cameras) | median 94 %, from 0 % (IMG_2606) to 99 % |
| Photos passing the depth-alignment check (MoGe-3 vs predicted, <= 3 % median error) | 61 / 71; all 3 of group 19 fail |

Pose source, measured on an earlier test table (tc2511, same splat recipe, 3k steps, 6 held-out keyframes): SfM poses
27.6 dB PSNR / SSIM 0.91, Pi3X poses 19.6 dB / 0.74.

## Inventory

| | |
|:--|:--|
| SAM 3 detections over 123 walk frames | 4,372 |
| Objects after Gaussian voting and fragment merging | 103 (68 items, 19 static, 16 furniture) |
| After the geometric merge | 96 (7 merges) |
| Names changed by adding SigLIP 2 to the SAM 3 votes | 23 |
| Names marked uncertain | 14 / 96 |

Earlier inventory versions: the first (per-frame winning phrase, duplicates merged by name) had 65 objects; a version
with hand-reviewed names and merges (61 objects) was dropped because the only allowed scene input is the name list.

## Change checker on held-out synthetic edits

864 training and 220 validation pairs; validation is the last 20 % of the walk. The splat was trained on all walk frames,
so validation renders are slightly easier than renders at new poses.

| | |
|:--|:--|
| Pixel F1, appeared / missing | 0.86 / 0.90 (precision 0.83 / 0.87, recall 0.90 / 0.92) |
| Edits found (an edit counts when 25 % of its region gets its class) | 0.93 of 191 |
| Remove / add / move 15-60 cm | 0.98 / 0.98 / 0.96 |
| Move 2-10 cm / 2-5 cm / rotate | 0.86 / 0.75 / 0.79 |
| Pixels flagged on unedited pairs (78) | 0.10 % |
| Training time | 2.8 min, 3.4 GB peak |

The same 48 validation pairs through the same object step (SAM automatic masks scored by each method):

| Method | Object recall | Wrongly detected, pairs without an edit | Wrongly detected, all pairs |
|:--|:-:|:-:|:-:|
| Hand-written depth and appearance rules (SAM mask matching) | 0.29 | 16 | 31 |
| Learned checker | 0.49 | 0 | 3 |

## Calibration

All fitted on the validation part of the walk and its synthetic-edit pairs; nothing is fitted on the change photos.

| Model | ECE (out of fold) | AUC | Positives / negatives |
|:--|:-:|:-:|:-:|
| P(change), missing | 0.014 | | 134 / 3347 |
| P(change), appeared | 0.016 | | 140 / 3347 |
| P(change), moved | 0.026 | | 58 / 584 |
| p_same (re-identification) | 0.027 | 0.92 | 997 / 4739 |
| p_added | 0.047 | 0.95 | 100 / 288 |

Same-place radius: 0.63 x max(object size, 0.15 m).

## Real photos

| Method | Objects wrongly reported as changed in the 5 untouched photos | Reported in groups 1-22 | Note |
|:--|:-:|:-:|:--|
| This pipeline (each group against the reference) | 0 | 59: 26 added, 18 removed, 15 moved | median 2.5 per group, 1 to 7 |
| Same pipeline carrying state between groups | 0 | 99 (35 removed, 22 moved, 22 returned, 20 added) | errors spread across groups |
| Official SceneDiff (DINOv2 variant, 32 reference frames) | 16 objects | 245 (166 removed, 43 added, 36 moved) | 49 s per pair without SAM, 9 GB |
| VLM judge, Qwen3-VL-4B, keyframe + photo | 21 items, 3 distinct | 179 items (132 after removing repeats) | output loops in 3 groups |
| Zero-shot rules on Pi3X features (per photo) | 0 changes, 16-27 of 27 objects "can't verify" | 2.7 per photo | 72 % of all answers "can't verify" |
| Depth and appearance change pixels fused into 3D blobs | 0 blobs | 238 blobs (108 "new") | no object identity |
| SAM automatic masks matched render vs photo | 0 | 109 (5 moves paired) | thin objects lost |
| Learned checker + SAM automatic masks (before the inventory) | 0 | 132 (13 moves paired) | many moves split into gone + new |

Confidences of the 59 reported changes: median 0.81, 19 at 0.9 or above, 11 below 0.6. The per-event table is
`changes/confidences.md`; the version for config1 is in [examples/config1_output/summary.json](../examples/config1_output/summary.json).

Baselines on the small synthetic test of an earlier Isaac Sim generator (1 test scene, 33 objects in no-change photos;
not comparable with the numbers above):

| Method | Accuracy | Macro-F1 | False change, no-change objects |
|:--|:-:|:-:|:-:|
| Zero-shot rules | 0.56 | 0.32 | 0 / 33 |
| VLM judge, full images | 0.29 | 0.20 | 5 / 33 |
| VLM judge, cropped | 0.29 | 0.24 | 8 / 33 |
| 2D SAM 3 mask differencing | 0.21 | 0.12 | 13 / 33 |

That generator (cousin scenes of boxes in Isaac Sim) and the set-transformer change model trained on it were dropped
when editing a splat of the real room gave photoreal training pairs.

## Manual evaluation on the real change sets

There is no independent ground truth for the real photos, so I scored reported events by hand. Sets 1 to 13 are scored
completely (106 events); 2 of the 7 events of set 14 were scored and are left out so that the three methods are compared
on the same sets, and sets 15 to 22 are not scored. The top events of each method (at most four per method per set, by the
method's own score) are mixed in one anonymous list per set, and events of the same type whose 3D boxes overlap (IoU of at
least 0.3, or one inside the other) are merged automatically with the same rule for every method. Each event gets one
verdict: correct, right object with the wrong change type, right object with the wrong name, duplicate of another event
in the list, nothing like that changed, or unsure.

I did not record how many real changes each set had, so recall is pooled: distinct real changes found by a method divided by
distinct real changes found by any of the three methods (62). Duplicate links join cards into one real change, and a
duplicate of another method's correct event counts as a correct detection for its own method. Type and name accuracy use
only the cards that were judged directly. Intervals are Wilson 95 %, narrower than they should be because events are
clustered in sets. The analysis is `whatchanged/evaluation/manual_scores.py`, the scores and the event key are in
`examples/manual_review/`, and the tables below come from
`python -m whatchanged.evaluation.manual_scores --scores examples/manual_review/scores.json --key examples/manual_review/key.json --groups 1-13`.

| Method | Events scored | Precision (95 % CI) | Recall, pooled (95 % CI) | Type accuracy (95 % CI) | Name accuracy (95 % CI) | Redundancy (95 % CI) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Full pipeline | 32 | 94 % (80-98) | 47 % (35-59) (29 of 62) | 80 % (61-91) | 100 % (87-100) | 0 % (0-11) |
| Official SceneDiff | 45 | 65 % (50-78) | 45 % (33-57) (28 of 62) | 65 % (45-81) | 91 % (73-98) | 4 % (1-15) |
| Rules (phrase pairing on the inventory) | 29 | 85 % (68-94) | 34 % (23-46) (21 of 62) | 50 % (27-73) | 100 % (78-100) | 7 % (2-22) |

| Method | Correct | Wrong type | Wrong name | Duplicate of another method's correct event | Duplicate within the method | Nothing like that changed | Unsure |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Full pipeline | 20 | 5 | 0 | 5 | 0 | 2 | 0 |
| Official SceneDiff | 13 | 8 | 2 | 5 | 2 | 10 | 5 |
| Rules (phrase pairing on the inventory) | 7 | 7 | 0 | 9 | 2 | 4 | 0 |

Verdict counts per method are in the second table. The rule-based version here is phrase pairing on the splat's inventory, not
the earlier rules on Pi3X features. The scorer is also the person who made the changes, so the scores are not independent,
and plain boxes are where they are least reliable.

## Differences between this repository and the research runs

The code here is a port of the research code. Each stage was rerun on the config1 inputs and compared with the research
outputs:

| Stage | Check | Result |
|:--|:--|:--|
| ingest | frame selection and pixels | 276 / 276 frames at the same times and roles; 0.26 / 255 mean pixel difference (one JPEG encode instead of two); 1920 px frames and photos identical |
| sfm | full rerun from the raw video and HEIC photos | 271 / 276 frames at 0.62 px, 101,340 points (research run: 271, 0.62 px, 101,519); 71 / 71 photos at 0.81 px median (research: 70 / 71 at 0.82 px; the photo pairs are chosen by matches instead of by Pi3X poses) |
| sfm | world frame on the research SfM model | up axis within 0.06 degrees; scale from MoGe-3 is 0.886 x the Pi3X scale used for the results |
| splat | localisation of 3 photos | same pose (<= 0.012 degrees), same residual, same kept / not kept |
| splat | support maps of 71 photos, known maps of 3 frames | at most 0.01 % of pixels differ |
| splat | training | identical first step; a 600-step run trains (19.2 dB); the full 30k-step run was not repeated |
| inventory | Gaussian voting on the research detections | 103 / 103 objects with the same names, 99.98 % of Gaussians with the same label |
| inventory | geometric merge and names | the same 7 merges, identical object ids and boxes, 96 / 96 names and confidences |
| checker | training pairs | 12 / 12 regenerated pairs bit-identical (with the research object groups) |
| checker | photo probabilities | within 0.002 (float16 storage) for all 71 photos |
| checker, changes | both calibrations | same samples, same ECE and AUC, weights within 1e-5 |
| changes | all 23 groups | 59 / 59 events identical in kind, object, name, confidence, centres, outlines and evidence |
| render | stills and clips of 3 groups | stills identical; clips have the same frames, 1.3 / 255 mean difference after video encoding |

Some behaviour differs from the research runs by design and was not checked end to end. The world frame and its metric scale come from the SfM model and MoGe-3 here, while the research runs took the scale and the table plane from Pi3X keyframe cameras, which kept Pi3X (non-commercial weights) in the pipeline for a single number. The inventory runs SAM 3 once with all phrases on every other walk frame, where the research inventory had 123 frames from several SAM 3 passes at scores of 0.4 and 0.35. The checker's training pairs edit the inventory's objects, because the object groups the research pairs edited came from an earlier perception step that no longer exists. The DINOv2 features of the walk frames are computed on the frames themselves, whereas the research code reused the features of each frame's first training pair, whose real image carries the colour augmentation, and re-identification on held-out walk frames lifts detections with MoGe depth aligned to the unedited render, where the research code used the first pair's edited and jittered render.
