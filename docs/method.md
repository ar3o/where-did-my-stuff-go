# Where did my stuff go: settings reference

The README explains each stage and why it is built that way. This page lists the numbers behind it and where each one
comes from. Every default is also in `configs/<stage>.yaml`. "Held-out" means the last 20 % of the walk by time, never a
query photo.

## Ingest

| Setting | Value | Source |
|:--|:--|:--|
| Decode | 10 fps, long side 960 px (1920 px copies for SAM 3 tiles and SigLIP 2 crops) | working size of the models |
| HDR | linearise at 100 nits, bt709 primaries, Hable tone map, bt709 limited range | grey frames without it |
| Frames | sharpest of each of 64 equal time bins, plus the sharpest of 220 bins at least 0.15 s from a keyframe | SfM needs about 250 frames |
| Acceptable frame | Laplacian variance >= 0.35 x median, grey std >= 12 | drops blur and a hand over the lens |
| Held out | every 8th extra frame | splat test set |

## SfM and world frame

| Setting | Value | Source |
|:--|:--|:--|
| SIFT | 8192 features, one shared PINHOLE camera for the walk, one per photo size | COLMAP |
| Photo focal prior | 24 mm equivalent | iPhone main camera; bundle adjustment refines it |
| Pairs | 12 neighbours, all keyframe pairs, 6 nearest keyframes; photos with all keyframes and 5 neighbours, then the frames around each photo's 6 best keyframes | chosen so the loop closes without all-pairs matching |
| Metric scale | median over keyframes of MoGe-3 depth / SfM depth at triangulated points | MoGe-3 is metric |
| Floor | greedy RANSAC planes (1 cm), >= 80 % of cameras above, normal agrees with image-up, then a refit within 3 cm (4 mm) | physics-free edits need a level floor |

## Splat

| Setting | Value | Source |
|:--|:--|:--|
| Strategy | gsplat DefaultStrategy, antialiased, packed; 30k steps; growth threshold 2e-4, x1.5 above 2.5M Gaussians | 3DGS defaults |
| Loss | 0.8 L1 + 0.2 D-SSIM + 0.01 x colour-affine L2 | 3DGS defaults |
| Colour | per-frame 3x4 affine, identity start | phone auto-exposure drift |
| Localisation | 6-DoF pose increment, 60 steps at 240x180 (blur 2) then 40 at 480x360 (blur 1), Geman-McClure on colour plus 0.5 x gradients, 3 sigma outlier mask | photometric alignment that ignores outliers |
| Pose kept | only if the median inlier residual improves | |
| Known pixel | seen by >= 3 reference cameras on their own surface (2 cm + 3 %), ray angle <= 60 degrees, 0.15-6 m | opacity alone does not flag unseen regions |
| Photo usable | SfM-registered and aligned MoGe-3 depth within 3 % (median) of the predicted depth | |

## Inventory

| Setting | Value | Source |
|:--|:--|:--|
| SAM 3 | min score 0.35, top 6 per phrase, full 960 px frame plus 2 x 2 tiles (55 % windows) of the 1920 px frame, every 2nd frame | small items need pixels |
| Voting | up to 6000 Gaussians per detection; 10 Gaussians minimum for elongated masks (axis ratio >= 4), 30 otherwise; link at cosine >= 0.2 | |
| Assignment | >= 50 % of a Gaussian's votes and >= 0.02; 4 cm core clean-up | |
| Region growing | 1.5 cm, colour 0.12, up to 30 rounds | thin objects |
| Fragment merge | touch < 1.5 cm, colour 0.12, DINOv2 cosine >= 0.75, merged box <= 0.5 m, nothing above 0.7 m merges | a looser rule merged a jug with a bowl |
| Names | z-normalised SAM 3 scores, Ochiai overlap, SigLIP 2 on 4 best views, "(uncertain)" below a 25 % margin | |

## Checker

| Setting | Value | Source |
|:--|:--|:--|
| Pairs | 4 per walk frame; about 35 % unedited; remove / add / move / rotate; moves 2-10 or 15-60 cm; rotations 5-90 degrees; decoy fills 30 % / 15 % | covers the changes in the photos |
| Jitter | 0-2 cm, 0-0.5 degrees | tiny misalignment is not a change |
| Features | DINOv2 ViT-L/14-reg4 at 924 px, PCA to 128 dims fitted on 150 walk frames | 2-5 cm moves need more than 46 patches |
| Model | 192-wide feature branch, 32-wide low-level branch, 3 classes at 1/4 resolution | |
| Training | AdamW 1e-3, one cycle, batch 16, 40 epochs, class weights sqrt(f0 / fc) capped at 20, plus 0.5 x focal | |
| Calibration | Platt scaling on [mean probability, log size], two folds by frame | |

## Changes

| Setting | Value | Source |
|:--|:--|:--|
| SAM 3 | min score 0.2, top 8 per phrase, furniture and item phrases only | |
| Object state | missing if calibrated P >= 0.5, else covered if P(appeared) >= 0.5, else present; clear view = >= 50 % of the object visible | |
| Same place | 0.63 x max(size, 0.15 m) | 95th percentile on held-out frames |
| Re-identification | logistic on SigLIP 2, DINOv2, size, colour and SAM 3 z; accepted at 0.5 | held-out frames |
| Added thing | calibrated p_added >= 0.5 and >= 2 photos (or one detection away from the border that no other photo sees); size <= 0.8 m | |
| Combination | noisy-OR over photos | extrapolation, see limitations |

## Film

| Setting | Value |
|:--|:--|
| Timing | 2.5 s before, 1.6 s flight, 2.4 s per change, 0.5 s hold, 1 s cross-fade, 2.5 s after |
| Vantage | a walk camera 0.6-2.5 m from the change, about 1.3 m preferred, target in the central 70 % |
| Changes shown | at most 5 per group, by confidence |
