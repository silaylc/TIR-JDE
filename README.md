# TIR-JDE

**Degradation-Adaptive Small Target Detection and Joint Identity Embedding for Multi-Object Tracking in Thermal Infrared UAV Imagery**

Multi-UAV tracking in thermal infrared imagery. A single lightweight network
produces an identity embedding alongside each box; an image-degradation
estimate from the same network both conditions the detection features and
sets the appearance-versus-motion weight **per detection** during
association.

Senior project (ISE 494), Atılım University, Information Systems
Engineering, in cooperative education with TÜBİTAK SAGE.

> **Status: skeleton.** No working code yet. This repository currently fixes
> the structure, the method provenance and the evaluation protocol.

---

## Problem

In the Track 3 data the average target is 10.56 × 9.06 pixels, with boxes at
the low end falling below 2 pixels. At that size a target has no texture.
Its visibility shifts from frame to frame with sensor noise, automatic gain
control, thermal drift, blur and atmospheric attenuation. UAVs within a
swarm look nearly identical to one another.

Existing Track 3 solutions respond at two extremes:

- **Abandon appearance entirely.** [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html)
  uses motion only (IoU + L2 fusion) and has no appearance embedding.
- **Use appearance with a fixed weight.** [Strong Baseline](https://arxiv.org/abs/2503.17237)
  and [PPTracker](https://mlanthology.org/cvprw/2025/qin2025cvprw-pptracker/)
  use a separately trained ReID passed through BoT-SORT's fixed thresholds.

[Context-Aware Identity Prediction](https://doi.org/10.3390/rs18132084) sits
between the two: it learns the appearance embedding end-to-end and weights it
by estimated reliability — but with **a single weight per frame**, computed
from a scene-wide descriptor.

TIR-JDE's claim: appearance reliability is a property of the **detection**,
not of the frame. Within the same frame, on the same sensor, at the same
instant, one UAV can be clearly visible while another blends into the
background — so a single frame-level score represents neither of them
correctly.

## Approach

1. **Lightweight detector.** Ultralytics YOLO (Nano/Small), with a P2
   (stride-4) head and 1280 px input for tiny targets.
2. **Degradation conditioning.** A code derived from the degradation estimate
   modulates detector features via [FiLM](https://arxiv.org/abs/1709.07871)
   (`γ(z) ⊙ F + β(z)`).
3. **Joint identity head.** A short (64–128 dimensional) embedding trained
   together with the detector — no separate ReID network.
4. **Degradation-aware association.** The same degradation estimate yields a
   per-detection reliability weight `r`:
   `cost = r × appearance distance + (1 − r) × position distance`.

The fourth is the actual contribution. Every degradation-aware method we
reviewed ([FiLM](https://arxiv.org/abs/1709.07871),
[DASR](https://arxiv.org/abs/2104.00416),
[AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html),
[DTRDNet](https://doi.org/10.3390/s24196330),
[RDMNet](https://github.com/xfwang23/RDMNet),
[DAISOD](https://arxiv.org/abs/2608.09311)) stops at restoration or
detection; none carries the degradation estimate into association.

---

## Method provenance

Where each component comes from, what we changed, and where it sits in the
code. Detailed notes on each paper are in Confluence.

| Component | Source | Our change | Code |
|---|---|---|---|
| Detector backbone | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics); [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (architectural template) | P2 head, 1280 px, added FiLM module | `detector/` |
| Appearance head design | [JDE](https://arxiv.org/abs/1909.12605) (joint-head idea), [FairMOT](https://arxiv.org/abs/2004.01888) (anchor-free, high resolution, short vector), [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (2×3×3 + 1×1 conv.) | Head at the P2 level; automatic three-task loss balancing (detection + identity + degradation) | `detector/heads/` |
| Identity supervision | [YOLO11-JDE](https://arxiv.org/abs/2501.13710) (label-free triplets via Mosaic), [FairMOT](https://arxiv.org/abs/2004.01888) (label-free pre-training) | Positive pairs from different frames using real IDs; pre-training on Track 1/2 SOT data treating each sequence as one identity | `detector/losses/` |
| Multi-point readout | [Motion-Guided Multi-Offset ReID Readout](https://doi.org/10.3390/rs18183238) | A single center point is fragile in a 6-pixel box; sample from several points instead | `detector/heads/` |
| Degradation conditioning | [FiLM](https://arxiv.org/abs/1709.07871) (mechanism), [DASR](https://arxiv.org/abs/2104.00416) / [AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html) (contrastive degradation encoder), [DAISOD](https://arxiv.org/abs/2608.09311) (explicit degradation estimation in IR) | TIR-specific synthetic degradation (column noise, AGC contrast compression, blur, attenuation); the synthetic parameters are free labels | `degradation/` |
| Tracking framework | [ByteTrack](https://arxiv.org/abs/2110.06864) (two-round matching), [BoT-SORT](https://arxiv.org/abs/2206.14651) (w/h Kalman, two-gate fusion) | Thresholds retuned for UAVs; TFPS and one-frame predictive coasting from [Edge-Aware](https://arxiv.org/abs/2607.12544) | `trackers/` |
| Association cost | [BoT-SORT](https://arxiv.org/abs/2206.14651)'s fixed gates, [ByteTrack](https://arxiv.org/abs/2110.06864)'s score rule, [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html)'s L2 + IoU fusion | A continuous, per-detection `r` instead of fixed gates; center distance or [NWD](https://arxiv.org/abs/2110.13389) trialled in place of IoU for tiny boxes | `trackers/*/matching.py` |
| Appearance memory | [JDE](https://arxiv.org/abs/1909.12605) (EMA update) | Update rate is not fixed but tied to `r` — appearance from a degraded frame changes the memory less | `trackers/*/track.py` |
| Camera motion (GMC) | [BoT-SORT](https://arxiv.org/abs/2206.14651) (pyramidal Lucas-Kanade + RANSAC) | Outside the core contribution; to be measured and decided by an on/off ablation (in some sequences the camera is completely static) | `trackers/*/gmc.py` |
| Main baseline | [Strong Baseline](https://arxiv.org/abs/2503.17237) ([code](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID)) | Retrained on the BSB training split; joint embedding compared against a separately trained ReID | `baselines/strong_baseline/` |
| Second baseline | [Dist-Tracker](https://openaccess.thecvf.com/content/CVPR2025W/Anti-UAV/html/Wang_Dist-Tracker_A_Small_Object-aware_Detector_and_Tracker_for_UAV_Tracking_CVPRW_2025_paper.html) (Track 3 winner) | Reports MOTA only; we will be the first to report its IDF1/IDSW/FPS | `baselines/dist_tracker/` |
| Comparison target | [Context-Aware Identity Prediction](https://doi.org/10.3390/rs18132084) (frame-level reliability gate) | Reimplement the frame-level gate inside our own framework and ablate it against per-detection `r` | `eval/ablations/` |
| Metrics | [HOTA](https://doi.org/10.1007/s11263-020-01375-2), IDF1, CLEAR MOT, [TrackEval](https://github.com/JonathonLuiten/TrackEval) | Alongside the averages, IDSW / AssA reported stratified by degradation severity | `eval/` |

### Considered and deliberately not adopted

- [MOTIP](https://arxiv.org/abs/2403.16848) / DETR-based end-to-end identity
  prediction (the backbone of Context-Aware) — does not fit the edge compute
  budget; conflicts with the real-time constraint.
- **AKKF** ([Edge-Aware](https://arxiv.org/abs/2607.12544)'s adaptive Kalman
  filter) — the reported gain is in the third decimal place
  (HOTA 0.82612 → 0.82661) while FPS drops 132 → 95. The idea (dynamic
  measurement noise) is valuable; this implementation is expensive.
- [Offline track relinking](https://arxiv.org/abs/2606.01694) — breaks the
  online tracking constraint.
- SOT template matching ([SiamSTA](https://openaccess.thecvf.com/content/ICCV2021W/AntiUAV/html/Huang_SiamSTA_Spatio-Temporal_Attention_Based_Siamese_Tracker_for_Tracking_UAVs_ICCVW_2021_paper.html),
  SiamDT) — works for a single target, does not scale to 30.

---

## Data

[Track 3](https://anti-uav.github.io/) of the 4th Anti-UAV Challenge
(CVPR 2025). 640×512 thermal, 200 training + 100 test sequences. The official
test labels are held on the
[challenge server](https://codalab.lisn.upsaclay.fr/competitions/21806).

Label format (MOTChallenge, 9 columns, frames start at 1):

```
frame, id, x_left, y_top, w, h, conf, class, visibility
```

`x,y` is the top-left corner (verified empirically). Columns 7–9 are constant
in this dataset (`1,1,1.0`) and carry no information.

The videos are MPEG-4 compressed. Part of the degradation we measure comes
from the codec rather than the sensor; the thesis therefore uses the term
"image degradation" (a combination of sensor noise, thermal drift, blur and
compression artifacts) rather than "sensor degradation".

**Split: BSB (Beyond Strong Baseline).** Strong Baseline split at the frame
level — frames from the same video fell into both training and validation.
The author acknowledges that this led to overfitting. We use
[Edge-Aware](https://arxiv.org/abs/2607.12544)'s sequence-wise 102/98
re-split instead.

The BSB test labels are hidden (only first-frame boxes are given):

- Local ablations → a fixed validation group carved out of the 102 training
  sequences
- Final numbers → the BSB leaderboard
- Split lists live under `data/splits/`; the data files are not in Git

**Known annotation errors** (as reported by Strong Baseline):
MultiUAV-230 (incorrect), MultiUAV-256 (redundant), MultiUAV-294 (missing),
MultiUAV-068 (test, one poor-quality frame).

## Evaluation

The primary metric is [HOTA](https://doi.org/10.1007/s11263-020-01375-2)
(via [TrackEval](https://github.com/JonathonLuiten/TrackEval)), reported
alongside IDF1 and IDSW.

MOTA is not primary: in Strong Baseline's ablation the ReID module moved MOTA
by only ~0.01, even though its contribution is to identity preservation. A
metric dominated by detection errors cannot measure a thesis whose
contribution is on the association side.

Every result is additionally reported:

- **On a single, declared split.** Published Track 3 numbers mix the official
  test set, the authors' own validation splits and BSB; they cannot be ranked
  against one another.
- **With edge efficiency.** Parameters, GFLOPs, model size and end-to-end FPS
  (detector + embedding + association), at the same resolution used for
  accuracy, on stated hardware.
- **Stratified by degradation severity.** "Average HOTA +2" is replaced by
  "+5 on degraded sequences, +0.3 on clean ones".

**Pre-registered possibility:** the finding "appearance helps only where
degradation is low" is itself a valid result. Refuting the hypothesis does
not refute the thesis.

## Setup

```bash
conda env create -f environment.yml
conda activate tir-jde
```

PyTorch and Ultralytics will be added at the detector stage.

## Repository structure

```
baselines/      comparison methods re-run by us
configs/        experiment configurations
data/splits/    BSB sequence lists (data files are not in Git)
detector/       detector, appearance head, losses
degradation/    degradation estimation and the FiLM module
eval/           TrackEval wrapper, ablations, stratified reporting
notebooks/      EDA and analysis
tools/          data preparation, format conversion, visualization
trackers/       tracking framework and association
```
