# Small-object detection on VisDrone: scale-aware evaluation, tiled inference and black-box explanations

**Maryam Tavasoli** · Technical report, November 2026 · Code: https://github.com/maryamtavasoli98/small-object-detection-visdrone

## Abstract

Objects in aerial images are often only a few pixels wide, and the usual practice of resizing a full frame to 640 px before detection removes most of the evidence a detector needs. We study this failure mode on VisDrone2019-DET with a small one-stage detector (YOLO11n) and three questions: where the detector fails as a function of object size, how much tiled (sliced) inference recovers and at what cost, and whether black-box saliency (D-RISE) explains detections of tiny objects in a way that is faithful to the model. With the standard 640 px resize the detector reaches AP50 0.274 overall but only 0.085 on objects smaller than 16 px. Tiled inference raises tiny-object AP50 to 0.199 and overall AP50 to 0.375 (with the full frame added), at about five times the inference time, while simply enlarging the test-time input to 1024 px gives AP50 0.365 at no extra cost. D-RISE maps are faithful for all object sizes (deletion AUC 0.08 vs 0.78 for a random pixel order on tiny objects), provided the deletion protocol uses log-spaced steps; with the usual linear steps every map of a tiny object reaches the same floor. All code, metrics and figures are reproducible on a laptop GPU.

## 1. Introduction

Aerial imagery has become a standard setting for traffic analysis, search and rescue and infrastructure inspection. Its defining difficulty is scale: a pedestrian photographed from about 50 m above occupies a 10 to 20 px patch in a 1360 × 765 frame. Detectors are trained and evaluated at inputs of 640 px on a side, so that patch shrinks to under 10 px, below the stride of the finest feature map of most architectures. The problem is well documented in the small-object detection literature [8], and is the main motivation behind drone-specific detectors [7] and slicing-based inference [2].

Average precision summarised over all objects hides the effect. We therefore report AP per object-size bin, with a finer split at the low end than COCO's "small" (below 32 px): tiny (below 16 px), small (16 to 32 px), medium (32 to 96 px) and large. We then measure how much a simple, training-free remedy, running the same weights on overlapping native-resolution tiles, recovers in each bin, and compare it with the other cheap remedy of enlarging the input.

The third question is motivated by applications such as rescue, where an operator needs to know *why* the system flags a 15-px blob as a person. Gradient-based maps such as Grad-CAM [9] need access to the network and were designed for classifiers. D-RISE [3] treats the detector as a black box: it perturbs the input with random masks and weights each mask by how well the target detection survives. We re-implement it for an Ultralytics detector and ask whether the resulting maps are faithful, using deletion and insertion curves [4] against a random-order baseline, stratified by object size. To our knowledge the size dependence of saliency faithfulness has not been examined for aerial imagery.

Contributions: (i) a size-binned, ignored-region-aware COCO evaluation of a YOLO11n baseline on VisDrone; (ii) a compact, tested implementation of tiled inference and a measurement of its gain per size bin and its time cost; (iii) a D-RISE implementation for Ultralytics detectors with a faithfulness study across object sizes. Everything runs on an Apple M3 Pro laptop in a few hours.

## 2. Related work

**Small objects from drones.** VisDrone [1] is the reference benchmark; its objects are small, dense and frequently occluded, and the official protocol ignores regions marked as such by the annotators. TPH-YOLOv5 [7] adds a prediction head for tiny objects and transformer blocks to a YOLOv5 backbone specifically for this benchmark. Cheng et al. [8] survey the field and show that small-object AP lags large-object AP by tens of points across detector families.

**Slicing.** SAHI [2] proposed slicing-aided hyper inference: run the detector on overlapping crops at native resolution and merge the results, optionally fine-tuning on slices as well. Our tiling module follows the inference part of that recipe and adds a per-size evaluation of its effect.

**Explaining detectors.** RISE [4] estimates the importance of each pixel for a classifier by Monte-Carlo sampling of random masks. D-RISE [3] extends it to detection by scoring each masked image with the similarity between the target detection and the best matching detection on the masked image. Both papers evaluate explanations with deletion and insertion curves, which measure how quickly the model's output decays or recovers when pixels are removed or restored in saliency order; sanity checks [10] warn that visually plausible maps can be insensitive to the model. We use the deletion/insertion protocol with a random-order baseline and report the usual AUCs together with two simple concentration statistics.

## 3. Data and protocol

VisDrone2019-DET [1] provides 6,471 training and 548 validation images with ten object classes. We convert the annotations to YOLO format, keep the ten classes, drop the 32 boxes labelled "others", and store the ignored regions separately. Table 1 summarises the splits.

| split | images | boxes | objects / image (median, max) | tiny <16 px | small 16-32 | medium 32-96 | large >96 |
|---|---|---|---|---|---|---|---|
| train | 6,471 | 343,204 | 42, 902 | 26.0 % | 34.5 % | 34.0 % | 5.5 % |
| val | 548 | 38,759 | 65, 317 | 30.8 % | 37.7 % | 28.7 % | 2.8 % |

Table 1. Dataset statistics after conversion. Size = √(w·h) in original-image pixels.

**Evaluation.** Predictions with confidence ≥ 0.001 (up to 1,000 per image) whose centre falls inside an ignored region are discarded; the rest are scored with pycocotools against the ten-class ground truth. We report COCO AP (IoU 0.50:0.95), AP50, average recall (AR) and AP50/AR per size bin, where the bin is defined by the ground-truth box. A detection of a tiny object therefore counts towards the "tiny" bin even if the predicted box is larger.

## 4. Method

**Baseline.** YOLO11n (2.6 M parameters) initialised from COCO weights and fine-tuned with the Ultralytics trainer [6] at 640 px for 30 epochs, batch 16, default augmentation, seed 0, on an Apple M3 Pro GPU (MPS backend).

**Tiled inference.** For an image of width W and height H we generate 640 × 640 tiles with stride 512 px (20 % overlap); the last tile on each axis is placed flush with the image border, so every pixel is covered and tiles never leave the image. The detector runs on batches of eight tiles at native resolution. Boxes are shifted back to image coordinates and merged with greedy per-class non-maximum suppression at IoU 0.5; duplicates arise only where tiles overlap. A variant adds the resized full frame as one more tile, which gives large objects a view in which they are not cut. The module is about 120 lines of NumPy and is tested against a brute-force NMS and on tile-coverage invariants.

**D-RISE for Ultralytics detectors.** For a target detection (box b, class c) on a tile x, we sample N soft masks M_i as in RISE: a 16 × 16 Bernoulli(0.5) grid, upsampled bilinearly and shifted by a random offset of up to one cell. The detector is run on x · M_i (element-wise) and scored with s_i = max over same-class detections of IoU(b, b') · conf(b'), an indicator version of the class term of the original paper, since the Ultralytics API exposes one class and one confidence per box. The saliency map is the mean-centred estimator S = (1/(N p)) Σ s_i (M_i − p), clipped at zero and normalised to a maximum of 1. Without the centring term, S carries a constant offset p · Σ s_i that makes the map look nearly uniform and makes mass statistics meaningless; the offset does not change the pixel ranking. We use N = 1,000 masks in batches of 16.

**Faithfulness.** Deletion replaces pixels of x, in decreasing order of S, by a heavily blurred copy of x (Gaussian, 51 px) and records the target score s at each step, normalised by the score on the unperturbed tile; insertion starts from the blurred copy and restores pixels in the same order. The removed fractions are log-spaced in 24 steps from 0.0001 to 1: a tiny object covers about 0.05 % of a 640 px tile, so linear 5 % steps would remove it completely in the first step and every map would reach the same floor (our first run showed exactly this). We report the area under both curves against the log-fraction rescaled to the unit interval and repeat the procedure with a random pixel order. Two descriptive statistics complete the picture: the share of saliency mass inside the target box, and the fraction of the tile that holds 90 % of the mass.

**Selection of targets.** Val images are visited in a seeded random order; on each native-resolution tile the detections with confidence ≥ 0.25 are matched one-to-one to ground-truth boxes fully inside the tile (same class, IoU ≥ 0.5). True positives are collected until each size bin holds eight, with at most two per image.

## 5. Experiments

### 5.1 Training

| run | init | input | epochs | batch | time | val mAP50 | val mAP50-95 |
|---|---|---|---|---|---|---|---|
| `y11n_640` | yolo11n.pt | 640 px | 30 | 16 | 2.4 h | 0.299 | 0.165 |

Table 2. Training runs (metrics from the Ultralytics validator, its own protocol).

### 5.2 Detection by object size

| inference | AP | AP50 | tiny | small | med. | large | AR | s/img |
|---|---|---|---|---|---|---|---|---|
| resize 640 | 0.153 | 0.274 | 0.085 | 0.226 | 0.419 | 0.570 | 0.232 | 0.012 |
| resize 1024 | 0.210 | 0.365 | 0.168 | 0.338 | 0.494 | 0.543 | 0.326 | 0.011 |
| 640 tiles | 0.206 | 0.366 | 0.199 | 0.354 | 0.474 | 0.423 | 0.346 | 0.044 |
| tiles + full | 0.212 | 0.375 | 0.198 | 0.356 | 0.494 | 0.553 | 0.359 | 0.062 |

Table 3. VisDrone val, YOLO11n trained at 640 px. AP = AP@[.50:.95]; tiny/small/med./large = AP50 per size bin (<16, 16-32, 32-96, >96 px); AR = average recall; seconds per image on the M3 Pro with batched tiles.

With the plain 640 px resize, AP50 grows monotonically with object size, from 0.085 on tiny objects to 0.570 on large ones, and recall follows (AR 0.053 tiny vs 0.577 large). Tiled inference with 640 px native-resolution tiles (5.2 tiles per image on average) more than doubles tiny-object AP50 (0.199) and raises small-object AP50 from 0.226 to 0.354. Tiling alone, however, lowers large-object AP50 from 0.570 to 0.423, because large vehicles near tile borders are split and per-class NMS cannot merge the parts. Adding the resized full frame as an extra tile repairs this (0.553) and gives the best configuration overall: AP50 0.375 and AP 0.212, against 0.274 and 0.153 for the plain baseline.

The cheapest alternative is surprisingly competitive. Feeding the same weights a 1024 px frame at test time, without any retraining, reaches AP50 0.365 at the same per-image time as the plain baseline (0.011 vs 0.012 s), whereas tiling costs 0.044 s and tiles plus full frame 0.062 s. Tiling keeps a clear advantage only in the tiny bin (0.199 vs 0.168 AP50) and a smaller one in the small bin (0.354 vs 0.338).

| class | median side (px) | AP50 plain 640 | AP50 tiled 640 | change |
|---|---|---|---|---|
| pedestrian | 17 | 0.324 | 0.501 | +0.177 |
| people | 17 | 0.200 | 0.315 | +0.115 |
| bicycle | 21 | 0.060 | 0.124 | +0.064 |
| car | 32 | 0.722 | 0.794 | +0.073 |
| van | 34 | 0.242 | 0.382 | +0.140 |
| truck | 39 | 0.261 | 0.303 | +0.042 |
| tricycle | 28 | 0.178 | 0.241 | +0.063 |
| awning-tricycle | 29 | 0.052 | 0.080 | +0.028 |
| bus | 35 | 0.374 | 0.485 | +0.111 |
| motor | 20 | 0.324 | 0.438 | +0.114 |

Table 4. Per-class AP50, plain 640 vs 640 tiles, same weights.

All ten classes improve with tiling. The largest gains are for the classes with the smallest objects: pedestrian (+0.177 AP50, median side 17 px), van (+0.140), people (+0.115) and motor (+0.114). Car, already the easiest class, gains +0.073. Bicycle and awning-tricycle remain weak even with tiling (0.124 and 0.080), so resolution alone does not solve them.

### 5.3 Faithfulness of D-RISE explanations

| size | del. D-RISE | del. random | ins. D-RISE | ins. random | mass in box | box area | 90 % area |
|---|---|---|---|---|---|---|---|
| tiny | 0.08±0.07 | 0.78±0.09 | 0.41±0.16 | 0.05±0.03 | 1.3 % | 0.05 % | 30 % |
| small | 0.22±0.14 | 0.79±0.09 | 0.46±0.09 | 0.04±0.02 | 3.7 % | 0.22 % | 33 % |
| medium | 0.30±0.13 | 0.77±0.09 | 0.50±0.15 | 0.03±0.01 | 8.2 % | 0.62 % | 33 % |
| large | 0.60±0.13 | 0.80±0.07 | 0.35±0.14 | 0.12±0.15 | 13.6 % | 4.40 % | 34 % |

Table 5. Deletion (lower = better) and insertion (higher = better) AUCs, mean ± std over 8 true-positive detections per bin, for D-RISE order vs random order; mean object side 13, 26, 47 and 116 px. Mass in box = share of saliency inside the target box; box area = share of the tile covered by the box; 90 % area = share of the tile holding 90 % of the saliency mass.

In every size bin, removing pixels in D-RISE order destroys the target detection far sooner than removing them in random order, and restoring them in D-RISE order recovers it far sooner (Table 5). For tiny objects (mean side 13 px) the deletion AUC is 0.08 ± 0.07 against 0.78 ± 0.09 for random order, and the insertion AUC is 0.41 ± 0.16 against 0.05 ± 0.03. The absolute deletion AUC grows with object size (0.60 for large objects), but this is largely geometric: on a log-fraction axis an ideal ordering must still remove roughly the pixels of the box, about 0.05 % of the tile for a tiny object and 4 % for a large one. The meaningful comparison is therefore within a bin.

The concentration statistics give a more nuanced picture. The target box holds 25 times more saliency mass than its area share for tiny objects, 17 times for small, 13 times for medium and 3 times for large objects, yet 90 % of the total mass is spread over 30 to 34 % of the tile in every bin. The maps thus have a sharp peak on the object and a broad, low-amplitude background, which is consistent with Monte-Carlo noise at N = 1,000 masks and with genuine use of context (road, neighbouring vehicles).

A first run of the study used linear 5 % deletion steps and the uncentred estimator. It produced a deletion AUC of 0.025 ± 0.000 in three of the four bins, which is exactly the minimum attainable with that step size, and a mass inside the box equal to the box's area share. Both symptoms were artefacts of the protocol, not properties of the maps; they motivated the log-spaced schedule and the mean-centred estimator described in Section 4.

## 6. Discussion

The size-binned evaluation makes the failure mode of a resize-to-640 detector explicit: the tiny bin, which holds 31 % of the validation objects, is detected at AP50 0.085. Two training-free remedies recover a large part of the gap. Their comparison is the most practical finding of this report. A 1024 px test-time input matches tiled inference overall at a fifth of the cost, so tiling is justified mainly when the tiniest objects matter most, and it must be combined with a full-frame pass to avoid splitting large objects. A natural next step is to train at the higher resolution as well, which removes the train/test scale mismatch that the 1024 px test-time setting currently has.

For explanations, the main lesson concerns evaluation. Deletion and insertion curves are usually computed with linear steps designed for ImageNet-sized objects. For aerial targets that cover a few hundred pixels of a 640 × 640 tile, that schedule cannot separate good maps from perfect ones. With a log-spaced schedule, D-RISE is clearly more faithful than chance for objects of every size, including 12 px pedestrians, which supports using black-box saliency to audit small-object detectors.

## 7. Limitations

One model size, one schedule, one seed: differences of a few tenths of a point between configurations are within seed noise, the size-bin pattern is not. Numbers are on the validation split, which also selected the checkpoint. Tiling buys recall with compute; the timing column reflects a laptop GPU with batched tiles. D-RISE with 1,000 masks is a noisy estimator for 16-px targets, and eight detections per bin show a trend rather than rank methods. The ignored-region filter follows the official toolkit's intent but is an independent implementation, so the numbers are not directly comparable with leaderboard entries.

## 8. Conclusion

We evaluated a compact YOLO11n detector on VisDrone with a size-binned, ignored-region-aware protocol, measured how much tiled inference and larger inputs recover for small objects, and studied the faithfulness of D-RISE explanations across object sizes. Small objects remain the dominant source of error. Tiling more than doubles AP50 on tiny objects, but a larger test-time input captures most of its benefit far more cheaply. D-RISE explanations are faithful even for the smallest objects once the faithfulness test is scaled to them. Code, metrics and figures are public and reproduce on a laptop.

## References

[1] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, H. Ling. Detection and Tracking Meet Drones Challenge. IEEE TPAMI 44(11), 2022. doi:10.1109/TPAMI.2021.3119563
[2] F. C. Akyon, S. O. Altinuc, A. Temizel. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. ICIP 2022. doi:10.1109/ICIP46576.2022.9897990
[3] V. Petsiuk, R. Jain, V. Manjunatha, V. I. Morariu, A. Mehra, V. Ordonez, K. Saenko. Black-box Explanation of Object Detectors via Saliency Maps. CVPR 2021. doi:10.1109/CVPR46437.2021.01128
[4] V. Petsiuk, A. Das, K. Saenko. RISE: Randomized Input Sampling for Explanation of Black-box Models. BMVC 2018. arXiv:1806.07421
[5] T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. doi:10.1007/978-3-319-10602-1_48
[6] G. Jocher, J. Qiu. Ultralytics YOLO11, version 8.3. 2024. https://github.com/ultralytics/ultralytics
[7] X. Zhu, S. Lyu, X. Wang, Q. Zhao. TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios. ICCV Workshops 2021. doi:10.1109/ICCVW54120.2021.00312
[8] G. Cheng et al. Towards Large-Scale Small Object Detection: Survey and Benchmarks. IEEE TPAMI 45(11), 2023. doi:10.1109/TPAMI.2023.3290594
[9] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017. doi:10.1109/ICCV.2017.74
[10] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, B. Kim. Sanity Checks for Saliency Maps. NeurIPS 2018. arXiv:1810.03292
