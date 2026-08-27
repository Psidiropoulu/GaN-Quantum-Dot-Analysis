# QD Detection

## Purpose

Develop, optimise and compare methods for automatically detecting individual InGaN QDs in processed AFM scans.

## Structure

* **`Classical-Optimised/`** — classical detection methods, optimisation and comparison.
* **`Biology-Inspired/`** — pretrained segmentation models tested or adapted for AFM QDs.
* **`Binary-CNN-Model/`** — task-specific binary U-Net training and prediction.

## Classical methods tested

* Laplacian of Gaussian (LoG)
* Otsu thresholding
* Hough circle transform
* Watershed segmentation
* Perona–Malik anisotropic diffusion
* Chambolle total-variation denoising
* Morphological Geodesic Active Contours
* Chan–Vese segmentation
* SIFT
* Gabor filtering

## Pretrained / external models tested

* Cellpose
* Omnipose
* StarDist
* MicroSAM
* Mask R-CNN

## Custom models
* Random forest 
* Stacked-ensemble Random forest (F1 0.9)
* Binary U-net


## Main notebooks

* **`npy-data-QD-DETECTION.ipynb`** — QD detection experiments using NumPy AFM data.
* **`npy-data-TRANSFORMS.ipynb`** — image transformations and intermediate detection experiments.

## Evaluation

Methods are compared using common ground-truth scans with metrics including:

* precision;
* recall;
* F1 score;
* Dice coefficient;
* IoU;
* QD count;
* object-level matching;
* QD feature errors.

Detailed analysis of missed QDs is kept in `QD-Analysis/`.
