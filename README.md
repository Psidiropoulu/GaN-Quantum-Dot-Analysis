# GaN Quantum Dot Analysis

Automated detection and quantitative analysis of GaN/InGaN quantum dots from atomic force microscopy (AFM) data.

The project covers the full workflow from raw AFM data conversion and preprocessing to quantum-dot detection, manual ground-truth construction, machine-learning segmentation, object-level evaluation, and validation of physical QD parameters such as height and FWHM.

## Project Pipeline

```text
SPM data
   ↓
Gwyddion conversion
   ↓
NumPy AFM arrays
   ↓
AFM preprocessing
   ↓
QD candidate detection
   ↓
Physical filtering / parameter measurement
   ↓
Comparison with manually curated ground truth
```

The current aim is not only to maximise segmentation performance, but to identify a detection pipeline that is accurate, robust across scans, and preserves physically meaningful QD measurements.

## Repository Structure

### `Data-Conversion/`

Tools for converting microscope data into the NumPy format used by the rest of the project.

```text
.spm → .gwy → .npy
```

The folder also contains unlabelled AFM datasets used for analysis.

### `Image-Preprocessing/`

AFM preprocessing and artefact-removal methods.

`basic_data_preprocessing.py` contains the standard preprocessing pipeline, including robust normalisation, global plane removal, robust scan-line polynomial flattening, scan-line offset alignment, and optional 2D background removal.

`foreground_preprocessing.py` contains a foreground-aware background-flattening method designed to reduce the influence of QD peaks on the fitted background surface.

### `Ground-Truth/`

Ground-truth construction, model training, model prediction, and physical-parameter validation.

This includes manually curated binary QD masks, the QD labelling GUI, false-positive and false-negative corrections, exported QD feature tables, U-Net training and prediction code, reference height/FWHM measurements, and automated-vs-manual parameter error analysis.

### `detection_algorithms.py`

Classical QD detection methods implemented using a common result structure.

Methods investigated include:

- Laplacian of Gaussian (LoG)
- Otsu and Multi-Otsu thresholding
- Circular and General Hough transforms
- phase symmetry / phase congruency
- SIFT
- Morphological Geodesic Active Contours
- Chan–Vese segmentation
- watershed segmentation
- Gabor-filter-based methods
- anisotropic-diffusion-assisted detection

Detection outputs can include binary masks, QD centres, local heights, radii, and areas.

### `detection_errors.py`

Quantitative evaluation of automatic detections against ground truth.

The project evaluates both pixel-level and object-level performance, including precision, recall, F1 score, Dice coefficient, intersection over union, TP/FP/FN counts, centre localisation, and parameter errors for matched QDs.

### `QD-gui.py`

Interactive interface for AFM analysis and inspection of QD detections.

The GUI supports visual comparison of detection algorithms and parameters and allows detected features and measurements to be inspected directly.

### Notebooks

`npy-data-QD-DETECTION.ipynb` — exploration of QD detection methods and the image-processing concepts behind them.

`method-COMPARISON.ipynb` — comparison of alternative QD detection algorithms.

`npy-data-QD-ANALYSIS.ipynb` — analysis of detected QD physical characteristics, including comparison of QDs grown on different layers.

`npy-data-TRANSFORMS.ipynb` — testing and visualisation of preprocessing and image transformations.

## AFM Preprocessing

Raw AFM scans can contain sample tilt, scan-line offsets, stripes, noise, and slowly varying background structure.

The standard preprocessing workflow is approximately:

```text
raw AFM
   ↓
global plane subtraction
   ↓
robust line flattening
   ↓
scan-line alignment
   ↓
optional background correction
```

A second foreground-aware method iteratively excludes probable QD regions from the background fit so that large foreground features do not distort the estimated surface.

Noise reduction experiments include both Gaussian smoothing and anisotropic diffusion. Gaussian smoothing is linear and isotropic, whereas anisotropic diffusion adapts the amount of smoothing to the local image gradient in order to preserve stronger feature boundaries.

## Ground Truth

Ground-truth masks are created and manually corrected using a dedicated GUI.

The tool allows removal of false-positive detections, addition of missed QDs, inspection of local QD height, FWHM measurement, export of corrected masks, and export of individual QD feature tables.

The resulting masks are used for classical-algorithm evaluation and supervised machine-learning training.

## QD Detection

Several complementary approaches are being compared because they make different assumptions about QD appearance.

LoG and SIFT operate across multiple spatial scales. Hough methods impose geometric assumptions. Active contours refine object boundaries using either gradients or regional intensity differences. Watershed separates nearby features. Phase-based and Gabor methods provide alternatives to purely intensity-based detection.

Candidate detections can also be filtered using a minimum local-height criterion:

```text
QD local height ≥ 6 Å
```

where height is measured relative to the local surrounding background.

## Machine Learning

The repository also contains a U-Net-like convolutional neural network for pixelwise QD segmentation.

AFM images and binary masks are divided into training patches, and the trained model predicts a probability map for unseen scans. A probability threshold is then applied to obtain a binary QD segmentation.

The ML approach is evaluated using the same ground-truth and object-level evaluation framework as the classical methods.

## Physical Parameter Validation

Detection quality alone is not sufficient for this project: the measured QD properties must also remain physically meaningful.

Automatically estimated QD height and FWHM are therefore compared with independently measured reference values.

For a matched QD:

```text
height error = automatic height − reference height
FWHM error   = automatic FWHM − reference FWHM
```

The comparison includes absolute and relative errors, MAE, RMSE, and error distributions.

## Current Questions

Current work focuses on:

- determining which preprocessing pipeline improves detection without distorting QD morphology
- identifying the strongest classical detection baseline
- testing whether combinations of complementary methods outperform individual detectors
- comparing classical detection with U-Net segmentation
- improving automatic height and FWHM measurements
- determining whether remaining parameter errors depend systematically on QD size, morphology, or local background
- moving toward a fully automated pipeline from AFM data to final QD statistics

## Methodology Overview

A more detailed methodology document, including equations, algorithm explanations, and figures, is maintained in:

```text
Methodology/main-overview.tex
```

and can be compiled to PDF with LaTeX.

## Status

This repository is under active development. The codebase contains both stable analysis components and experimental notebooks/scripts used to compare alternative approaches.
