# Image Preprocessing

## Purpose

Correct AFM background variation, scan-line artefacts and noise before QD detection, while preserving QD height, shape and local contrast.

## Methods

- **Global plane subtraction** — removes large-scale sample tilt using robust plane fitting.
- **Polynomial line flattening** — removes scan-line background trends using polynomial fits.
- **Scan-line offset alignment** — corrects row-to-row height offsets.
- **Weighted scan-line smoothing / destriping** — reduces horizontal stripe artefacts using weighted neighbouring-line information.
- **Robust 2D background subtraction** — removes slowly varying 2D background structure.
- **Foreground-aware flattening** — excludes likely foreground features from the background fit to reduce distortion of QD peaks.
- **Gaussian smoothing** — reduces high-frequency noise.
- **Median filtering** — suppresses isolated noise while preserving edges.
- **FFT band-pass filtering** — removes selected low- and high-spatial-frequency components.

## Main files

- `basic_data_preprocessing.py` — standard preprocessing functions and combined pipelines.
- `foreground_preprocessing.py` — foreground-aware background correction methods.
- `preprocessing_metrics.py` — quantitative measurements of background quality and preprocessing performance.
- `preprocessing-COMPARISON.ipynb` — comparison of preprocessing methods across AFM scans.

## Output

Processed AFM height maps suitable for consistent QD detection and quantitative analysis.