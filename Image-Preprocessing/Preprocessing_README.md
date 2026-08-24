# Image Preprocessing

Quick note: this folder contains the preprocessing methods used to prepare the AFM scans before QD detection. The main aim is to remove scan artefacts and background variation while preserving the QD features as much as possible.

The main methods included here are:

* **Global plane subtraction** — removes large-scale sample tilt.
* **Robust line flattening** — corrects scan-line background trends using polynomial fitting.
* **Scan-line alignment** — reduces row-to-row offsets and horizontal striping.
* **2D background removal** — removes slowly varying background structure.
* **Foreground-aware flattening** — excludes likely QD regions while fitting the background, reducing the risk of fitting through the QD peaks themselves.

There are also supporting functions for preprocessing comparison and background-quality measurements, used to quantify how much each method improves scan flatness and noise characteristics.

## Main files

* **`basic_data_preprocessing.py`**
  Contains the standard preprocessing pipeline and its individual stages.

* **`foreground_preprocessing.py`**
  Contains the foreground-aware polynomial flattening approach.

* **`preprocessing_metrics.py`**
  Contains metrics used to compare preprocessing methods quantitatively.

The preprocessing methods are generally intended to be tested individually as well as in combination, since different scans can contain different levels of tilt, striping, noise, and slowly varying background structure.
