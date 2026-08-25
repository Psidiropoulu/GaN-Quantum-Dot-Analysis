# Ground Truth

## Purpose

Create and store labelled reference data for detector training, evaluation and validation of automated QD measurements.

## Structure

* **`Images/`** — AFM images used for manual labelling.
* **`NPY-Ground-Truth/`** — labelled AFM scans and QD masks stored as NumPy arrays.
* **`Height-HWFM/`** — manual QD height and HWHM measurements used for comparison with automated measurements.

## Main file

* **`QD-gui-npy-TRAINING-SETS.py`** — interactive GUI used to manually label QDs and generate ground-truth masks / training sets.

## Used for

* QD masks and positions;
* QD counts;
* training and test datasets;
* object-level detector evaluation;
* comparison of automated and manual QD height / width measurements.

The same ground-truth data are used across classical and machine-learning detection methods wherever possible.
