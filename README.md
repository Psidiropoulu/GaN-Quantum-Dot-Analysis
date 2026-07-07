# GaN Quantum Dot AFM Analysis

Pipeline for analysing AFM images of semiconductor quantum dots.

## Pipeline

.spm
↓
Gwyddion
↓
.npy conversion
↓
AFM preprocessing
↓
FFT / filtering
↓
dot detection
↓
machine learning analysis

## Environment

Python 3.11
TensorFlow
pySPM
scikit-image