# Automated Analysis of InGaN Quantum Dots

Automated detection and quantitative analysis of InGaN quantum dots on and outside of the GaN pits from AFM measurements.

## Purpose

Develop a reproducible pipeline that can take raw AFM data and:

1. convert it into usable numerical data;
2. remove background and scan artefacts;
3. detect individual QDs;
4. measure their physical and morphological properties;
5. evaluate detection accuracy against manually labelled ground truth;
6. analyse QD distributions across different surface environments;
7. analyse why certain algorithms fail to detect certain QDs;
8. create a more aware model.

This is a circular pipeline, as, of course, the more labelled data the better. Which in turn changes model's confidence thresholds.
**`QD-gui.py/`** is the app that labels the QDs based on the algorithm selected.

## Repository structure

* **`Data-Conversion/`** — conversion of raw AFM data into Gwyddion and NumPy formats.
* **`Ground-Truth/`** — manually labelled QDs and reference measurements.
* **`Image-Preprocessing/`** — AFM flattening, destriping, background removal and filterin (all custom methods, created to simplify the workflow and quality of the images).
* **`Detection/`** — classical, pretrained and custom QD detection methods.
* **`QD-Analysis/`** — QD morphology, pit distributions and detector failure analysis.
* **`Methodology/`** — more detailed project overview with better explanation behind each of the methods used.

## Detection approaches

The project compares:

* classical image-processing methods;
* pretrained segmentation models;
* fine-tuned segmentation models;
* a task-specific binary U-Net.

Methods are evaluated primarily using object-level precision, recall and F1 score, together with errors in recovered QD properties.
Image pre-processing is also evaluated for each of the algorithms based on same characteristics to understand which methods help QD detection the most. 

## Analysis

Detected QDs are characterised using height, size and shape measurements. Their distributions are compared between QDs inside and outside GaN pits, and detection failures are analysed as a function of QD properties.
