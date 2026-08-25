# Data Conversion

## Purpose

Converting raw AFM data into formats used by the rest of the project.
Done so that the user doesn't have to open Gwyddion everytime they are trying to detect and analyse quantum dots on the image.

## Contents

- Read AFM `.spm` files.
- Extract relevant height channels. Channel 0 is used for the height.
- Convert data to NumPy arrays, so that it is easier to analyse the data.
- Preserve scan metadata where required.

Converted data is passed to `Image-Preprocessing/`.

## Main files

This folder contains the scripts and notebooks used for importing and converting the original AFM measurements.

## Output

NumPy-based AFM height data used by `Image-Preprocessing/`, `Ground-Truth/` and the downstream analysis pipeline.