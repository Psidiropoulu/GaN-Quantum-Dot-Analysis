# QD Analysis

## Purpose

Analyse QD properties, spatial distributions and detector failure modes after QDs have been labelled or detected.
Investigate what are QD properties that are causing a detection algorithm/ U-net to fail and use this data in order to reate a more aware model which could have interpretability.

## Main files

* **`QD_analysis.py`** — reusable functions for extracting and analysing individual QD properties.
* **`pit_distribution_analysis.ipynb`** — compares QD property distributions inside and outside GaN pits.
* **`qd_detection_failure_analysis.ipynb`** — analyses which QD properties are associated with missed detections.

## Properties analysed

* local-background-corrected height;
* radius / width;
* HWHM;
* area;
* circularity;
* eccentricity;
* solidity;
* QD position;
* pit / non-pit classification.

## Main analyses

* QD morphology distributions;
* pit vs non-pit comparison;
* detected vs missed QDs;
* comparison with manual measurements;
* detector-dependent bias in recovered QD populations.

## Output

QD-level measurements and distributions used to interpret both detector performance and the physical QD population.
