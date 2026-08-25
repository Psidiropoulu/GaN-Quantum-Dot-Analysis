# Methodology

## Purpose

Document the methods used to build and validate the automated InGaN QD analysis pipeline.
The document contains overview and some written analysis. 

## Structure

The methodology follows the main analysis workflow:

1. **AFM data conversion** — extraction of numerical height data from raw AFM files.
2. **Image preprocessing** — correction of tilt, background variation, striping and noise.
3. **Ground-truth creation** — manual QD labelling and reference measurements.
4. **QD detection** — classical, pretrained and custom neural-network approaches.
5. **Detector evaluation** — object-level precision, recall, F1 and measurement accuracy.
6. **QD property extraction** — automated height, size and morphology measurements.
7. **QD population analysis** — comparison of QDs inside and outside GaN pits.
8. **Detection failure analysis** — identification of QD properties associated with missed detections.