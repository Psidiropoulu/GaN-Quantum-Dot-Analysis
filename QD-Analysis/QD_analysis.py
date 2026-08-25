import numpy as np
import pandas as pd

from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from skimage.measure import label, regionprops


def extract_qd_features(image, ground_truth_mask, background_inner=3, background_outer=8):
    image = np.asarray(image, dtype=float)
    gt_mask = np.asarray(ground_truth_mask, dtype=bool)

    labelled = label(gt_mask, connectivity=2)
    regions = regionprops(labelled)

    centres = np.asarray([r.centroid for r in regions], dtype=float)

    rows = []

    for i, region in enumerate(regions):

        qd_mask = labelled == region.label

        inner = binary_dilation(qd_mask, iterations=background_inner)
        outer = binary_dilation(qd_mask, iterations=background_outer)

        background_ring = (outer & ~inner & ~gt_mask)

        qd_values = image[qd_mask & np.isfinite(image)]

        bg_values = image[background_ring & np.isfinite(image)]

        if len(qd_values) == 0:
            continue

        if len(bg_values):
            bg_median = np.nanmedian(bg_values)
            bg_std = np.nanstd(bg_values)

            bg_mad = (1.4826 * np.nanmedian(np.abs(bg_values - np.nanmedian(bg_values))))
        else:
            bg_median = np.nan
            bg_std = np.nan
            bg_mad = np.nan

        height = np.nanmax(qd_values) - bg_median
        contrast = np.nanmedian(qd_values) - bg_median

        if len(centres) > 1:
            distances = np.linalg.norm(centres - centres[i], axis=1)
            distances[i] = np.inf
            nearest_distance = np.min(distances)
        else:
            nearest_distance = np.nan

        cy, cx = region.centroid
        radius = np.sqrt(region.area / np.pi)

        # Distance from nearest image edge
        edge_distance = min(cy, cx, image.shape[0] - 1 - cy, image.shape[1] - 1 - cx)

        rows.append({
            "qd_id": region.label,

            "cy": cy,
            "cx": cx,

            "height": height,
            "area": region.area,
            "radius": radius,

            "local_contrast": contrast,
            "local_background_std": bg_std,
            "local_background_mad": bg_mad,

            "nearest_qd_distance": nearest_distance,
            "edge_distance": edge_distance,
        })

    return pd.DataFrame(rows)


def match_detections(qd_df, predicted_centres, tolerance=3.0,):

    gt_centres = qd_df[["cy", "cx"]].to_numpy(dtype=float)

    predicted_centres = np.asarray(predicted_centres, dtype=float)
    predicted_centres = predicted_centres.reshape(-1, 2)

    detected = np.zeros(len(gt_centres), dtype=bool)
    detection_distance = np.full(len(gt_centres), np.nan)

    if len(predicted_centres) == 0:
        return detected, detection_distance

    distances = cdist(gt_centres, predicted_centres)

    gt_indices, pred_indices = (linear_sum_assignment(distances))

    for gt_i, pred_i in zip(gt_indices, pred_indices,):
        distance = distances[gt_i, pred_i]
        if distance <= tolerance:
            detected[gt_i] = True
            detection_distance[gt_i] = distance

    return detected, detection_distance