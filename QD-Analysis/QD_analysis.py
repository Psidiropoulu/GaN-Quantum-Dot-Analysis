import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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



from skimage.measure import label, regionprops


def centres_from_binary_mask(binary_mask):
    binary_mask = np.asarray(binary_mask, dtype=bool)
    labelled_mask = label(binary_mask, connectivity=2)

    centres = [region.centroid for region in regionprops(labelled_mask)]

    if not centres:
        return np.empty((0, 2), dtype=float)

    return np.asarray(centres, dtype=float)


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


def safe_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_missed_qds(image, qd_df, noerror_name,):
    missed = qd_df[~qd_df[f"{noerror_name}_after"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.nanpercentile(image, [1, 99])

    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.scatter(missed["cx"], missed["cy"], facecolors="none", edgecolors="red", s=100, linewidths=1.5,)

    for _, row in missed.iterrows():
        ax.text(row["cx"] + 4, row["cy"], str(int(row["qd_id"])), fontsize=7)

    ax.set_title(f"{noerror_name}: "f"{len(missed)} missed QDs")

    plt.tight_layout()
    plt.show()


# General empirical porbability function 

def detection_probability_by_feature(qd_df, algorithm, feature, n_bins=6):
    detection_col = f"{safe_name(algorithm)}_after"
    data = qd_df[[feature, detection_col]].dropna().copy()
    data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")

    result = (data.groupby("bin", observed=True).agg(
            probability=(detection_col, "mean"),
            n=(detection_col, "size"),
            feature_mean=(feature, "mean"),
            feature_median=(feature, "median"),)
            .reset_index()
        )
    
    return result

def detection_probability_by_pit(qd_df, algorithms):
    rows = []

    for algorithm in algorithms:
        col = f"{safe_name(algorithm)}_after"

        for on_pit in [False, True]:
            subset = qd_df[qd_df["on_pit"] == on_pit]

            rows.append({
                "algorithm": algorithm,
                "location": "On pit" if on_pit else "Off pit",
                "probability": subset[col].mean(),
                "n": len(subset),
            })

    return pd.DataFrame(rows)


from sklearn.linear_model import LogisticRegression


def fit_detection_probability(qd_df, algorithm, feature):
    detection_col = f"{safe_name(algorithm)}_after"

    data = qd_df[[feature, detection_col]].dropna()

    X = data[[feature]].to_numpy()
    y = data[detection_col].astype(int).to_numpy()

    model = LogisticRegression()
    model.fit(X, y)

    x_grid = np.linspace(
        X.min(),
        X.max(),
        200,
    ).reshape(-1, 1)

    probability = model.predict_proba(x_grid)[:, 1]

    return model, x_grid[:, 0], probability



from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def fit_multivariable_detection_model(qd_df, algorithm):
    detection_col = f"{safe_name(algorithm)}_after"

    features = ["height", "local_contrast", "on_pit",]

    data = qd_df[features + [detection_col]].dropna().copy()

    X = data[features].astype(float)
    y = data[detection_col].astype(int)

    model = make_pipeline(StandardScaler(), LogisticRegression(),)
    model.fit(X, y)

    return model



# Complemetary probabilities of QD detection based on whether or not they have been detected by some other algorithm 
def probability_detected_given_missed(qd_df, detector, missed_by):
    detector_col = f"{safe_name(detector)}_after"
    missed_col = f"{safe_name(missed_by)}_after"

    subset = qd_df[~qd_df[missed_col]]

    if len(subset) == 0:
        return np.nan

    return subset[detector_col].mean()


# Full pairwise heatmap 
def build_rescue_matrix(qd_df, algorithms):
    matrix = pd.DataFrame(
        index=algorithms,
        columns=algorithms,
        dtype=float,
    )

    for missed_by in algorithms:
        for detector in algorithms:
            matrix.loc[detector, missed_by] = (
                probability_detected_given_missed(
                    qd_df,
                    detector=detector,
                    missed_by=missed_by,
                )
            )

    return matrix