import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def count_error(predicted_count, true_count):
    return predicted_count - true_count


def absolute_count_error(predicted_count, true_count):
    return abs(predicted_count - true_count)


def relative_count_error(predicted_count, true_count):
    if true_count == 0:
        return np.nan

    return abs(predicted_count - true_count) / true_count



# pixel-level metrics
def pixel_precision(predicted_mask, true_mask):
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    true_positive = np.count_nonzero(predicted_mask & true_mask)
    false_positive = np.count_nonzero(predicted_mask & ~true_mask)

    denominator = true_positive + false_positive
    return true_positive / denominator if denominator > 0 else 1.0


def pixel_recall(predicted_mask, true_mask):
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    true_positive = np.count_nonzero(predicted_mask & true_mask)
    false_negative = np.count_nonzero(~predicted_mask & true_mask)

    denominator = true_positive + false_negative
    return true_positive / denominator if denominator > 0 else 1.0


def pixel_f1_score(predicted_mask, true_mask):
    precision = pixel_precision(predicted_mask, true_mask)
    recall = pixel_recall(predicted_mask, true_mask)

    denominator = precision + recall
    return 2 * precision * recall / denominator if denominator > 0 else 0.0


def dice_score(predicted_mask, true_mask):
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    intersection = np.count_nonzero(predicted_mask & true_mask)
    denominator = np.count_nonzero(predicted_mask) + np.count_nonzero(true_mask)

    return 2 * intersection / denominator if denominator > 0 else 1.0


def iou_score(predicted_mask, true_mask):
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)

    intersection = np.count_nonzero(predicted_mask & true_mask)
    union = np.count_nonzero(predicted_mask | true_mask)

    return intersection / union if union > 0 else 1.0


# before localisation or parameter errors, need to march predicted QDs to ground-truth QDs
def match_qds(predicted_centres, true_centres, max_distance=5):
    predicted_centres = np.asarray(predicted_centres, dtype=float).reshape(-1, 2)
    true_centres = np.asarray(true_centres, dtype=float).reshape(-1, 2)

    if len(predicted_centres) == 0 or len(true_centres) == 0:
        return {
            "predicted_indices": np.empty(0, dtype=int),
            "true_indices": np.empty(0, dtype=int),
            "distances": np.empty(0, dtype=float),
            "unmatched_predicted_indices": np.arange(len(predicted_centres)),
            "unmatched_true_indices": np.arange(len(true_centres)),
        }

    distance_matrix = cdist(predicted_centres, true_centres)
    predicted_indices, true_indices = linear_sum_assignment(distance_matrix)
    distances = distance_matrix[predicted_indices, true_indices]

    valid = distances <= max_distance
    matched_predicted = predicted_indices[valid]
    matched_true = true_indices[valid]
    matched_distances = distances[valid]

    unmatched_predicted = np.setdiff1d(np.arange(len(predicted_centres)), matched_predicted)
    unmatched_true = np.setdiff1d(np.arange(len(true_centres)), matched_true)

    return {
        "predicted_indices": matched_predicted,
        "true_indices": matched_true,
        "distances": matched_distances,
        "unmatched_predicted_indices": unmatched_predicted,
        "unmatched_true_indices": unmatched_true,
    }



# Object level precision, recall, and F1 score
def object_detection_metrics(predicted_centres, true_centres, max_distance=5):
    matches = match_qds(predicted_centres, true_centres, max_distance=max_distance)

    true_positive = len(matches["predicted_indices"])
    false_positive = len(matches["unmatched_predicted_indices"])
    false_negative = len(matches["unmatched_true_indices"])

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = true_positive / precision_denominator if precision_denominator > 0 else 1.0
    recall = true_positive / recall_denominator if recall_denominator > 0 else 1.0

    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


# Localisation error metrics
def localisation_error(predicted_centres, true_centres, max_distance=5):
    matches = match_qds(predicted_centres, true_centres, max_distance=max_distance)
    distances = matches["distances"]

    if len(distances) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "rmse": np.nan,
            "maximum": np.nan,
            "distances": distances,
        }

    return {
        "mean": np.mean(distances),
        "median": np.median(distances),
        "rmse": np.sqrt(np.mean(distances**2)),
        "maximum": np.max(distances),
        "distances": distances,
    }

    localisation_nm = localisation_pixels * pixel_size_nm


# Parameter error metrics
def parameter_error(predicted_values, true_values):
    predicted_values = np.asarray(predicted_values, dtype=float)
    true_values = np.asarray(true_values, dtype=float)

    valid = np.isfinite(predicted_values) & np.isfinite(true_values)
    predicted_values = predicted_values[valid]
    true_values = true_values[valid]

    if len(predicted_values) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mean_signed_error": np.nan,
            "mean_relative_error": np.nan,
            "values": np.empty(0),
        }

    errors = predicted_values - true_values
    nonzero_true = true_values != 0

    relative_errors = np.abs(errors[nonzero_true] / true_values[nonzero_true])

    return {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean_signed_error": np.mean(errors),
        "mean_relative_error": np.mean(relative_errors) if len(relative_errors) > 0 else np.nan,
        "values": errors,
    }


def matched_parameter_errors(
    predicted_result,
    true_result,
    max_distance=5,
):
    predicted_centres = predicted_result["centres"]
    true_centres = true_result["centres"]

    matches = match_qds(
        predicted_centres,
        true_centres,
        max_distance=max_distance,
    )

    predicted_indices = matches["predicted_indices"]
    true_indices = matches["true_indices"]

    return {
        "height": parameter_error(
            predicted_result["heights"][predicted_indices],
            true_result["heights"][true_indices],
        ),
        "radius": parameter_error(
            predicted_result["radii"][predicted_indices],
            true_result["radii"][true_indices],
        ),
        "area": parameter_error(
            predicted_result["areas"][predicted_indices],
            true_result["areas"][true_indices],
        ),
    }



def evaluate_detection(
    predicted_result,
    true_result,
    max_distance=5,
):
    object_metrics = object_detection_metrics(
        predicted_result["centres"],
        true_result["centres"],
        max_distance=max_distance,
    )

    return {
        "count_error": len(predicted_result["centres"]) - len(true_result["centres"]),
        "absolute_count_error": abs(len(predicted_result["centres"]) - len(true_result["centres"])),
        "pixel_precision": pixel_precision(predicted_result["mask"], true_result["mask"]),
        "pixel_recall": pixel_recall(predicted_result["mask"], true_result["mask"]),
        "pixel_f1": pixel_f1_score(predicted_result["mask"], true_result["mask"]),
        "dice": dice_score(predicted_result["mask"], true_result["mask"]),
        "iou": iou_score(predicted_result["mask"], true_result["mask"]),
        "object_precision": object_metrics["precision"],
        "object_recall": object_metrics["recall"],
        "object_f1": object_metrics["f1"],
        "localisation": localisation_error(
            predicted_result["centres"],
            true_result["centres"],
            max_distance=max_distance,
        ),
        "parameters": matched_parameter_errors(
            predicted_result,
            true_result,
            max_distance=max_distance,
        ),
    }