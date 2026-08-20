import numpy as np


def background_metrics(image, qd_mask):
    image = np.asarray(image, dtype=float)
    qd_mask = np.asarray(qd_mask, dtype=bool)

    background_mask = (~qd_mask) & np.isfinite(image)
    background = image[background_mask]

    median = np.median(background)

    background_mad = 1.4826 * np.median(
        np.abs(background - median)
    )

    background_std = np.std(background)

    return {
        "background_mad": background_mad,
        "background_std": background_std,
    }

def residual_plane_slope(image, qd_mask):
    image = np.asarray(image, dtype=float)
    qd_mask = np.asarray(qd_mask, dtype=bool)

    yy, xx = np.indices(image.shape)

    mask = (~qd_mask) & np.isfinite(image)

    A = np.column_stack([
        xx[mask],
        yy[mask],
        np.ones(np.count_nonzero(mask)),
    ])

    coefficients, *_ = np.linalg.lstsq(
        A,
        image[mask],
        rcond=None,
    )

    a, b, c = coefficients

    slope = np.sqrt(a**2 + b**2)

    return {
        "x_slope": a,
        "y_slope": b,
        "residual_slope": slope,
    }

def row_offset_std(image, qd_mask):
    image = np.asarray(image, dtype=float)
    qd_mask = np.asarray(qd_mask, dtype=bool)

    row_medians = []

    for row in range(image.shape[0]):

        valid = (
            (~qd_mask[row])
            & np.isfinite(image[row])
        )

        if np.any(valid):
            row_medians.append(
                np.median(image[row, valid])
            )

    row_medians = np.asarray(row_medians)

    return np.std(row_medians)

def quantify_background(image, qd_mask):

    metrics = background_metrics(
        image,
        qd_mask,
    )

    metrics.update(
        residual_plane_slope(
            image,
            qd_mask,
        )
    )

    metrics["row_offset_std"] = row_offset_std(
        image,
        qd_mask,
    )

    return metrics