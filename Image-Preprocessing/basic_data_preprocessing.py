from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter

def mad(values):
    """Median absolute deviation scaled like standard deviation."""
    values = np.asarray(values, dtype=float)
    centre = np.nanmedian(values)

    return 1.4826 * np.nanmedian(
        np.abs(values - centre)
    )

def robust_normalise(z):
    median = np.nanmedian(z)
    mad = np.nanmedian(np.abs(z - median))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale == 0:
        scale = np.nanstd(z)

    return (z - median) / scale


def subtract_global_plane(
    z,
    n_iterations=6,
    sigma=2.5,
):
    """
    Fit and subtract z = ax + by + c using robust outlier rejection.
    """
    z = np.asarray(z, dtype=float)

    rows, columns = z.shape
    yy, xx = np.indices(z.shape)

    finite = np.isfinite(z)
    fit_mask = finite.copy()

    if np.count_nonzero(finite) < 3:
        raise ValueError("Not enough finite pixels for plane fitting.")

    coefficients = np.zeros(3, dtype=float)

    for _ in range(n_iterations):

        design = np.column_stack(
            (
                xx[fit_mask],
                yy[fit_mask],
                np.ones(np.count_nonzero(fit_mask)),
            )
        )

        coefficients, *_ = np.linalg.lstsq(
            design,
            z[fit_mask],
            rcond=None,
        )

        plane = (
            coefficients[0] * xx
            + coefficients[1] * yy
            + coefficients[2]
        )

        residual = z - plane

        centre = np.nanmedian(
            residual[fit_mask]
        )

        scale = mad(
            residual[fit_mask]
        )

        if not np.isfinite(scale) or scale == 0:
            break

        fit_mask = (
            finite
            & (
                np.abs(residual - centre)
                < sigma * scale
            )
        )

        if np.count_nonzero(fit_mask) < 3:
            break

    plane = (
        coefficients[0] * xx
        + coefficients[1] * yy
        + coefficients[2]
    )

    flattened = z - plane

    return flattened, plane


def robust_polynomial_line_flatten(
    z,
    degree=1,
    n_iterations=6,
    sigma=2.5,
    axis=1,
):
    """
    Flatten each scan line using robust polynomial fitting.

    axis=1 processes horizontal rows.
    axis=0 processes vertical columns.
    """
    z = np.asarray(z, dtype=float)

    if axis == 1:
        working = z.copy()
    elif axis == 0:
        working = z.T.copy()
    else:
        raise ValueError("axis must be 0 or 1")

    result = np.full_like(
        working,
        np.nan,
        dtype=float,
    )

    n_lines, n_points = working.shape
    x = np.linspace(-1.0, 1.0, n_points)

    for line_index in range(n_lines):

        line = working[line_index]
        finite = np.isfinite(line)

        if np.count_nonzero(finite) <= degree + 1:
            result[line_index] = line
            continue

        fit_mask = finite.copy()
        coefficients = None

        for _ in range(n_iterations):

            if np.count_nonzero(fit_mask) <= degree + 1:
                break

            coefficients = np.polyfit(
                x[fit_mask],
                line[fit_mask],
                degree,
            )

            background = np.polyval(
                coefficients,
                x,
            )

            residual = line - background

            centre = np.nanmedian(
                residual[fit_mask]
            )

            scale = mad(
                residual[fit_mask]
            )

            if not np.isfinite(scale) or scale == 0:
                break

            new_mask = (
                finite
                & (
                    np.abs(residual - centre)
                    < sigma * scale
                )
            )

            if np.array_equal(new_mask, fit_mask):
                break

            fit_mask = new_mask

        if coefficients is None:
            result[line_index] = line
            continue

        background = np.polyval(
            coefficients,
            x,
        )

        result[line_index] = (
            line - background
        )

    if axis == 0:
        return result.T

    return result


def align_scanline_offsets(
    z,
    smoothing_window=21,
):
    """
    Remove rapid row-to-row height offsets.
    """
    z = np.asarray(z, dtype=float)

    if smoothing_window % 2 == 0:
        smoothing_window += 1

    row_offsets = np.nanmedian(
        z,
        axis=1,
    )

    valid = np.isfinite(row_offsets)

    if not np.any(valid):
        return z.copy()

    filled = row_offsets.copy()

    if not np.all(valid):
        row_numbers = np.arange(
            len(row_offsets)
        )

        filled[~valid] = np.interp(
            row_numbers[~valid],
            row_numbers[valid],
            row_offsets[valid],
        )

    smooth_offsets = median_filter(
        filled,
        size=smoothing_window,
        mode="nearest",
    )

    rapid_offset_error = (
        filled - smooth_offsets
    )

    return (
        z - rapid_offset_error[:, None]
    )

def robust_2d_background(
    image,
    sigma_y=30,
    sigma_x=60,
    iterations=8,
    threshold=2.5,
):
    """
    Estimate and subtract a smooth two-dimensional background.
    """
    image = np.asarray(
        image,
        dtype=float,
    )

    finite = np.isfinite(image)

    if not np.any(finite):
        return image.copy()

    filled = image.copy()
    filled[~finite] = np.nanmedian(
        image[finite]
    )

    background = gaussian_filter(
        filled,
        sigma=(sigma_y, sigma_x),
        mode="reflect",
    )

    mask = finite.copy()

    for _ in range(iterations):

        residual = image - background

        centre = np.nanmedian(
            residual[mask]
        )

        scale = mad(
            residual[mask]
        )

        if not np.isfinite(scale) or scale == 0:
            break

        mask = (
            finite
            & (
                np.abs(residual - centre)
                < threshold * scale
            )
        )

        weighted_values = np.where(
            mask,
            image,
            0.0,
        )

        weights = mask.astype(float)

        smooth_values = gaussian_filter(
            weighted_values,
            sigma=(sigma_y, sigma_x),
            mode="reflect",
        )

        smooth_weights = gaussian_filter(
            weights,
            sigma=(sigma_y, sigma_x),
            mode="reflect",
        )

        background = (
            smooth_values
            / np.maximum(
                smooth_weights,
                1e-12,
            )
        )

    flattened = image - background
    flattened[~finite] = np.nan

    return flattened


def preprocess_afm(
    z,
    remove_plane=True,
    line_flatten=True,
    align_rows=True,
    remove_2d_background=False,
):
    """
    Apply selected AFM preprocessing steps.
    """
    result = np.asarray(z, dtype=float).copy()
    stages = {"raw": result.copy()}

    if remove_plane:
        result, _ = subtract_global_plane(result, n_iterations=6, sigma=2.5)
        stages["plane removed"] = (result.copy())

    if line_flatten:
        result = robust_polynomial_line_flatten(
            result,
            degree=1,
            n_iterations=6,
            sigma=2.5,
            axis=1,
        )

        stages["line flattened"] = (result.copy())

    if align_rows:
        result = align_scanline_offsets(result, smoothing_window=21)
        stages["rows aligned"] = (result.copy())

    if remove_2d_background:
        result = robust_2d_background(
            result,
            sigma_y=30,
            sigma_x=60,
            iterations=8,
            threshold=2.5,
        )

        stages["2D background removed"] = (
            result.copy()
        )

    result = (result - np.nanmedian(result))
    stages["final"] = result.copy()
    return result, stages


def fft_bandpass_filter(image, low_sigma=5, high_sigma=40,):
    image = np.asarray(image, dtype=float)
    small_scale = gaussian_filter(image, sigma=low_sigma,)
    large_scale = gaussian_filter(image, sigma=high_sigma,)
    filtered = small_scale - large_scale
    return filtered
