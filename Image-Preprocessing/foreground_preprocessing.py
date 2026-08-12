import numpy as np
from scipy.ndimage import binary_dilation


"""

This method is inspired by this article: https://www.beilstein-journals.org/bjnano/articles/9/91

Automated foreground-aware flattening is the strongest classical baseline. 

Rather than fitting a plane or polynomial to every pixel:
1. segment foreground features, 
2. exclude them from the background fit, subtract a polynomial surface. 

Wang et al. automate this two-step procedure and add sliding-window polynomial fitting for complex background trends. 
For quantum dots, this is preferable to blindly fitting across dot peaks, which can subtract part of the objects you want to measure.

"""

import basic_data_preprocessing as bdp


def _polynomial_design_matrix(x, y, degree):
    """
    Construct a 2D polynomial design matrix.

    degree=1: 1, x, y
    degree=2: 1, y, y^2, x, xy, x^2

    More generally includes all x^i y^j for i+j <= degree.
    """
    columns = []

    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            columns.append((x ** i) * (y ** j))

    return np.column_stack(columns)


def fit_polynomial_surface(z, fit_mask=None, degree=2):
    """
    Fit a 2D polynomial surface to selected pixels.

    Parameters
    ----------
    z : 2D ndarray AFM height image.

    fit_mask : 2D bool ndarray or None
        Pixels allowed to contribute to the background fit.
        If None, all finite pixels are used.

    degree : int, Polynomial degree.

    Returns
    -------
    surface : 2D ndarray
        Fitted polynomial background.

    coefficients : 1D ndarray
        Least-squares polynomial coefficients.
    """
    z = np.asarray(z, dtype=float)
    rows, columns = z.shape
    yy, xx = np.indices(z.shape, dtype=float)

    # Normalising coordinates keeps polynomial fitting
    # numerically better behaved.
    if columns > 1:
        xx = (2.0 * xx / (columns - 1) - 1.0)
    else:
        xx[:] = 0.0

    if rows > 1:
        yy = (2.0 * yy / (rows - 1) - 1.0)
    else:
        yy[:] = 0.0

    finite = np.isfinite(z)

    if fit_mask is None:
        fit_mask = finite
    else:
        fit_mask = (np.asarray(fit_mask, dtype=bool) & finite)

    minimum_points = ((degree + 1) * (degree + 2) // 2)

    if (np.count_nonzero(fit_mask) < minimum_points):
        raise ValueError(
            "Not enough background pixels "
            "for polynomial surface fitting."
        )

    design = _polynomial_design_matrix(xx[fit_mask], yy[fit_mask], degree)
    coefficients, *_ = np.linalg.lstsq(design, z[fit_mask], rcond=None)
    full_design = _polynomial_design_matrix(xx.ravel(), yy.ravel(), degree)
    surface = (full_design @ coefficients).reshape(z.shape)

    return surface, coefficients


def foreground_aware_flatten(
    z,
    degree=2,
    foreground_sigma=2.5,
    dilation_iterations=3,
    n_iterations=4,
):
    """
    Foreground-aware polynomial flattening for AFM images.
    Designed primarily for convex foreground features such as quantum dots.

    The procedure is:

        1. Fit a polynomial surface.
        2. Calculate residual heights.
        3. Detect sufficiently high positive residuals.
        4. Dilate the foreground mask so QD edges are excluded.
        5. Refit the polynomial using background pixels only.
        6. Repeat.
        7. Subtract the final background surface.

    Parameters
    ----------
    z : 2D ndarray, Raw AFM height image.

    degree : int, default=2, Degree of 2D polynomial background.

    foreground_sigma : float, default=2.5
        Residual threshold used for detecting raised foreground features.

    dilation_iterations : int, default=3
        Number of binary dilation iterations around detected foreground pixels.

        This prevents the slopes/edges of QDs from entering the background fit.

    n_iterations : int, default=4
        Number of foreground detection / background refitting iterations.

    Returns
    -------
    flattened : 2D ndarray. Flattened AFM image.
    background : 2D ndarray. Estimated polynomial background.
    foreground_mask : 2D bool ndarray. Pixels excluded as foreground.
    background_mask : 2D bool ndarray. Pixels used for the final background fit.
    """
    z = np.asarray(z, dtype=float,)
    finite = np.isfinite(z)

    if not np.any(finite):
        return (
            z.copy(),
            np.full_like(z, np.nan),
            np.zeros_like(z, dtype=bool),
            np.zeros_like(z, dtype=bool),
        )

    # -------------------------------------------------------
    # Initial fit.
    # -------------------------------------------------------

    background_mask = finite.copy()
    background, _ = fit_polynomial_surface(z, fit_mask=background_mask, degree=degree)
    foreground_mask = np.zeros_like(z, dtype=bool)

    # -------------------------------------------------------
    # Iteratively: fit -> find QDs -> exclude them -> refit
    # -------------------------------------------------------

    for _ in range(n_iterations):

        residual = z - background
        valid_residuals = residual[background_mask]
        centre = np.nanmedian(valid_residuals)

        scale = bdp.mad(valid_residuals)

        if (not np.isfinite(scale) or scale == 0):
            break

        # QDs are convex / raised objects, therefore only detect POSITIVE deviations.
        new_foreground = (finite & (residual > centre + foreground_sigma * scale))

        # Expand the mask around each detected QD so the shoulders of the dot cannot influence the fit.
        if dilation_iterations > 0:
            new_foreground = binary_dilation(new_foreground, iterations=dilation_iterations)

        # Keep foreground detected in previous iterations.
        foreground_mask |= new_foreground
        background_mask = (finite & ~foreground_mask)

        if np.count_nonzero(background_mask) < 10:
            break

        background, _ = fit_polynomial_surface(z, fit_mask=background_mask, degree=degree,)

    # -------------------------------------------------------
    # Final background fit.
    # -------------------------------------------------------

    background, _ = fit_polynomial_surface(z, fit_mask=background_mask, degree=degree,)
    flattened = z - background

    flattened[~finite] = np.nan
    background[~finite] = np.nan

    return (
        flattened,
        background,
        foreground_mask,
        background_mask,
    )


def foreground_preprocess_afm(
    z,
    remove_plane=True,
    foreground_flatten=False,
    line_flatten=True,
    align_rows=True,
    remove_2d_background=False,
):
    """
    Apply selected AFM preprocessing steps.
    """
    result = np.asarray(z, dtype=float).copy()
    stages = {"raw": result.copy(),}

    if foreground_flatten:
        (result,
            foreground_background,
            foreground_mask,
            background_mask,
        ) = foreground_aware_flatten(
            result,
            degree=2,
            foreground_sigma=2.5,
            dilation_iterations=3,
            n_iterations=4,
        )

        stages["foreground mask"] = (
            foreground_mask.copy()
        )

        stages["foreground background"] = (
            foreground_background.copy()
        )

        stages["foreground flattened"] = (
            result.copy()
        )

    if remove_plane and not foreground_flatten:
        result, _ = bdp.subtract_global_plane(result, n_iterations=6, sigma=2.5)
        stages["plane removed"] = (result.copy())

    if line_flatten:
        result = bdp.robust_polynomial_line_flatten(
            result,
            degree=1,
            n_iterations=6,
            sigma=2.5,
            axis=1,
        )

        stages["line flattened"] = (result.copy())

    if align_rows:
        result = bdp.align_scanline_offsets(result, smoothing_window=21)
        stages["rows aligned"] = (result.copy())

    if remove_2d_background:
        result = bdp.robust_2d_background(
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