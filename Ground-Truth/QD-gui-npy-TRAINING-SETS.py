from __future__ import annotations

import os
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import disk, white_tophat
from skimage.measure import label, regionprops
from skimage.draw import disk as draw_disk
from skimage.filters import threshold_otsu


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_NPY_DIR = PROJECT_DIR / "Data-Conversion/NPY"

# Physical QD acceptance requirement:
# 6 Å = 0.6 nm = 6e-10 m.
MINIMUM_QD_HEIGHT_M = 6e-10

# Local-height measurement settings.
# The apex search is fixed, but the background ring scales with each QD radius.
HEIGHT_SEARCH_RADIUS = 2
BACKGROUND_INNER_FACTOR = 1.0
BACKGROUND_OUTER_FACTOR = 1.5
MINIMUM_BACKGROUND_PIXELS = 10


# Converting pixels to angstroms.. roughly...
SCAN_WIDTH_M = 500e-9
IMAGE_WIDTH_PX = 512
PIXEL_SIZE_M = SCAN_WIDTH_M / IMAGE_WIDTH_PX
PIXEL_SIZE_A = PIXEL_SIZE_M * 1e10


def load_npy_image(path: str | Path) -> np.ndarray:
    """
    Load one already-converted AFM .npy image.

    The GUI expects a 2D array:
        shape = (height, width)

    It converts the data to float32 and replaces NaN/inf values.
    """
    path = Path(path)
    z = np.load(path)

    if z.ndim != 2:
        raise ValueError(f"Expected a 2D AFM image, got shape {z.shape}")

    z = np.asarray(z, dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    return z


def percentile_limits(z: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    """
    Return robust display limits for imshow.

    This affects only visual contrast, not the actual segmentation data.
    """
    vmin, vmax = np.percentile(z, [low, high])

    if vmax <= vmin:
        vmin = float(np.min(z))
        vmax = float(np.max(z))

    return float(vmin), float(vmax)


def measure_fwhm(
    peak_y: int,
    peak_x: int,
    peak_value: float,
    local_background: float,
    z_physical: np.ndarray,
    search_radius: int,
) -> dict[str, float] | None:
    """
    Measure QD full width at half maximum through the AFM apex.
    Half-height is defined relative to the local background: half_level = background + 0.5 * (peak - background)
    FWHM is measured independently along x and y using linear interpolation between pixels surrounding each half-height crossing.
    """

    z_physical = np.asarray(z_physical, dtype=float)
    rows, columns = z_physical.shape

    half_level = (local_background + 0.5 * (peak_value - local_background))

    def find_crossing(profile, peak_index, direction):
        """
        Move away from the peak until the profile crosses the half-height.
        Return the interpolated crossing position.
        """

        i = peak_index

        while True:
            j = i + direction

            if j < 0 or j >= len(profile):
                return None

            value_i = profile[i]
            value_j = profile[j]

            if not np.isfinite(value_i) or not np.isfinite(value_j):
                return None

            # Crossing from above half-height to below half-height.
            if value_i >= half_level and value_j < half_level:

                if value_j == value_i:
                    return float(i)
                fraction = (half_level - value_i) / (value_j - value_i)

                return float(i + fraction * (j - i))
            i = j

    # Horizontal profile through apex
    x0 = max(0, peak_x - search_radius)
    x1 = min(columns, peak_x + search_radius + 1)

    profile_x = z_physical[peak_y, x0:x1]
    peak_x_local = peak_x - x0

    left = find_crossing(profile_x, peak_x_local, direction=-1)
    right = find_crossing(profile_x, peak_x_local, direction=1)

    if left is not None and right is not None:
        fwhm_x = right - left
    else:
        fwhm_x = np.nan


    # Vertical profile through apex
    y0 = max(0, peak_y - search_radius)
    y1 = min(rows, peak_y + search_radius + 1)

    profile_y = z_physical[y0:y1, peak_x]
    peak_y_local = peak_y - y0

    top = find_crossing(profile_y, peak_y_local, direction=-1)
    bottom = find_crossing(profile_y, peak_y_local, direction=1)

    if top is not None and bottom is not None:
        fwhm_y = bottom - top
    else:
        fwhm_y = np.nan

    valid_widths = [width for width in (fwhm_x, fwhm_y) if np.isfinite(width)]

    if not valid_widths:
        return None

    fwhm_px = float(np.mean(valid_widths))

    return {
        "fwhm_x_px": float(fwhm_x),
        "fwhm_y_px": float(fwhm_y),
        "fwhm_px": fwhm_px,

        "fwhm_x_A": float(fwhm_x * PIXEL_SIZE_A),
        "fwhm_y_A": float(fwhm_y * PIXEL_SIZE_A),
        "fwhm_A": float(fwhm_px * PIXEL_SIZE_A),

        "half_maximum_m": float(half_level),
    }


def measure_local_height(
    candidate_y: float,
    candidate_x: float,
    qd_radius: float,
    z_physical: np.ndarray,
    search_radius: int = HEIGHT_SEARCH_RADIUS,
    background_inner_factor: float = BACKGROUND_INNER_FACTOR,
    background_outer_factor: float = BACKGROUND_OUTER_FACTOR,
    minimum_background_pixels: int = MINIMUM_BACKGROUND_PIXELS,
) -> dict[str, float] | None:
    """
    Relocate a candidate to the nearby AFM apex and measure its local
    height relative to a QD-size-dependent annular background.

    The background ring extends from:

        inner radius = background_inner_factor × QD radius
        outer radius = background_outer_factor × QD radius

    With the default values, the ring spans r to 1.5r.
    """
    z_physical = np.asarray(z_physical, dtype=np.float64)
    rows, columns = z_physical.shape

    candidate_y_i = int(round(candidate_y))
    candidate_x_i = int(round(candidate_x))

    # Search a small fixed neighbourhood for the actual AFM apex.
    y0 = max(0, candidate_y_i - search_radius)
    y1 = min(rows, candidate_y_i + search_radius + 1)
    x0 = max(0, candidate_x_i - search_radius)
    x1 = min(columns, candidate_x_i + search_radius + 1)

    search_patch = z_physical[y0:y1, x0:x1]

    if search_patch.size == 0 or not np.any(np.isfinite(search_patch)):
        return None

    patch_y, patch_x = np.unravel_index(
        np.nanargmax(search_patch),
        search_patch.shape,
    )

    peak_y = int(y0 + patch_y)
    peak_x = int(x0 + patch_x)
    peak_value = float(z_physical[peak_y, peak_x])

    # Scale the background ring to the measured QD radius.
    qd_radius = max(float(qd_radius), 1.0)
    background_inner_radius = background_inner_factor * qd_radius
    background_outer_radius = background_outer_factor * qd_radius

    # Guarantee at least a one-pixel-wide annulus.
    background_outer_radius = max(
        background_outer_radius,
        background_inner_radius + 1.0,
    )

    outer_radius_pixels = int(np.ceil(background_outer_radius))

    by0 = max(0, peak_y - outer_radius_pixels)
    by1 = min(rows, peak_y + outer_radius_pixels + 1)
    bx0 = max(0, peak_x - outer_radius_pixels)
    bx1 = min(columns, peak_x + outer_radius_pixels + 1)

    background_patch = z_physical[by0:by1, bx0:bx1]

    yy, xx = np.ogrid[by0:by1, bx0:bx1]
    distance_squared = (
        (yy - peak_y) ** 2
        + (xx - peak_x) ** 2
    )

    ring_mask = (
        (distance_squared >= background_inner_radius**2)
        & (distance_squared <= background_outer_radius**2)
        & np.isfinite(background_patch)
    )

    background_pixel_count = int(np.count_nonzero(ring_mask))

    if background_pixel_count < minimum_background_pixels:
        return None

    local_background = float(np.median(background_patch[ring_mask]))
    local_height = float(peak_value - local_background)

    fwhm_measurement = measure_fwhm(
        peak_y=peak_y,
        peak_x=peak_x,
        peak_value=peak_value,
        local_background=local_background,
        z_physical=z_physical,
        search_radius=max(int(np.ceil(3.0 * qd_radius)), 5),
    )

    if fwhm_measurement is None:
        fwhm_x_px = np.nan
        fwhm_y_px = np.nan
        fwhm_px = np.nan

        fwhm_x_A = np.nan
        fwhm_y_A = np.nan
        fwhm_A = np.nan

    else:
        fwhm_x_px = fwhm_measurement["fwhm_x_px"]
        fwhm_y_px = fwhm_measurement["fwhm_y_px"]
        fwhm_px = fwhm_measurement["fwhm_px"]

        fwhm_x_A = fwhm_measurement["fwhm_x_A"]
        fwhm_y_A = fwhm_measurement["fwhm_y_A"]
        fwhm_A = fwhm_measurement["fwhm_A"]

    return {
        "peak_y": float(peak_y),
        "peak_x": float(peak_x),

        "peak_value_m": peak_value,
        "local_background_m": local_background,

        "local_height_m": local_height,
        "local_height_nm": local_height * 1e9,
        "local_height_A": local_height * 1e10,

        "fwhm_x_px": fwhm_x_px,
        "fwhm_y_px": fwhm_y_px,
        "fwhm_px": fwhm_px,

        "fwhm_x_A": fwhm_x_A,
        "fwhm_y_A": fwhm_y_A,
        "fwhm_A": fwhm_A,

        "background_inner_radius_px": float(background_inner_radius),
        "background_outer_radius_px": float(background_outer_radius),
        "background_inner_radius_A": float(background_inner_radius * PIXEL_SIZE_A),
        "background_outer_radius_A": float(background_outer_radius * PIXEL_SIZE_A),
        "background_pixel_count": float(background_pixel_count),
    }

# ==========================================================
# Optional AFM preprocessing operations used by GUI buttons
# ==========================================================

def robust_mad(values: np.ndarray) -> float:
    """Median absolute deviation scaled like a standard deviation."""
    values = np.asarray(values, dtype=float)
    centre = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - centre)))


def subtract_global_plane(
    z: np.ndarray,
    n_iterations: int = 6,
    sigma: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly fit and subtract z = ax + by + c."""
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    yy, xx = np.indices(z.shape)
    finite = np.isfinite(z)

    if np.count_nonzero(finite) < 3:
        raise ValueError("Not enough finite pixels for plane fitting.")

    mask = finite.copy()
    design = np.column_stack((xx[finite], yy[finite], np.ones(np.count_nonzero(finite))))
    values = z[finite]
    coefficients = np.array([0.0, 0.0, np.nanmedian(values)])

    for _ in range(n_iterations):
        fit_mask = mask[finite]
        if np.count_nonzero(fit_mask) < 3:
            break

        coefficients, *_ = np.linalg.lstsq(
            design[fit_mask], values[fit_mask], rcond=None
        )
        plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
        residual = z - plane
        centre = np.nanmedian(residual[mask])
        scale = robust_mad(residual[mask])

        if not np.isfinite(scale) or scale == 0:
            break

        mask = finite & (np.abs(residual - centre) < sigma * scale)

    plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    flattened = z - plane
    flattened -= np.nanmedian(flattened)
    return flattened, plane, mask


def robust_polynomial_line_flatten(
    z: np.ndarray,
    degree: int = 1,
    n_iterations: int = 6,
    sigma: float = 2.5,
    axis: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly flatten each row (axis=1) or column (axis=0)."""
    z = np.asarray(z, dtype=float)
    if axis == 0:
        working = z.T.copy()
    elif axis == 1:
        working = z.copy()
    else:
        raise ValueError("axis must be 0 or 1")

    result = np.full_like(working, np.nan)
    backgrounds = np.full_like(working, np.nan)
    n_lines, n_points = working.shape
    x = np.linspace(-1.0, 1.0, n_points)

    for row_index in range(n_lines):
        line = working[row_index]
        finite = np.isfinite(line)
        if np.count_nonzero(finite) <= degree + 1:
            result[row_index] = line
            continue

        fit_mask = finite.copy()
        for _ in range(n_iterations):
            if np.count_nonzero(fit_mask) <= degree + 1:
                break
            coefficients = np.polyfit(x[fit_mask], line[fit_mask], degree)
            background = np.polyval(coefficients, x)
            residual = line - background
            centre = np.nanmedian(residual[fit_mask])
            scale = robust_mad(residual[fit_mask])
            if not np.isfinite(scale) or scale == 0:
                break
            new_mask = finite & (np.abs(residual - centre) < sigma * scale)
            if np.array_equal(new_mask, fit_mask):
                break
            fit_mask = new_mask

        if np.count_nonzero(fit_mask) <= degree + 1:
            fit_mask = finite
        coefficients = np.polyfit(x[fit_mask], line[fit_mask], degree)
        background = np.polyval(coefficients, x)
        result[row_index] = line - background
        backgrounds[row_index] = background

    if axis == 0:
        result, backgrounds = result.T, backgrounds.T
    result -= np.nanmedian(result)
    return result, backgrounds


def align_scanline_offsets(
    z: np.ndarray,
    smoothing_window: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove rapid row-to-row median offsets while preserving slow variation."""
    z = np.asarray(z, dtype=float)
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    row_offsets = np.nanmedian(z, axis=1)
    valid = np.isfinite(row_offsets)
    if not np.any(valid):
        raise ValueError("No finite scan-line offsets could be estimated.")

    filled = row_offsets.copy()
    if not np.all(valid):
        rows = np.arange(len(row_offsets))
        filled[~valid] = np.interp(rows[~valid], rows[valid], row_offsets[valid])

    smooth_offsets = ndimage.median_filter(
        filled, size=smoothing_window, mode="nearest"
    )
    rapid_offset_error = filled - smooth_offsets
    corrected = z - rapid_offset_error[:, None]
    corrected -= np.nanmedian(corrected)
    return corrected, rapid_offset_error


def robust_2d_background(
    image: np.ndarray,
    sigma_y: float = 30,
    sigma_x: float = 60,
    iterations: int = 8,
    threshold: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract a smooth 2D background while rejecting prominent features."""
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")

    filled = image.copy()
    filled[~finite] = np.nanmedian(image[finite])
    background = ndimage.gaussian_filter(
        filled, sigma=(sigma_y, sigma_x), mode="reflect"
    )
    mask = finite.copy()

    for _ in range(iterations):
        residual = image - background
        centre = np.nanmedian(residual[mask])
        scale = robust_mad(residual[mask])
        if not np.isfinite(scale) or scale == 0:
            break
        mask = finite & (np.abs(residual - centre) < threshold * scale)
        weighted_values = np.where(mask, image, 0.0)
        weights = mask.astype(float)
        smooth_values = ndimage.gaussian_filter(
            weighted_values, sigma=(sigma_y, sigma_x), mode="reflect"
        )
        smooth_weights = ndimage.gaussian_filter(
            weights, sigma=(sigma_y, sigma_x), mode="reflect"
        )
        background = smooth_values / np.maximum(smooth_weights, 1e-12)

    flattened = image - background
    flattened -= np.nanmedian(flattened)
    return flattened, background, mask


def remove_partial_horizontal_stripes(
    image: np.ndarray,
    sigma_x: float = 35,
    sigma_y_small: float = 1.0,
    sigma_y_large: float = 12.0,
    strength: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and subtract horizontally elongated stripe artefacts."""
    if sigma_y_large <= sigma_y_small:
        raise ValueError("sigma_y_large must exceed sigma_y_small.")

    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")

    filled = image.copy()
    filled[~finite] = np.nanmedian(image[finite])
    pre = ndimage.gaussian_filter(filled, sigma=(0.6, 0.6), mode="reflect")
    horiz_small_y = ndimage.gaussian_filter(
        pre, sigma=(sigma_y_small, sigma_x), mode="reflect"
    )
    horiz_large_y = ndimage.gaussian_filter(
        pre, sigma=(sigma_y_large, sigma_x), mode="reflect"
    )
    stripe_map = horiz_small_y - horiz_large_y
    corrected = filled - strength * stripe_map
    corrected -= np.nanmedian(corrected)
    corrected[~finite] = np.nan
    stripe_map[~finite] = np.nan
    return corrected, stripe_map


# ==========================================================
# Tkinter Application
# ==========================================================

class AFMSegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AFM Blob Segmentation & Ground Truth Tool - NPY")
        self.root.geometry("1450x900")

        self.filepath: str | None = None
        self.npy_files: list[Path] = []
        self.current_file_index: int | None = None

        self.raw_image: np.ndarray | None = None
        self.corrected_image: np.ndarray | None = None
        self.tophat_image: np.ndarray | None = None
        self.log_image: np.ndarray | None = None
        self.raw_mask: np.ndarray | None = None
        self.binary_mask: np.ndarray | None = None

        # Each feature stores its measurements and exact mask coordinates.
        self.features: list[dict] = []
        self.false_positives: list[dict] = []
        self.false_negatives: list[dict] = []

        self.edit_mode: str | None = None  # 'FP', 'FN', or None
        self.sliders_active = True
        self.show_labels = True
        self._needs_otsu = False
        self.preprocessing_history: list[str] = []

        # Cache trackers: used so that only changed pipeline stages rerun
        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None

        self._setup_ui()



    def _setup_ui(self) -> None:
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Load .NPY", command=self.load_file_dialog).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Previous", command=self.load_previous_file).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Next", command=self.load_next_file).pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_fp = tk.Button(toolbar, text="Mark False Positive", command=lambda: self.set_mode("FP"))
        self.btn_fp.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_fn = tk.Button(toolbar, text="Mark False Negative", command=lambda: self.set_mode("FN"))
        self.btn_fn.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(toolbar, text="Exit Edit Mode", command=self.clear_mode).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Toggle Labels", command=self.toggle_labels).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Export Mask & CSV", command=self.export_data, bg="lightblue").pack(side=tk.RIGHT, padx=5, pady=5)

        processing_toolbar = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        processing_toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(processing_toolbar, text="Preprocess:").pack(side=tk.LEFT, padx=(6, 3))
        tk.Button(processing_toolbar, text="Reset Raw", command=self.reset_preprocessing).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="Remove Plane", command=self.apply_global_plane).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="Line Flatten", command=self.apply_line_flatten).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="Align Rows", command=self.apply_row_alignment).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="2D Background", command=self.apply_2d_background).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="Horizontal Destripe", command=self.apply_horizontal_destripe).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(processing_toolbar, text="Standard Sequence", command=self.apply_standard_preprocessing, bg="lightyellow").pack(side=tk.LEFT, padx=8, pady=4)

        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)

        # Left control panel
        self.slider_frame = tk.Frame(main_pane, width=300)
        main_pane.add(self.slider_frame)

        self.status_label = tk.Label(self.slider_frame, text="No file loaded", wraplength=260, justify="left")
        self.status_label.pack(fill=tk.X, padx=10, pady=(10, 15))

        self.val_tophat = tk.IntVar(value=21)
        self.scale_tophat = self._create_slider_row(
            self.slider_frame, "Top-Hat Radius (px)", self.val_tophat, 5, 80, 1, 1
        )

        self.val_sigma = tk.DoubleVar(value=2.0)
        self.scale_sigma = self._create_slider_row(
            self.slider_frame, "LoG Sigma", self.val_sigma, 0.5, 10.0, 0.1, 0.1
        )

        self.val_thresh = tk.DoubleVar(value=0.2)
        self.scale_thresh = self._create_slider_row(
            self.slider_frame, "Binary Threshold (Norm 0-1)", self.val_thresh, 0.001, 1.0, 0.001, 0.001
        )

        self.val_circ = tk.DoubleVar(value=0.5)
        self.scale_circ = self._create_slider_row(
            self.slider_frame, "Circularity Cut-off", self.val_circ, 0.0, 1.0, 0.05, 0.05
        )

        self.val_area_upper = tk.DoubleVar(value=500.0)
        self.scale_area_upper = self._create_slider_row(
            self.slider_frame, "Max Area (px²)", self.val_area_upper, 5.0, 3000.0, 1.0, 5.0
        )

        self.val_area_lower = tk.DoubleVar(value=5.0)
        self.scale_area_lower = self._create_slider_row(
            self.slider_frame, "Min Area (px²)", self.val_area_lower, 1.0, 300.0, 1.0, 1.0
        )

        # Right plot panel
        plot_frame = tk.Frame(main_pane)
        main_pane.add(plot_frame)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.fig.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_canvas_hover)
        self.fig.tight_layout()

        self.update_empty_plots()

    def _create_slider_row(
        self,
        parent: tk.Widget,
        label_text: str,
        var: tk.Variable,
        from_: float,
        to: float,
        resolution: float,
        step: float,
    ) -> tk.Scale:
        container = tk.Frame(parent)
        container.pack(fill=tk.X, pady=(10, 0), padx=10)

        tk.Label(container, text=label_text).pack(side=tk.TOP)

        ctrl_frame = tk.Frame(container)
        ctrl_frame.pack(fill=tk.X)

        value_label = tk.Label(container, text=str(var.get()))
        value_label.pack(side=tk.TOP)

        def on_change(*_args):
            value_label.configure(text=str(var.get()))
            self.run_pipeline()

        def decrease():
            if not self.sliders_active:
                return
            new_val = round(float(var.get()) - step, 5)
            if new_val >= from_:
                var.set(new_val)
                on_change()

        def increase():
            if not self.sliders_active:
                return
            new_val = round(float(var.get()) + step, 5)
            if new_val <= to:
                var.set(new_val)
                on_change()

        tk.Button(ctrl_frame, text="-", command=decrease, width=2).pack(side=tk.LEFT)
        scale = tk.Scale(
            ctrl_frame,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda _value: on_change(),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(ctrl_frame, text="+", command=increase, width=2).pack(side=tk.RIGHT)

        return scale


    def load_previous_file(self) -> None:
        if not self.npy_files:
            return

        if self.current_file_index is None:
            idx = 0
        else:
            idx = max(0, self.current_file_index - 1)

        self.load_npy_by_index(idx)

    def load_next_file(self) -> None:
        if not self.npy_files:
            return

        if self.current_file_index is None:
            idx = 0
        else:
            idx = min(len(self.npy_files) - 1, self.current_file_index + 1)

        self.load_npy_by_index(idx)

    def load_file_dialog(self) -> None:
        initial_dir = DEFAULT_NPY_DIR if DEFAULT_NPY_DIR.exists() else PROJECT_DIR

        path = filedialog.askopenfilename(
            title="Choose converted AFM .npy file",
            initialdir=str(initial_dir),
            filetypes=[("NumPy AFM arrays", "*.npy")],
        )

        if not path:
            return

        path = Path(path)

        # If selected file is already in the scanned list, use its index.
        if path in self.npy_files:
            idx = self.npy_files.index(path)
            self.load_npy_by_index(idx)
        else:
            self.load_npy_file(path)

    def load_npy_by_index(self, idx: int) -> None:
        self.current_file_index = idx
        self.load_npy_file(self.npy_files[idx])

    def load_npy_file(self, path: str | Path) -> None:
        path = Path(path)
        print("\n=== Loading converted AFM array ===", flush=True)
        print("Path:", path, flush=True)

        try:
            z = load_npy_image(path)

            self.filepath = str(path)
            self.raw_image = z.copy()
            self.corrected_image = z.copy()
            self.preprocessing_history = []

            print("Loaded .npy array", flush=True)
            print("Shape:", z.shape, flush=True)
            print("Dtype:", z.dtype, flush=True)
            print("Min/max:", float(np.min(z)), float(np.max(z)), flush=True)

            self.reset_pipeline_state()
            self._needs_otsu = True
            self.run_pipeline()

        except Exception:
            print("LOAD FAILED:", flush=True)
            traceback.print_exc()
            self.status_label.configure(text="Load failed. Check terminal.")

    def reset_pipeline_state(self) -> None:
        self.tophat_image = None
        self.log_image = None
        self.raw_mask = None
        self.binary_mask = None
        self.features = []
        self.false_positives = []
        self.false_negatives = []

        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None



    def _finish_preprocessing(self, image: np.ndarray, operation_name: str) -> None:
        """Store a processed image and rerun the detection pipeline."""
        image = np.asarray(image, dtype=np.float32)
        if image.shape != self.raw_image.shape:
            raise ValueError("Preprocessing changed the image shape.")
        if not np.any(np.isfinite(image)):
            raise ValueError("Preprocessing produced no finite pixels.")

        self.corrected_image = image
        self.preprocessing_history.append(operation_name)
        self.reset_pipeline_state()
        self._needs_otsu = True
        self.run_pipeline()
        print(f"Applied preprocessing: {operation_name}", flush=True)

    def reset_preprocessing(self) -> None:
        if self.raw_image is None:
            return
        self.corrected_image = self.raw_image.copy()
        self.preprocessing_history = []
        self.reset_pipeline_state()
        self._needs_otsu = True
        self.run_pipeline()
        print("Reset to raw image.", flush=True)

    def apply_global_plane(self) -> None:
        if self.corrected_image is None:
            return
        try:
            corrected, _plane, _mask = subtract_global_plane(self.corrected_image)
            self._finish_preprocessing(corrected, "global plane")
        except Exception:
            traceback.print_exc()

    def apply_line_flatten(self) -> None:
        if self.corrected_image is None:
            return
        try:
            corrected, _background = robust_polynomial_line_flatten(
                self.corrected_image, degree=1, n_iterations=6, sigma=2.5, axis=1
            )
            self._finish_preprocessing(corrected, "row polynomial flatten")
        except Exception:
            traceback.print_exc()

    def apply_row_alignment(self) -> None:
        if self.corrected_image is None:
            return
        try:
            corrected, _offsets = align_scanline_offsets(
                self.corrected_image, smoothing_window=21
            )
            self._finish_preprocessing(corrected, "row-offset alignment")
        except Exception:
            traceback.print_exc()

    def apply_2d_background(self) -> None:
        if self.corrected_image is None:
            return
        try:
            corrected, _background, _mask = robust_2d_background(
                self.corrected_image, sigma_y=30, sigma_x=60, iterations=8, threshold=2.5
            )
            self._finish_preprocessing(corrected, "robust 2D background")
        except Exception:
            traceback.print_exc()

    def apply_horizontal_destripe(self) -> None:
        if self.corrected_image is None:
            return
        try:
            corrected, _stripe_map = remove_partial_horizontal_stripes(
                self.corrected_image, sigma_x=35, sigma_y_small=1.0,
                sigma_y_large=12.0, strength=0.75
            )
            self._finish_preprocessing(corrected, "horizontal destripe")
        except Exception:
            traceback.print_exc()

    def apply_standard_preprocessing(self) -> None:
        """Apply a conservative plane → 2D background → destripe sequence."""
        if self.raw_image is None:
            return
        try:
            image, _plane, _plane_mask = subtract_global_plane(
                self.raw_image, n_iterations=6, sigma=2.5
            )
            image, _background, _background_mask = robust_2d_background(
                image, sigma_y=30, sigma_x=60, iterations=8, threshold=2.5
            )
            image, _stripe_map = remove_partial_horizontal_stripes(
                image, sigma_x=35, sigma_y_small=1.0,
                sigma_y_large=12.0, strength=0.75
            )
            self.corrected_image = np.asarray(image, dtype=np.float32)
            self.preprocessing_history = [
                "global plane", "robust 2D background", "horizontal destripe"
            ]
            self.reset_pipeline_state()
            self._needs_otsu = True
            self.run_pipeline()
            print("Applied standard preprocessing sequence.", flush=True)
        except Exception:
            traceback.print_exc()

    def run_pipeline(self, *args) -> None:
        if self.corrected_image is None or not self.sliders_active:
            return

        r_th = int(self.val_tophat.get())
        sigma = float(self.val_sigma.get())
        thresh_val = float(self.val_thresh.get())
        circ_cutoff = float(self.val_circ.get())
        area_upper = float(self.val_area_upper.get())
        area_lower = float(self.val_area_lower.get())

        run_tophat = (self.last_r_th != r_th) or self.tophat_image is None
        run_log = run_tophat or (self.last_sigma != sigma) or self.log_image is None
        run_thresh = run_log or (self.last_thresh != thresh_val) or self._needs_otsu or self.raw_mask is None
        run_geom = (
            run_thresh
            or (self.last_circ != circ_cutoff)
            or (self.last_area_upper != area_upper)
            or (self.last_area_lower != area_lower)
        )

        self.last_r_th = r_th
        self.last_sigma = sigma
        self.last_thresh = thresh_val
        self.last_circ = circ_cutoff
        self.last_area_upper = area_upper
        self.last_area_lower = area_lower

        # 1. White top-hat: extracts bright features smaller than the disk radius.
        if run_tophat:
            selem = disk(r_th)
            self.tophat_image = white_tophat(self.corrected_image, footprint=selem)

        # 2. Laplacian-of-Gaussian response.
        if run_log:
            log_response = -ndimage.gaussian_laplace(self.tophat_image, sigma=sigma)
            log_min = float(np.min(log_response))
            log_max = float(np.max(log_response))

            if log_max > log_min:
                self.log_image = (log_response - log_min) / (log_max - log_min)
            else:
                self.log_image = np.zeros_like(log_response, dtype=np.float32)

            if self._needs_otsu:
                self._needs_otsu = False
                try:
                    otsu_val = float(threshold_otsu(self.log_image))
                    thresh_val = round(otsu_val, 3)
                    self.val_thresh.set(thresh_val)
                    self.last_thresh = thresh_val
                    print(f"Auto Otsu threshold: {thresh_val}", flush=True)
                except Exception as e:
                    print(f"Otsu threshold failed: {e}", flush=True)

        # 3. Binary threshold of LoG response.
        if run_thresh:
            self.raw_mask = self.log_image > thresh_val

        # 4. Connected-component analysis + geometric filtering.
        if run_geom:
            labeled_mask = label(self.raw_mask)
            regions = regionprops(labeled_mask)

            self.features = []
            self.binary_mask = np.zeros_like(self.raw_mask, dtype=np.uint8)

            for prop in regions:
                perimeter = float(prop.perimeter)
                if perimeter <= 0:
                    circularity = 0.0
                else:
                    circularity = float((4 * np.pi * prop.area) / (perimeter**2))

                if circularity >= circ_cutoff and area_lower <= prop.area <= area_upper:
                    candidate_y, candidate_x = prop.centroid
                    radius = float(prop.equivalent_diameter_area / 2.0)

                    measurement = measure_local_height(
                        candidate_y=candidate_y,
                        candidate_x=candidate_x,
                        qd_radius=radius,
                        z_physical=self.corrected_image,
                    )

                    # A segmented feature is accepted as a QD only when
                    # its locally measured height is at least 6 Å.
                    if (
                        measurement is None
                        or measurement["local_height_m"] < MINIMUM_QD_HEIGHT_M
                    ):
                        continue

                    feature = {
                        "cx": measurement["peak_x"],
                        "cy": measurement["peak_y"],
                        "r": radius,
                        "r_A": radius * PIXEL_SIZE_A,
                        "area": float(prop.area),
                        "area_A2": float(prop.area) * PIXEL_SIZE_A**2,
                        "circularity": circularity,
                        "peak_value_m": measurement["peak_value_m"],
                        "local_background_m": measurement["local_background_m"],
                        "local_height_m": measurement["local_height_m"],
                        "local_height_nm": measurement["local_height_nm"],
                        "local_height_A": measurement["local_height_A"],
                        "background_inner_radius_px": measurement["background_inner_radius_px"],
                        "background_outer_radius_px": measurement["background_outer_radius_px"],
                        "background_inner_radius_A": measurement["background_inner_radius_A"],
                        "background_outer_radius_A": measurement["background_outer_radius_A"],
                        "background_pixel_count": measurement["background_pixel_count"],
                        "manual_status": "automatic",
                        "mask_coords": prop.coords.copy(),
                        "fwhm_x_px": measurement["fwhm_x_px"],
                        "fwhm_y_px": measurement["fwhm_y_px"],
                        "fwhm_px": measurement["fwhm_px"],
                        "fwhm_x_A": measurement["fwhm_x_A"],
                        "fwhm_y_A": measurement["fwhm_y_A"],
                        "fwhm_A": measurement["fwhm_A"],
                        "manual_status": "automatic",
                        "mask_coords": prop.coords.copy(),
                    }

                    self.features.append(feature)
                    self.binary_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

        self.update_plots()
        



    def update_empty_plots(self) -> None:
        for ax in self.axs.ravel():
            ax.clear()
            ax.axis("off")
            ax.text(0.5, 0.5, "Load a .npy file", ha="center", va="center", transform=ax.transAxes)

        self.canvas.draw()

    def update_plots(self) -> None:
        if self.corrected_image is None:
            self.update_empty_plots()
            return

        for ax in self.axs.ravel():
            ax.clear()
            ax.axis("off")

        z_vmin, z_vmax = percentile_limits(self.corrected_image, 1, 99)

        # Plot 1: original working AFM image + circles
        self.axs[0].imshow(self.corrected_image, cmap="afmhot", vmin=z_vmin, vmax=z_vmax)
        self.axs[0].set_title(f"AFM image | QDs ≥ 6 Å: {len(self.features)}")

        if self.show_labels:
            for f in self.features:
                circle = plt.Circle((f["cx"], f["cy"]), f["r"] + 2, color="cyan", fill=False, lw=1.0)
                self.axs[0].add_patch(circle)

        """

        I FELT LIKE WE DIDN'T NEED TO SEE THE TOP-HAT AND LOG RESPONSE PLOTS, SO I COMMENTED THEM OUT.....

        # Plot 2: top-hat image
        if self.tophat_image is not None:
            th_vmin, th_vmax = percentile_limits(self.tophat_image, 1, 99.7)
            self.axs[0, 1].imshow(self.tophat_image, cmap="gray", vmin=th_vmin, vmax=th_vmax)
        self.axs[0, 1].set_title("Top-hat")

        # Plot 3: normalized LoG response
        if self.log_image is not None:
            self.axs[1, 0].imshow(self.log_image, cmap="gray", vmin=0, vmax=1)
        self.axs[1, 0].set_title("Top-hat + LoG response")
        """

        # Plot 4: final binary mask
        if self.binary_mask is not None:
            self.axs[1].imshow(self.binary_mask, cmap="gray", vmin=0, vmax=1)
        self.axs[1].set_title("Final binary mask")

        if self.filepath:
            file_label = Path(self.filepath).name
        else:
            file_label = "No file"

        self.status_label.configure(
            text=(
                f"File: {file_label}\n"
                f"Shape: {self.corrected_image.shape}\n"
                f"QDs ≥ 6 Å: {len(self.features)}\n"
                f"Minimum local height: {MINIMUM_QD_HEIGHT_M * 1e10:.1f} Å\n"
                f"Mode: {self.edit_mode or 'normal'}"
            )
        )

        self.hover_annotations = {}

        for ax in (self.axs[0], self.axs[1]):
            annotation = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=4,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
                arrowprops={"arrowstyle": "->"},
                visible=False,
            )
            self.hover_annotations[ax] = annotation

        self.fig.tight_layout()

        # Remove the previous floating QD profile axes if it exists.
        if hasattr(self, "profile_ax"):
            try:
                self.profile_ax.remove()
            except Exception:
                pass


        # Floating QD profile graph.
        # [left, bottom, width, height] values are fractions of the entire Matplotlib canvas.
        self.profile_ax = self.fig.add_axes(
            [0.37, 0.58, 0.26, 0.28],
            zorder=20,
        )

        self.profile_ax.set_visible(False)

        # Tracks which QD is currently being hovered.
        self._hovered_feature_index = None

        self.canvas.draw_idle()

    # ------------------------------------------------------
    # Manual editing
    # ------------------------------------------------------

    def toggle_labels(self) -> None:
        self.show_labels = not self.show_labels
        self.update_plots()

    def set_mode(self, mode: str) -> None:
        self.edit_mode = mode
        self.sliders_active = False
        self.set_slider_state("disabled")

        self.btn_fp.configure(relief=tk.SUNKEN if mode == "FP" else tk.RAISED)
        self.btn_fn.configure(relief=tk.SUNKEN if mode == "FN" else tk.RAISED)

        print(f"Mode set to {mode}. Sliders locked.", flush=True)
        self.update_plots()

    def clear_mode(self) -> None:
        self.edit_mode = None
        self.sliders_active = True
        self.set_slider_state("normal")

        self.btn_fp.configure(relief=tk.RAISED)
        self.btn_fn.configure(relief=tk.RAISED)

        print("Edit mode cleared. Sliders unlocked.", flush=True)
        self.update_plots()

    def set_slider_state(self, state: str) -> None:
        for scale in [
            self.scale_tophat,
            self.scale_sigma,
            self.scale_thresh,
            self.scale_circ,
            self.scale_area_upper,
            self.scale_area_lower,
        ]:
            scale.configure(state=state)

    def on_canvas_click(self, event) -> None:
        if self.binary_mask is None or self.corrected_image is None:
            return

        if not self.edit_mode or event.inaxes not in [self.axs[0], self.axs[1]]:
            return

        if event.xdata is None or event.ydata is None:
            return

        click_x = float(event.xdata)
        click_y = float(event.ydata)

        if self.edit_mode == "FP":
            self.remove_nearest_feature(click_x, click_y)
        elif self.edit_mode == "FN":
            self.add_manual_feature(click_x, click_y)

        self.update_plots()

    def get_qd_axis_profiles(self, feature: dict, radius_factor: float = 3.0) -> dict[str, np.ndarray] | None:
        """
        Extract horizontal and vertical AFM height profiles through
        the measured QD apex.

        Horizontal: z(y_peak, x)
        Vertical: z(y, x_peak)

        Lateral distance is returned in Å.
        Height is returned relative to the QD local background in Å.
        """

        if self.corrected_image is None:
            return None

        z = np.asarray(self.corrected_image, dtype=float)

        rows, columns = z.shape

        peak_x = int(round(feature["cx"]))
        peak_y = int(round(feature["cy"]))
        radius_px = float(feature["r"])
        profile_radius_px = max(int(np.ceil(radius_factor * radius_px)), 5)

        # Horizontal profile through QD apex
        x0 = max(0, peak_x - profile_radius_px)
        x1 = min(columns, peak_x + profile_radius_px + 1)

        horizontal_height_m = z[peak_y, x0:x1]
        horizontal_pixel_positions = (np.arange(x0, x1) - peak_x)
        horizontal_distance_A = (horizontal_pixel_positions*PIXEL_SIZE_A)

        # Vertical profile through QD apex
        y0 = max(0, peak_y - profile_radius_px)
        y1 = min(rows, peak_y + profile_radius_px + 1)

        vertical_height_m = z[y0:y1, peak_x]
        vertical_pixel_positions = np.arange(y0, y1) - peak_y
        vertical_distance_A = vertical_pixel_positions * PIXEL_SIZE_A

        # Remove local background
        background_m = feature["local_background_m"]

        horizontal_height_A = (horizontal_height_m - background_m) * 1e10
        vertical_height_A = (vertical_height_m - background_m) * 1e10

        return {
            "horizontal_distance_A": horizontal_distance_A,
            "horizontal_height_A": horizontal_height_A,
            "vertical_distance_A": vertical_distance_A,
            "vertical_height_A": vertical_height_A,
        }

    def update_qd_profile_plot(self, feature: dict) -> None:
        """
        Update the floating profile plot for the currently hovered QD.
        """

        profiles = self.get_qd_axis_profiles(feature)

        if profiles is None:
            self.profile_ax.set_visible(False)
            return

        ax = self.profile_ax

        ax.clear()
        ax.set_visible(True)

        # Horizontal centre-line profile
        ax.plot(profiles["horizontal_distance_A"], profiles["horizontal_height_A"], label="Horizontal", linewidth=1.5,)
        # Vertical centre-line profile
        ax.plot(profiles["vertical_distance_A"], profiles["vertical_height_A"], label="Vertical", linewidth=1.5, linestyle="--",)

        # Local background
        ax.axhline(0, linewidth=0.8, linestyle=":", color="gray")

        # Half maximum
        half_height_A = (feature["local_height_A"] / 2.0)
        ax.axhline(half_height_A, linewidth=0.8, linestyle=":",)

        # QD centre
        ax.axvline(0, linewidth=0.8, linestyle=":",)

        ax.set_xlabel("Distance from apex (Å)", fontsize=4)

        ax.set_ylabel("Height above background (Å)", fontsize=4)
        ax.set_title("QD centre profiles", fontsize=4)

        ax.tick_params(axis="both", labelsize=3)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.2)

    def on_canvas_hover(self, event) -> None:
        if not hasattr(self, "hover_annotations"):
            return

        valid_axes = (self.axs[0], self.axs[1])

        if event.inaxes not in valid_axes or event.xdata is None or event.ydata is None:

            changed = False

            for annotation in self.hover_annotations.values():
                if annotation.get_visible():
                    annotation.set_visible(False)
                    changed = True

            if hasattr(self, "profile_ax"):
                if self.profile_ax.get_visible():
                    self.profile_ax.set_visible(False)
                    changed = True

            self._hovered_feature_index = None

            if changed:
                self.canvas.draw_idle()

            return

        if not self.features:
            return

        mouse_x = float(event.xdata)
        mouse_y = float(event.ydata)

        distances = [
            np.hypot(feature["cx"] - mouse_x, feature["cy"] - mouse_y)
            for feature in self.features
        ]

        nearest_index = int(np.argmin(distances))
        nearest_feature = self.features[nearest_index]
        nearest_distance = distances[nearest_index]

        hover_distance = nearest_feature["r"] + 3

        for ax, annotation in self.hover_annotations.items():
            annotation.set_visible(False)

        if nearest_distance > hover_distance:

            if nearest_distance > hover_distance:
                if hasattr(self, "profile_ax"):
                    self.profile_ax.set_visible(False)

                self._hovered_feature_index = None

                self.canvas.draw_idle()
                return

        # Only redraw the profile when we move onto a different QD.
        if self._hovered_feature_index != nearest_index:

            self.update_qd_profile_plot(
                nearest_feature
            )

            self._hovered_feature_index = nearest_index
            
        annotation = self.hover_annotations[event.inaxes]
        annotation.xy = (nearest_feature["cx"], nearest_feature["cy"])

        annotation.set_text(
            f"Height: {nearest_feature['local_height_A']:.2f} Å\n"
            f"FWHM: {nearest_feature['fwhm_A']:.2f} Å\n"
            f"FWHM X: {nearest_feature['fwhm_x_A']:.2f} Å\n"
            f"FWHM Y: {nearest_feature['fwhm_y_A']:.2f} Å\n"
            f"Radius: {nearest_feature['r_A']:.2f} Å\n"
            f"Area: {nearest_feature['area_A2']:.1f} Å²\n"
            f"Background ring: "
            f"{nearest_feature['background_inner_radius_A']:.1f}–"
            f"{nearest_feature['background_outer_radius_A']:.1f} Å"
        )

        annotation.set_visible(True)
        self.canvas.draw_idle()

    def rebuild_binary_mask(self) -> None:
        """Recreate the final mask exactly from the features currently retained."""
        if self.raw_mask is None:
            self.binary_mask = None
            return

        self.binary_mask = np.zeros_like(self.raw_mask, dtype=np.uint8)

        for feature in self.features:
            coordinates = np.asarray(feature.get("mask_coords", []), dtype=int).reshape(-1, 2)
            if len(coordinates) == 0:
                continue
            self.binary_mask[coordinates[:, 0], coordinates[:, 1]] = 1

    def remove_nearest_feature(self, x: float, y: float) -> None:
        if not self.features:
            return

        distances = [(idx, np.hypot(feature["cx"] - x, feature["cy"] - y)) for idx, feature in enumerate(self.features)]
        nearest_idx, distance = min(distances, key=lambda item: item[1])
        nearest_feature = self.features[nearest_idx]

        if distance > nearest_feature["r"] + 5:
            return

        removed = self.features.pop(nearest_idx)
        removed["manual_status"] = "false_positive"
        self.false_positives.append(removed.copy())
        self.rebuild_binary_mask()

    def add_manual_feature(self, x: float, y: float) -> None:
        """Add a manually selected false negative to the corrected ground truth."""
        if self.features:
            radius = float(np.median([feature["r"] for feature in self.features]))
        else:
            radius = 5.0

        measurement = measure_local_height(
            candidate_y=y,
            candidate_x=x,
            qd_radius=radius,
            z_physical=self.corrected_image,
        )

        if measurement is None:
            print("Manual candidate rejected: local height could not be measured.", flush=True)
            return

        rr, cc = draw_disk(
            (measurement["peak_y"], measurement["peak_x"]),
            radius,
            shape=self.binary_mask.shape,
        )
        mask_coords = np.column_stack((rr, cc))

        feature = {
            "cx": measurement["peak_x"],
            "cy": measurement["peak_y"],
            "r": radius,
            "r_A": radius * PIXEL_SIZE_A,
            "area": float(len(mask_coords)),
            "area_A2": float(len(mask_coords)) * PIXEL_SIZE_A**2,
            "circularity": 1.0,
            "peak_value_m": measurement["peak_value_m"],
            "local_background_m": measurement["local_background_m"],
            "local_height_m": measurement["local_height_m"],
            "local_height_nm": measurement["local_height_nm"],
            "local_height_A": measurement["local_height_A"],
            "fwhm_x_px": measurement["fwhm_x_px"],
            "fwhm_y_px": measurement["fwhm_y_px"],
            "fwhm_px": measurement["fwhm_px"],
            "fwhm_x_A": measurement["fwhm_x_A"],
            "fwhm_y_A": measurement["fwhm_y_A"],
            "fwhm_A": measurement["fwhm_A"],
            "background_inner_radius_A": measurement["background_inner_radius_A"],
            "background_outer_radius_A": measurement["background_outer_radius_A"],
            "manual_status": "false_negative",
            "mask_coords": mask_coords,
        }

        self.features.append(feature)
        self.false_negatives.append(feature.copy())
        self.rebuild_binary_mask()

        print(f"Manual QD: {measurement['local_height_A']:.2f} Å.", flush=True)

    @staticmethod
    def features_to_dataframe(features: list[dict]) -> pd.DataFrame:
        """Convert feature dictionaries to a CSV-safe table."""
        rows = [{key: value for key, value in feature.items() if key != "mask_coords"} for feature in features]
        return pd.DataFrame(rows)

    def export_data(self) -> None:
        if self.binary_mask is None or self.filepath is None:
            print("No data to export.", flush=True)
            return

        base_name = Path(self.filepath).stem
        save_path_base = filedialog.asksaveasfilename(
            defaultextension="",
            initialfile=base_name,
            title="Save Base Name; creates _mask.npy and _features.csv",
        )

        if not save_path_base:
            return

        if save_path_base.endswith(".npy") or save_path_base.endswith(".csv"):
            save_path_base = save_path_base.rsplit(".", 1)[0]

        np.save(f"{save_path_base}_mask.npy", self.binary_mask)
        self.features_to_dataframe(self.features).to_csv(f"{save_path_base}_features.csv", index=False)
        self.features_to_dataframe(self.false_positives).to_csv(f"{save_path_base}_false_positives.csv", index=False)
        self.features_to_dataframe(self.false_negatives).to_csv(f"{save_path_base}_false_negatives.csv", index=False)

        print("Successfully exported:", flush=True)
        print(f"  {save_path_base}_mask.npy", flush=True)
        print(f"  {save_path_base}_features.csv", flush=True)
        print(f"  {save_path_base}_false_positives.csv", flush=True)
        print(f"  {save_path_base}_false_negatives.csv", flush=True)




if __name__ == "__main__":
    root = tk.Tk()
    app = AFMSegmentationApp(root)
    root.mainloop()
