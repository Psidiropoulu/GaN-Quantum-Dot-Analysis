from __future__ import annotations

import os
import traceback
from pathlib import Path
from dataclasses import dataclass, field

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
DEFAULT_NPY_DIR = PROJECT_DIR / "data_numpy"


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



# ==========================================================
# Preprocessing functions
# ==========================================================

def mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled like a standard deviation."""
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
    design = np.column_stack(
        (xx[finite], yy[finite], np.ones(np.count_nonzero(finite)))
    )
    values = z[finite]
    coefficients = np.zeros(3, dtype=float)

    for _ in range(n_iterations):
        fit_mask = mask[finite]
        if np.count_nonzero(fit_mask) < 3:
            break

        coefficients, *_ = np.linalg.lstsq(
            design[fit_mask], values[fit_mask], rcond=None
        )
        plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
        residual = z - plane
        scale = mad(residual[mask])

        if not np.isfinite(scale) or scale == 0:
            break

        centre = np.nanmedian(residual[mask])
        mask = finite & (np.abs(residual - centre) < sigma * scale)

    plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    return z - plane, plane, mask


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
        coefficients = np.zeros(degree + 1, dtype=float)

        for _ in range(n_iterations):
            if np.count_nonzero(fit_mask) <= degree + 1:
                break

            coefficients = np.polyfit(x[fit_mask], line[fit_mask], degree)
            background = np.polyval(coefficients, x)
            residual = line - background
            centre = np.nanmedian(residual[fit_mask])
            scale = mad(residual[fit_mask])

            if not np.isfinite(scale) or scale == 0:
                break

            fit_mask = finite & (np.abs(residual - centre) < sigma * scale)

        if np.count_nonzero(fit_mask) <= degree + 1:
            fit_mask = finite

        coefficients = np.polyfit(x[fit_mask], line[fit_mask], degree)
        background = np.polyval(coefficients, x)
        result[row_index] = line - background
        backgrounds[row_index] = background

    if axis == 0:
        return result.T, backgrounds.T
    return result, backgrounds


def align_scanline_offsets(
    z: np.ndarray,
    smoothing_window: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove rapid row-to-row offsets while preserving slow variation."""
    z = np.asarray(z, dtype=float)

    if smoothing_window % 2 == 0:
        smoothing_window += 1

    row_offsets = np.nanmedian(z, axis=1)
    valid = np.isfinite(row_offsets)
    filled = row_offsets.copy()

    if not np.any(valid):
        return z.copy(), np.zeros(z.shape[0], dtype=float)

    if not np.all(valid):
        row_numbers = np.arange(len(row_offsets))
        filled[~valid] = np.interp(
            row_numbers[~valid], row_numbers[valid], row_offsets[valid]
        )

    smooth_offsets = ndimage.median_filter(
        filled, size=smoothing_window, mode="nearest"
    )
    rapid_offset_error = filled - smooth_offsets
    corrected = z - rapid_offset_error[:, None]
    return corrected, rapid_offset_error


def robust_2d_background(
    image: np.ndarray,
    sigma_y: float = 30,
    sigma_x: float = 60,
    iterations: int = 8,
    threshold: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and subtract a smooth 2D background robustly."""
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
        scale = mad(residual[mask])

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

    return image - background, background, mask


def remove_partial_horizontal_stripes(
    image: np.ndarray,
    sigma_x: float = 35,
    sigma_y_small: float = 1.0,
    sigma_y_large: float = 12.0,
    strength: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and partially subtract horizontally elongated stripe artefacts."""
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
    corrected[~finite] = np.nan
    stripe_map[~finite] = np.nan
    return corrected, stripe_map


# ==========================================================
# Generic algorithm placeholders
# ==========================================================

def algorithm_1(app: "QDAnalysisApp") -> None:
    """Placeholder algorithm 1. Replace this body with your own algorithm."""
    print("Running Algorithm 1 (current top-hat + LoG pipeline).", flush=True)
    app.reset_detection_cache()
    app.run_pipeline()


def algorithm_2(app: "QDAnalysisApp") -> None:
    """Placeholder algorithm 2. Replace this body with your own algorithm."""
    print("Running Algorithm 2 placeholder.", flush=True)
    app.reset_detection_cache()
    app.run_pipeline()


def algorithm_3(app: "QDAnalysisApp") -> None:
    """Placeholder algorithm 3. Replace this body with your own algorithm."""
    print("Running Algorithm 3 placeholder.", flush=True)
    app.reset_detection_cache()
    app.run_pipeline()


ALGORITHM_FUNCTIONS = {
    "Algorithm 1": algorithm_1,
    "Algorithm 2": algorithm_2,
    "Algorithm 3": algorithm_3,
}


# ==========================================================
# Blob model
# ==========================================================

@dataclass
class Blob:
    """Represents one detected quantum dot."""

    cx: float
    cy: float
    radius: float
    area: float
    circularity: float

    dot_height: float | None = None
    background_height: float | None = None
    relative_height: float | None = None
    notes: str = ""

    def contains(self, x: float, y: float, padding: float = 2.0) -> bool:
        """Return True when image coordinate (x, y) lies inside the blob."""
        distance = float(np.hypot(self.cx - x, self.cy - y))
        return distance <= self.radius + padding

    def calculate_height_parameters(
        self,
        image: np.ndarray,
        background_inner_scale: float = 1.5,
        background_outer_scale: float = 2.5,
    ) -> None:
        rows, cols = np.indices(image.shape)
        distance_squared = (cols - self.cx) ** 2 + (rows - self.cy) ** 2

        dot_mask = distance_squared <= self.radius**2
        inner_radius = background_inner_scale * self.radius
        outer_radius = background_outer_scale * self.radius
        background_mask = (
            (distance_squared >= inner_radius**2)
            & (distance_squared <= outer_radius**2)
            & np.isfinite(image)
        )

        dot_pixels = image[dot_mask & np.isfinite(image)]
        background_pixels = image[background_mask]

        if dot_pixels.size == 0 or background_pixels.size == 0:
            self.dot_height = np.nan
            self.background_height = np.nan
            self.relative_height = np.nan
            return

        self.dot_height = float(np.max(dot_pixels))
        self.background_height = float(np.median(background_pixels))
        self.relative_height = self.dot_height - self.background_height

    def to_dict(self) -> dict[str, object]:
        """Return all blob data in a CSV-friendly dictionary."""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "r": self.radius,
            "area": self.area,
            "circularity": self.circularity,
            "blob.dot_height": self.dot_height,
            "blob.background_height": self.background_height,
            "blob.relative_height": self.relative_height,
            "notes": self.notes,
        }


# ==========================================================
# Tkinter Application
# ==========================================================

class QDAnalysisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("QDAnalysisApp")
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

        # Each detected quantum dot is represented by a Blob object.
        self.features: list[Blob] = []

        self.edit_mode: str | None = None  # 'FP', 'FN', or None
        self.sliders_active = True
        self.show_labels = True
        self._needs_otsu = False
        self.preprocessing_history: list[str] = []
        self.selected_algorithm = tk.StringVar(value="Algorithm 1")

        # Independent popup windows, keyed by feature index.
        # Multiple blob windows can remain open simultaneously.
        self.blob_windows: dict[int, tk.Toplevel] = {}

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

        tk.Button(toolbar, text="Load .NPY", command=self.load_file_dialog).pack(side=tk.LEFT, padx=4, pady=5)

        tk.Button(toolbar, text="Reset Raw", command=self.reset_to_raw).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Remove Plane", command=self.apply_global_plane).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Line Flatten", command=self.apply_line_flatten).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Align Rows", command=self.apply_row_alignment).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="2D Background", command=self.apply_2d_background).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Destripe", command=self.apply_horizontal_destripe).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Standard", command=self.apply_standard_preprocessing).pack(side=tk.LEFT, padx=2, pady=5)

        ttk.Label(toolbar, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 3), pady=5)
        self.algorithm_combo = ttk.Combobox(
            toolbar,
            textvariable=self.selected_algorithm,
            values=list(ALGORITHM_FUNCTIONS),
            state="readonly",
            width=12,
        )
        self.algorithm_combo.pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Run", command=self.run_selected_algorithm).pack(side=tk.LEFT, padx=3, pady=5)

        tk.Button(toolbar, text="Exit Edit Mode", command=self.clear_mode).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Toggle Labels", command=self.toggle_labels).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Export Mask & CSV", command=self.export_data, bg="lightblue").pack(side=tk.RIGHT, padx=5, pady=5)

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

        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.fig.canvas.mpl_connect("button_press_event", self.on_canvas_click)
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

        self.load_npy_file(path)

    def load_npy_by_index(self, idx: int) -> None:
        if not self.npy_files:
            return
        idx = max(0, min(idx, len(self.npy_files) - 1))
        self.current_file_index = idx
        self.load_npy_file(self.npy_files[idx])

    def load_npy_file(self, path: str | Path) -> None:
        path = Path(path)
        print("\n=== Loading converted AFM array ===", flush=True)
        print("Path:", path, flush=True)

        try:
            z = load_npy_image(path)

            self.filepath = str(path)
            self.raw_image = z
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
        self.close_all_blob_windows()
        self.tophat_image = None
        self.log_image = None
        self.raw_mask = None
        self.binary_mask = None
        self.features = []

        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None



    def reset_detection_cache(self) -> None:
        """Invalidate cached detection stages without changing the image."""
        self.close_all_blob_windows()
        self.tophat_image = None
        self.log_image = None
        self.raw_mask = None
        self.binary_mask = None
        self.features = []
        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None
        self._needs_otsu = True

    def _set_processed_image(self, image: np.ndarray, step_name: str) -> None:
        """Store a preprocessing result and rerun the selected detector."""
        image = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(image)
        if not np.any(finite):
            raise ValueError("Preprocessing produced no finite pixels.")

        image = image - np.nanmedian(image[finite])
        self.corrected_image = image
        self.preprocessing_history.append(step_name)
        self.reset_detection_cache()
        self.run_selected_algorithm()

    def reset_to_raw(self) -> None:
        if self.raw_image is None:
            return
        self.corrected_image = self.raw_image.copy()
        self.preprocessing_history = []
        self.reset_detection_cache()
        self.run_selected_algorithm()

    def apply_global_plane(self) -> None:
        if self.corrected_image is None:
            return
        corrected, _plane, _mask = subtract_global_plane(self.corrected_image)
        self._set_processed_image(corrected, "global plane")

    def apply_line_flatten(self) -> None:
        if self.corrected_image is None:
            return
        corrected, _background = robust_polynomial_line_flatten(
            self.corrected_image, degree=1, n_iterations=6, sigma=2.5, axis=1
        )
        self._set_processed_image(corrected, "line flatten")

    def apply_row_alignment(self) -> None:
        if self.corrected_image is None:
            return
        corrected, _offsets = align_scanline_offsets(
            self.corrected_image, smoothing_window=21
        )
        self._set_processed_image(corrected, "row alignment")

    def apply_2d_background(self) -> None:
        if self.corrected_image is None:
            return
        corrected, _background, _mask = robust_2d_background(
            self.corrected_image,
            sigma_y=30,
            sigma_x=60,
            iterations=8,
            threshold=2.5,
        )
        self._set_processed_image(corrected, "2D background")

    def apply_horizontal_destripe(self) -> None:
        if self.corrected_image is None:
            return
        corrected, _stripe_map = remove_partial_horizontal_stripes(
            self.corrected_image,
            sigma_x=35,
            sigma_y_small=1.0,
            sigma_y_large=12.0,
            strength=0.75,
        )
        self._set_processed_image(corrected, "horizontal destripe")

    def apply_standard_preprocessing(self) -> None:
        """Start from raw and apply plane, 2D background, and destriping."""
        if self.raw_image is None:
            return

        image, _plane, _plane_mask = subtract_global_plane(
            self.raw_image, n_iterations=6, sigma=2.5
        )
        image, _background, _background_mask = robust_2d_background(
            image,
            sigma_y=30,
            sigma_x=60,
            iterations=8,
            threshold=2.5,
        )
        image, _stripe_map = remove_partial_horizontal_stripes(
            image,
            sigma_x=35,
            sigma_y_small=1.0,
            sigma_y_large=12.0,
            strength=0.75,
        )

        self.preprocessing_history = []
        self._set_processed_image(image, "standard sequence")

    def run_selected_algorithm(self) -> None:
        """Dispatch to the function selected in the algorithm combobox."""
        if self.corrected_image is None:
            return

        algorithm_name = self.selected_algorithm.get()
        function = ALGORITHM_FUNCTIONS.get(algorithm_name)

        if function is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        try:
            function(self)
        except Exception:
            print(f"{algorithm_name} failed:", flush=True)
            traceback.print_exc()
            self.status_label.configure(text=f"{algorithm_name} failed. Check terminal.")

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
                    circularity = float((4 * np.pi * prop.area) / (2 * perimeter)**2)

                if circularity >= circ_cutoff and area_lower <= prop.area <= area_upper:
                    cy, cx = prop.centroid
                    radius = float(prop.equivalent_diameter_area / 2.0)

                    blob = Blob(
                        cx=float(cx),
                        cy=float(cy),
                        radius=radius,
                        area=float(prop.area),
                        circularity=circularity,
                    )

                    blob.calculate_height_parameters(self.raw_image)
                    self.features.append(blob)

                    # self.binary_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

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
        self.axs[0, 0].imshow(self.corrected_image, cmap="afmhot", vmin=z_vmin, vmax=z_vmax)
        self.axs[0, 0].set_title(f"AFM image | features: {len(self.features)}")

        if self.show_labels:
            for f in self.features:
                circle = plt.Circle((f.cx, f.cy), f.radius + 2, color="cyan", fill=False, lw=1.0)
                self.axs[0, 0].add_patch(circle)

        # Plot 2: top-hat image
        if self.tophat_image is not None:
            th_vmin, th_vmax = percentile_limits(self.tophat_image, 1, 99.7)
            self.axs[0, 1].imshow(self.tophat_image, cmap="gray", vmin=th_vmin, vmax=th_vmax)
        self.axs[0, 1].set_title("Top-hat")

        # Plot 3: normalized LoG response
        if self.log_image is not None:
            self.axs[1, 0].imshow(self.log_image, cmap="gray", vmin=0, vmax=1)
        self.axs[1, 0].set_title("Top-hat + LoG response")

        # Plot 4: final binary mask
        if self.binary_mask is not None:
            self.axs[1, 1].imshow(self.binary_mask, cmap="gray", vmin=0, vmax=1)
        self.axs[1, 1].set_title("Final binary mask")

        if self.filepath:
            file_label = Path(self.filepath).name
        else:
            file_label = "No file"

        self.status_label.configure(
            text=(
                f"File: {file_label}\n"
                f"Shape: {self.corrected_image.shape}\n"
                f"Features: {len(self.features)}\n"
                f"Algorithm: {self.selected_algorithm.get()}\n"
                f"Preprocessing: {', '.join(self.preprocessing_history) or 'none'}\n"
                f"Mode: {self.edit_mode or 'normal'}"
            )
        )

        self.fig.tight_layout()
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

        if hasattr(self, "btn_fp"):
            self.btn_fp.configure(relief=tk.SUNKEN if mode == "FP" else tk.RAISED)
        if hasattr(self, "btn_fn"):
            self.btn_fn.configure(relief=tk.SUNKEN if mode == "FN" else tk.RAISED)

        print(f"Mode set to {mode}. Sliders locked.", flush=True)
        self.update_plots()

    def clear_mode(self) -> None:
        self.edit_mode = None
        self.sliders_active = True
        self.set_slider_state("normal")

        if hasattr(self, "btn_fp"):
            self.btn_fp.configure(relief=tk.RAISED)
        if hasattr(self, "btn_fn"):
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

        if event.inaxes not in [self.axs[0, 0], self.axs[1, 1]]:
            return

        if event.xdata is None or event.ydata is None:
            return

        click_x = float(event.xdata)
        click_y = float(event.ydata)

        # Normal mode: clicking inside a detected blob opens an independent
        # popup. Different blobs can remain open simultaneously.
        if self.edit_mode is None:
            feature_index = self.find_feature_at(click_x, click_y)
            if feature_index is not None:
                self.open_blob_dialog(feature_index)
            return

        if self.edit_mode == "FP":
            self.remove_nearest_feature(click_x, click_y)
        elif self.edit_mode == "FN":
            self.add_manual_feature(click_x, click_y)

        self.update_plots()

    def find_feature_at(self, x: float, y: float) -> int | None:
        """Return the index of the blob containing image coordinate (x, y)."""
        matches: list[tuple[float, int]] = []

        for index, blob in enumerate(self.features):
            if blob.contains(x, y):
                distance = float(np.hypot(blob.cx - x, blob.cy - y))
                matches.append((distance, index))

        if not matches:
            return None

        return min(matches, key=lambda item: item[0])[1]

    def open_blob_dialog(self, feature_index: int) -> None:
        """Open an independent popup window for one detected blob."""
        if not 0 <= feature_index < len(self.features):
            return

        existing_window = self.blob_windows.get(feature_index)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.deiconify()
            existing_window.lift()
            existing_window.focus_force()
            return

        blob = self.features[feature_index]

        popup = tk.Toplevel(self.root)
        popup.title(f"Quantum dot {feature_index + 1}")
        popup.geometry("460x500")
        popup.minsize(420, 440)

        self.blob_windows[feature_index] = popup

        def close_this_window() -> None:
            self.close_blob_window(feature_index)

        popup.protocol("WM_DELETE_WINDOW", close_this_window)
        popup.bind("<Escape>", lambda _event: close_this_window())

        header = tk.Frame(popup, bg="#eeeeee")
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text=f"Quantum dot {feature_index + 1}",
            bg="#eeeeee",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(side=tk.LEFT, padx=12, pady=10)

        info = tk.Frame(popup)
        info.pack(fill=tk.X, padx=18, pady=(16, 8))

        values = [
            # ("Centre x", f'{float(blob.cx):.2f} px'),
            # ("Centre y", f'{float(blob.cy):.2f} px'),
            ("Radius", f'{float(blob.radius):.2f} px'),
            ("Area", f'{float(blob.area):.2f} px²'),
            ("Circularity", f'{float(blob.circularity):.3f}'),
        ]

        for row, (name, value) in enumerate(values):
            tk.Label(info, text=name, anchor="w").grid(
                row=row, column=0, sticky="w", pady=2
            )
            tk.Label(info, text=value, anchor="e").grid(
                row=row, column=1, sticky="e", padx=(30, 0), pady=2
            )

        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)

        height_frame = ttk.LabelFrame(popup, text="Height measurements")
        height_frame.pack(fill=tk.X, padx=18, pady=(8, 8))
        height_frame.columnconfigure(1, weight=1)

        height_entries = [
            ("Dot height", blob.dot_height),
            ("Average background height", blob.background_height),
            ("Dot height above background", blob.relative_height),
        ]

        for row, (label_text, value) in enumerate(height_entries):
            ttk.Label(
                height_frame,
                text=label_text,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(10, 12),
                pady=5,
            )

            entry = ttk.Entry(
                height_frame,
                width=22,
            )

            entry.insert(0, f"{value:.10e}")
            entry.configure(state="readonly")

            entry.grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(0, 10),
                pady=5,
            )

        tk.Label(popup, text="Notes / future data", anchor="w").pack(
            fill=tk.X, padx=18, pady=(12, 4)
        )

        notes_box = tk.Text(popup, height=6, wrap=tk.WORD)
        notes_box.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 12))
        notes_box.insert("1.0", str(blob.notes))

        buttons = tk.Frame(popup)
        buttons.pack(fill=tk.X, padx=18, pady=(0, 16))

        tk.Button(buttons, text="Close", command=close_this_window).pack(side=tk.RIGHT)
        tk.Button(
            buttons,
            text="Save",
            command=lambda: self.save_blob_dialog_data(feature_index, notes_box),
            bg="lightblue",
        ).pack(side=tk.RIGHT, padx=(0, 8))

        notes_box.focus_set()

    def save_blob_dialog_data(self, feature_index: int, notes_box: tk.Text) -> None:
        """Save one popup's data without closing the popup."""
        if not 0 <= feature_index < len(self.features):
            self.close_blob_window(feature_index)
            return

        notes = notes_box.get("1.0", tk.END).strip()
        self.features[feature_index].notes = notes

        popup = self.blob_windows.get(feature_index)
        if popup is not None and popup.winfo_exists():
            popup.title(f"Quantum dot {feature_index + 1} — saved")

    def close_blob_window(self, feature_index: int) -> None:
        """Close only one blob popup."""
        popup = self.blob_windows.pop(feature_index, None)
        if popup is not None and popup.winfo_exists():
            popup.destroy()

    def close_all_blob_windows(self) -> None:
        """Close every open blob popup."""
        for feature_index in list(self.blob_windows):
            self.close_blob_window(feature_index)

    def remove_nearest_feature(self, x: float, y: float) -> None:
        if not self.features:
            return

        distances = [(idx, np.hypot(f.cx - x, f.cy - y)) for idx, f in enumerate(self.features)]
        nearest_idx, dist = min(distances, key=lambda item: item[1])
        nearest_feat = self.features[nearest_idx]

        if dist <= nearest_feat.radius + 5:
            removed = self.features.pop(nearest_idx)

            rr, cc = draw_disk(
                (removed.cy, removed.cx),
                removed.radius + 1,
                shape=self.binary_mask.shape,
            )
            self.binary_mask[rr, cc] = 0

    def add_manual_feature(self, x: float, y: float) -> None:
        if self.features:
            radius = float(np.median([blob.radius for blob in self.features]))
        else:
            radius = 5.0

        area = float(np.pi * radius**2)

        blob = Blob(
            cx=x,
            cy=y,
            radius=radius,
            area=area,
            circularity=1.0,
        )
        blob.calculate_height_parameters(self.raw_image)
        self.features.append(blob)

        rr, cc = draw_disk((y, x), radius, shape=self.binary_mask.shape)
        self.binary_mask[rr, cc] = 1



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

        df = pd.DataFrame([blob.to_dict() for blob in self.features])
        df.to_csv(f"{save_path_base}_features.csv", index=False)

        print(f"Successfully exported:", flush=True)
        print(f"  {save_path_base}_mask.npy", flush=True)
        print(f"  {save_path_base}_features.csv", flush=True)




if __name__ == "__main__":
    root = tk.Tk()
    app = QDAnalysisApp(root)
    root.mainloop()

