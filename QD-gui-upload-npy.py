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

        # Each feature is a dict: cx, cy, r, area, circularity
        self.features: list[dict[str, float]] = []

        self.edit_mode: str | None = None  # 'FP', 'FN', or None
        self.sliders_active = True
        self.show_labels = True
        self._needs_otsu = False

        # Cache trackers: used so that only changed pipeline stages rerun
        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None

        self._setup_ui()
        self.scan_default_npy_folder()



    def _setup_ui(self) -> None:
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Load .NPY", command=self.load_file_dialog).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Scan data_numpy", command=self.scan_default_npy_folder).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Previous", command=self.load_previous_file).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="Next", command=self.load_next_file).pack(side=tk.LEFT, padx=5, pady=5)

        self.file_combo = ttk.Combobox(toolbar, width=65, state="readonly")
        self.file_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_combo_selected)

        self.btn_fp = tk.Button(toolbar, text="Mark False Positive", command=lambda: self.set_mode("FP"))
        self.btn_fp.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_fn = tk.Button(toolbar, text="Mark False Negative", command=lambda: self.set_mode("FN"))
        self.btn_fn.pack(side=tk.LEFT, padx=5, pady=5)

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



    def scan_default_npy_folder(self) -> None:
        """
        Recursively find all .npy files inside data_numpy.
        """
        if not DEFAULT_NPY_DIR.exists():
            self.npy_files = []
            self.file_combo["values"] = []
            self.status_label.configure(text=f"Folder not found:\n{DEFAULT_NPY_DIR}")
            return

        self.npy_files = sorted(DEFAULT_NPY_DIR.rglob("*.npy"))

        display_names = [str(p.relative_to(DEFAULT_NPY_DIR)) for p in self.npy_files]
        self.file_combo["values"] = display_names

        if self.npy_files:
            self.status_label.configure(text=f"Found {len(self.npy_files)} .npy files in data_numpy")
            self.file_combo.current(0)
        else:
            self.status_label.configure(text="No .npy files found in data_numpy")

    def on_combo_selected(self, _event=None) -> None:
        idx = self.file_combo.current()
        if idx >= 0:
            self.load_npy_by_index(idx)

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
        self.file_combo.current(idx)
        self.load_npy_file(self.npy_files[idx])

    def load_npy_file(self, path: str | Path) -> None:
        path = Path(path)
        print("\n=== Loading converted AFM array ===", flush=True)
        print("Path:", path, flush=True)

        try:
            z = load_npy_image(path)

            self.filepath = str(path)
            self.raw_image = z
            self.corrected_image = z

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

        self.last_r_th = None
        self.last_sigma = None
        self.last_thresh = None
        self.last_circ = None
        self.last_area_upper = None
        self.last_area_lower = None



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
                    cy, cx = prop.centroid
                    radius = float(prop.equivalent_diameter_area / 2.0)

                    self.features.append(
                        {
                            "cx": float(cx),
                            "cy": float(cy),
                            "r": radius,
                            "area": float(prop.area),
                            "circularity": circularity,
                        }
                    )

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
        self.axs[0, 0].imshow(self.corrected_image, cmap="afmhot", vmin=z_vmin, vmax=z_vmax)
        self.axs[0, 0].set_title(f"AFM image | features: {len(self.features)}")

        if self.show_labels:
            for f in self.features:
                circle = plt.Circle((f["cx"], f["cy"]), f["r"] + 2, color="cyan", fill=False, lw=1.0)
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

        if not self.edit_mode or event.inaxes not in [self.axs[0, 0], self.axs[1, 1]]:
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

    def remove_nearest_feature(self, x: float, y: float) -> None:
        if not self.features:
            return

        distances = [(idx, np.hypot(f["cx"] - x, f["cy"] - y)) for idx, f in enumerate(self.features)]
        nearest_idx, dist = min(distances, key=lambda item: item[1])
        nearest_feat = self.features[nearest_idx]

        if dist <= nearest_feat["r"] + 5:
            removed = self.features.pop(nearest_idx)

            rr, cc = draw_disk(
                (removed["cy"], removed["cx"]),
                removed["r"] + 1,
                shape=self.binary_mask.shape,
            )
            self.binary_mask[rr, cc] = 0

    def add_manual_feature(self, x: float, y: float) -> None:
        if self.features:
            radius = float(np.median([f["r"] for f in self.features]))
        else:
            radius = 5.0

        area = float(np.pi * radius**2)

        self.features.append(
            {
                "cx": x,
                "cy": y,
                "r": radius,
                "area": area,
                "circularity": 1.0,
            }
        )

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

        df = pd.DataFrame(self.features)
        df.to_csv(f"{save_path_base}_features.csv", index=False)

        print(f"Successfully exported:", flush=True)
        print(f"  {save_path_base}_mask.npy", flush=True)
        print(f"  {save_path_base}_features.csv", flush=True)




if __name__ == "__main__":
    root = tk.Tk()
    app = AFMSegmentationApp(root)
    root.mainloop()

