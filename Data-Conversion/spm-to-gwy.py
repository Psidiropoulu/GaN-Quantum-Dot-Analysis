from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import gwyfile


PROJECT_DIR = Path(__file__).resolve().parent
SPM_DIR = PROJECT_DIR / "SPM"
RESULTS_DIR = PROJECT_DIR / "Data-DISPLAY"

RAW_FOLDERS = [
    SPM_DIR / "1nm QD AFMs",
    SPM_DIR / "2nm QD AFMs",
    SPM_DIR / "3nm QD AFMs",
    SPM_DIR / "raw",
    SPM_DIR / "Uncapped InGaN QDs_different miscut",
    SPM_DIR / "Ground-Truth-SPM",
]

GWY_DIR = PROJECT_DIR / "GWY"
GWY_DIR.mkdir(parents=True, exist_ok=True)

def find_gwyddion_command() -> str:
    """
    Find the Gwyddion executable on macOS.
    """
    candidates = [
        "gwyddion",
        "/opt/homebrew/bin/gwyddion",
        "/usr/local/bin/gwyddion",
    ]

    for cmd in candidates:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            pass

    raise RuntimeError(
        "Could not find Gwyddion. Try running `which gwyddion` in Terminal."
    )


def find_spm_files() -> list[Path]:
    """
    Find all .spm files in the raw AFM folders.
    """
    files: list[Path] = []

    for folder in RAW_FOLDERS:
        if folder.exists():
            files.extend(folder.rglob("*.spm"))
            files.extend(folder.rglob("*.SPM"))

    return sorted(files)


def safe_output_name(spm_path: Path) -> str:
    """
    Create a safe filename that remembers which folder the file came from.
    """
    relative = spm_path.relative_to(PROJECT_DIR)
    name = "__".join(relative.parts)
    name = name.replace(" ", "_")
    name = name.replace(".spm", ".gwy")
    name = name.replace(".SPM", ".gwy")
    return name


def convert_spm_to_gwy(spm_path: Path, gwy_path: Path, gwyddion_cmd: str) -> None:
    """
    Convert .spm to .gwy using Gwyddion command line.
    """
    if gwy_path.exists():
        print(f"Already converted: {gwy_path.name}")
        return

    print(f"Converting: {spm_path}")

    subprocess.run(
        [
            gwyddion_cmd,
            f"--convert-to-gwy={gwy_path}",
            str(spm_path),
        ],
        check=True,
    )


def find_datafields(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """
    Recursively find 2D image-like data fields inside a .gwy file.
    These are usually channels like Height, ZSensor, PeakForce Error, etc.
    """
    found: list[tuple[str, Any]] = []

    if hasattr(obj, "data"):
        arr = np.asarray(obj.data)
        if arr.ndim == 2:
            found.append((prefix, obj))

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            found.extend(find_datafields(value, child_prefix))

    return found


def plane_level(z: np.ndarray) -> np.ndarray:
    """
    Remove a best-fit plane from the AFM image.
    This removes sample tilt.
    """
    z = np.asarray(z, dtype=float)

    ny, nx = z.shape
    y, x = np.mgrid[:ny, :nx]

    mask = np.isfinite(z)

    A = np.column_stack(
        [
            x[mask].ravel(),
            y[mask].ravel(),
            np.ones(mask.sum()),
        ]
    )

    b = z[mask].ravel()

    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)

    plane = coeff[0] * x + coeff[1] * y + coeff[2]

    return z - plane


def save_channel_preview(
    gwy_path: Path,
    channel_name: str,
    z: np.ndarray,
    channel_index: int,
) -> None:
    """
    Save a PNG preview of one channel.
    """
    z_levelled = plane_level(z)

    safe_channel_name = (
        channel_name.replace("/", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
    )

    out_path = RESULTS_DIR / f"{gwy_path.stem}__channel_{channel_index:02d}__{safe_channel_name}.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(z_levelled, origin="lower", cmap="afmhot")
    plt.colorbar(label="Z value")
    plt.title(f"{gwy_path.name}\n{channel_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"    saved preview: {out_path.name}")


def inspect_gwy_file(gwy_path: Path) -> None:
    """
    Load a .gwy file and print/save its channels.
    """
    print()
    print(f"Inspecting: {gwy_path.name}")

    gwy = gwyfile.load(str(gwy_path))
    fields = find_datafields(gwy)

    print(f"  found {len(fields)} image-like channels")

    for i, (name, field) in enumerate(fields):
        z = np.asarray(field.data, dtype=float)

        print(f"  channel {i:02d}: {name}")
        print(f"    shape: {z.shape}")
        print(f"    min:   {np.nanmin(z):.6g}")
        print(f"    max:   {np.nanmax(z):.6g}")
        print(f"    mean:  {np.nanmean(z):.6g}")

        save_channel_preview(
            gwy_path=gwy_path,
            channel_name=name,
            z=z,
            channel_index=i,
        )


def main() -> None:
    gwyddion_cmd = find_gwyddion_command()
    print(f"Using Gwyddion: {gwyddion_cmd}")

    spm_files = find_spm_files()

    if not spm_files:
        print("No .spm files found.")
        print("Check that your .spm files are inside:")
        for folder in RAW_FOLDERS:
            print(f"  {folder}")
        return

    print(f"Found {len(spm_files)} .spm files")

    gwy_files: list[Path] = []

    for spm_path in spm_files:
        gwy_name = safe_output_name(spm_path)
        gwy_path = GWY_DIR / gwy_name

        convert_spm_to_gwy(
            spm_path=spm_path,
            gwy_path=gwy_path,
            gwyddion_cmd=gwyddion_cmd,
        )

        gwy_files.append(gwy_path)

    for gwy_path in gwy_files:
        inspect_gwy_file(gwy_path)


if __name__ == "__main__":
    main()