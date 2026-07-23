from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gwyfile


PROJECT_DIR = Path(__file__).resolve().parent

GWY_DIR = PROJECT_DIR / "GWY"
NPY_DIR = PROJECT_DIR / "NPY"

NPY_DIR.mkdir(exist_ok=True)


def find_datafields(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """
    Recursively search through a .gwy file and find 2D image channels.

    A .gwy file is a nested container.
    The useful AFM images are usually stored as 2D data fields.
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


def safe_name(text: str) -> str:
    """
    Convert a file/channel name into something safe for filenames.
    """
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("__", "_")
    )


def extract_one_gwy(gwy_path: Path) -> None:
    """
    Load one .gwy file and save every 2D image channel as .npy.
    """
    print()
    print(f"Reading: {gwy_path.name}")

    gwy = gwyfile.load(str(gwy_path))
    fields = find_datafields(gwy)

    if not fields:
        print("  No image channels found.")
        return

    file_out_dir = NPY_DIR / safe_name(gwy_path.stem)
    file_out_dir.mkdir(exist_ok=True)

    print(f"  Found {len(fields)} image channels")

    for channel_index, (channel_name, field) in enumerate(fields):
        z = np.asarray(field.data, dtype=np.float64)

        output_name = f"channel_{channel_index:02d}__{safe_name(channel_name)}.npy"
        output_path = file_out_dir / output_name

        np.save(output_path, z)

        print(f"  Saved: {output_path}")
        print(f"    shape: {z.shape}")
        print(f"    min:   {np.nanmin(z):.6g}")
        print(f"    max:   {np.nanmax(z):.6g}")
        print(f"    mean:  {np.nanmean(z):.6g}")


def main() -> None:
    gwy_files = sorted(GWY_DIR.glob("*.gwy"))

    if not gwy_files:
        print(f"No .gwy files found in: {GWY_DIR}")
        return

    print(f"Found {len(gwy_files)} .gwy files")

    for gwy_path in gwy_files:
        extract_one_gwy(gwy_path)


if __name__ == "__main__":
    main()