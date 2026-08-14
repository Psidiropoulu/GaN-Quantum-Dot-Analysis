from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEASURED_PATH = (
    PROJECT_ROOT
    / "Ground-Truth"
    / "NPY-Ground-Truth"
    / "SPM_Uncapped_InGaN_QDs_different_miscut_AIX7104C_CTA0.25°_500nm_1"
    / "channel_00___0_data_features.csv"
)

TRUTH_PATH = (
    PROJECT_ROOT
    / "Ground-Truth"
    / "Height-HWFM"
    / "CTA_0.25.csv"
)

print("Measured:", MEASURED_PATH)
print("Exists:", MEASURED_PATH.exists())
print("Truth:", TRUTH_PATH)
print("Exists:", TRUTH_PATH.exists())

measured = pd.read_csv(MEASURED_PATH)
truth = pd.read_csv(TRUTH_PATH)

# CLEAN MEASURED QD NUMBERS
measured = measured[measured["qd_number"].notna()].copy()
measured["qd_number"] = measured["qd_number"].astype(int)

# ----------------------------------------------------------
# FIX GROUND-TRUTH NUMBERING
#
# Ground truth accidentally contains:
# ... 34, 35, 36, 36, 37, 38, 39, ...
#
# Correct numbering should be:
# ... 34, 35, 36, 37, 38, 39, 40, ...
#
# Therefore:
# first 36 stays 36
# second 36 becomes 37
# everything after that is shifted by +1
# ----------------------------------------------------------

truth = truth.copy()
truth["original_qd_number"] = truth["qd_number"].astype(int)

corrected_numbers = []
seen_first_36 = False

for q in truth["original_qd_number"]:
    if q < 36:
        corrected_numbers.append(q)

    elif q == 36 and not seen_first_36:
        corrected_numbers.append(36)
        seen_first_36 = True

    elif q == 36 and seen_first_36:
        corrected_numbers.append(37)

    else:
        corrected_numbers.append(q + 1)

truth["qd_number"] = corrected_numbers

# CHECK THE CORRECTION
print("\nGround-truth numbering around duplicated 36:")
print(
    truth[
        (truth["original_qd_number"] >= 33)
        & (truth["original_qd_number"] <= 40)
    ][
        ["original_qd_number", "qd_number", "true_height_A", "true_fwhm_A"]
    ].to_string(index=False)
)

# MAKE SURE CORRECTED NUMBERS ARE UNIQUE
if truth["qd_number"].duplicated().any():
    duplicates = truth.loc[
        truth["qd_number"].duplicated(keep=False),
        ["original_qd_number", "qd_number"]
    ]
    raise ValueError(
        "Corrected truth qd_number still contains duplicates:\n"
        + duplicates.to_string(index=False)
    )

# ----------------------------------------------------------
# MERGE MEASURED QDS WITH CORRECTED GROUND TRUTH
# ----------------------------------------------------------

comparison = measured.merge(
    truth,
    on="qd_number",
    how="inner",
)

print(f"\nMeasured QDs: {len(measured)}")
print(f"Truth measurements: {len(truth)}")
print(f"Matched QDs: {len(comparison)}")

# SHOW UNMATCHED MEASURED QDS
matched_numbers = set(comparison["qd_number"])
unmatched_measured = measured[
    ~measured["qd_number"].isin(matched_numbers)
]

if len(unmatched_measured):
    print("\nMeasured QDs without height/FWHM truth:")
    print(
        unmatched_measured[
            ["qd_number", "local_height_A", "fwhm_A", "manual_status"]
        ].to_string(index=False)
    )

# ----------------------------------------------------------
# HEIGHT ERRORS IN Å
# ----------------------------------------------------------

comparison["height_error_A"] = (
    comparison["local_height_A"]
    - comparison["true_height_A"]
)

comparison["height_abs_error_A"] = np.abs(
    comparison["height_error_A"]
)

# ----------------------------------------------------------
# FWHM ERRORS IN Å
# ----------------------------------------------------------

comparison["fwhm_error_A"] = (
    comparison["fwhm_A"]
    - comparison["true_fwhm_A"]
)

comparison["fwhm_abs_error_A"] = np.abs(
    comparison["fwhm_error_A"]
)

# ----------------------------------------------------------
# MAE + RMSE
# ----------------------------------------------------------

height_mae = comparison["height_abs_error_A"].mean()

height_rmse = np.sqrt(
    np.mean(comparison["height_error_A"] ** 2)
)

fwhm_mae = comparison["fwhm_abs_error_A"].mean()

fwhm_rmse = np.sqrt(
    np.mean(comparison["fwhm_error_A"] ** 2)
)

# ----------------------------------------------------------
# RELATIVE ERRORS
# ----------------------------------------------------------

comparison["height_relative_error_pct"] = (
    100
    * comparison["height_abs_error_A"]
    / np.abs(comparison["true_height_A"])
)

comparison["fwhm_relative_error_pct"] = (
    100
    * comparison["fwhm_abs_error_A"]
    / np.abs(comparison["true_fwhm_A"])
)

# ----------------------------------------------------------
# PRINT RESULTS
# ----------------------------------------------------------

print(f"\nHeight MAE:  {height_mae:.3f} Å")
print(f"Height RMSE: {height_rmse:.3f} Å")

print(f"FWHM MAE:    {fwhm_mae:.3f} Å")
print(f"FWHM RMSE:   {fwhm_rmse:.3f} Å")

print(
    f"Mean height relative error: "
    f"{comparison['height_relative_error_pct'].mean():.1f}%"
)

print(
    f"Median height relative error: "
    f"{comparison['height_relative_error_pct'].median():.1f}%"
)

print(
    f"Mean FWHM relative error: "
    f"{comparison['fwhm_relative_error_pct'].mean():.1f}%"
)

print(
    f"Median FWHM relative error: "
    f"{comparison['fwhm_relative_error_pct'].median():.1f}%"
)

# ----------------------------------------------------------
# WORST ERRORS
# ----------------------------------------------------------

print("\nWORST HEIGHT ERRORS:")

print(
    comparison.nlargest(
        10,
        "height_abs_error_A"
    )[
        [
            "qd_number",
            "original_qd_number",
            "local_height_A",
            "true_height_A",
            "height_error_A",
            "height_relative_error_pct",
        ]
    ].to_string(index=False)
)

print("\nWORST FWHM ERRORS:")

print(
    comparison.nlargest(
        10,
        "fwhm_abs_error_A"
    )[
        [
            "qd_number",
            "original_qd_number",
            "fwhm_A",
            "true_fwhm_A",
            "fwhm_error_A",
            "fwhm_relative_error_pct",
        ]
    ].to_string(index=False)
)

# ----------------------------------------------------------
# SAVE HUMAN-READABLE COMPARISON TABLE
# ----------------------------------------------------------

pretty_table = comparison[
    [
        "qd_number",
        "original_qd_number",
        "local_height_A",
        "true_height_A",
        "height_error_A",
        "height_abs_error_A",
        "height_relative_error_pct",
        "fwhm_A",
        "true_fwhm_A",
        "fwhm_error_A",
        "fwhm_abs_error_A",
        "fwhm_relative_error_pct",
    ]
].copy()

pretty_table = pretty_table.rename(
    columns={
        "qd_number": "QD",
        "original_qd_number": "Original GT label",
        "local_height_A": "Measured height (Å)",
        "true_height_A": "True height (Å)",
        "height_error_A": "Height error (Å)",
        "height_abs_error_A": "Height abs. error (Å)",
        "height_relative_error_pct": "Height error (%)",
        "fwhm_A": "Measured FWHM (Å)",
        "true_fwhm_A": "True FWHM (Å)",
        "fwhm_error_A": "FWHM error (Å)",
        "fwhm_abs_error_A": "FWHM abs. error (Å)",
        "fwhm_relative_error_pct": "FWHM error (%)",
    }
)

pretty_table = pretty_table.sort_values("QD").round(2)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Ground-Truth"
    / "Height-HWFM"
    / "CTA_0.25_error_comparison.xlsx"
)

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    pretty_table.to_excel(writer, index=False, sheet_name="QD comparison")

    ws = writer.book["QD comparison"]

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column_cells in ws.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 24)

print(f"\nPretty comparison table saved to:\n{OUTPUT_PATH}")

import matplotlib.pyplot as plt

# HEIGHT ERROR HISTOGRAM
height_errors = comparison["height_error_A"].dropna()

plt.figure(figsize=(7, 5))
plt.hist(height_errors, bins=20, edgecolor="black", alpha=0.8)

plt.axvline(0, linestyle="--", linewidth=1.5, label="Zero error")
plt.axvline(height_errors.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {height_errors.mean():.2f} Å")

plt.xlabel("Height error (Å)")
plt.ylabel("Frequency")
plt.title("Distribution of QD Height Errors")
plt.legend()
plt.tight_layout()

FIGURE_PATH = PROJECT_ROOT / "Ground-Truth" / "Height-HWFM" / "height_error_histogram.pdf"
plt.savefig(FIGURE_PATH, bbox_inches="tight")
plt.show()

print(f"Histogram saved to:\n{FIGURE_PATH}")