import pandas as pd
import numpy as np

measured = pd.read_csv("your_scan_features.csv")
truth = pd.read_csv("your_true_values.csv")

measured = measured[measured["qd_number"].notna()].copy()
measured["qd_number"] = measured["qd_number"].astype(int)

comparison = measured.merge(truth, on="qd_number", how="inner")

comparison["height_error_A"] = comparison["local_height_A"] - comparison["true_height_A"]
comparison["height_abs_error_A"] = np.abs(comparison["height_error_A"])

comparison["fwhm_error_A"] = comparison["fwhm_A"] - comparison["true_fwhm_A"]
comparison["fwhm_abs_error_A"] = np.abs(comparison["fwhm_error_A"])

height_mae = comparison["height_abs_error_A"].mean()
height_rmse = np.sqrt(np.mean(comparison["height_error_A"] ** 2))

fwhm_mae = comparison["fwhm_abs_error_A"].mean()
fwhm_rmse = np.sqrt(np.mean(comparison["fwhm_error_A"] ** 2))

print(f"Height MAE:  {height_mae:.3f} Å")
print(f"Height RMSE: {height_rmse:.3f} Å")

print(f"FWHM MAE:    {fwhm_mae:.3f} Å")
print(f"FWHM RMSE:   {fwhm_rmse:.3f} Å")