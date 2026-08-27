import sys, importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

PROJECT_ROOT = Path.cwd().resolve()

while not (
    (PROJECT_ROOT / "Detection").exists()
    and (PROJECT_ROOT / "Image-Preprocessing").exists()
    and (PROJECT_ROOT / "Ground-Truth").exists()
    and (PROJECT_ROOT / "QD-Analysis").exists()
):
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError("Could not find project root.")
    PROJECT_ROOT = PROJECT_ROOT.parent

DETECTION_DIR = PROJECT_ROOT / "Detection"
PREPROCESSING_DIR = PROJECT_ROOT / "Image-Preprocessing"
GROUND_TRUTH_DIR = PROJECT_ROOT / "Ground-Truth"
QD_ANALYSIS_DIR = PROJECT_ROOT / "QD-Analysis"
CLASSICAL_DIR = DETECTION_DIR / "Classical-Optimised"
CNN_DIR = DETECTION_DIR / "Binary-Unet"

for path in [PROJECT_ROOT, DETECTION_DIR, PREPROCESSING_DIR, GROUND_TRUTH_DIR, QD_ANALYSIS_DIR, CLASSICAL_DIR, CNN_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import basic_data_preprocessing as bdp
import detection_algorithms as da
import QD_analysis as qa

importlib.reload(qa)

top5_algorithms = ["U-Net", "Watershed", "Chambolle", "Otsu", "LoG"]

print("PROJECT_ROOT:", PROJECT_ROOT)

def predict_full_scan(model, image, patch_size=128):
    h, w = image.shape
    prediction = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for r in range(0, h - patch_size + 1, patch_size):
        for c in range(0, w - patch_size + 1, patch_size):
            patch = image[r:r + patch_size, c:c + patch_size]
            pred_patch = model.predict(patch[None, ..., None], verbose=0)[0, :, :, 0]
            prediction[r:r + patch_size, c:c + patch_size] += pred_patch
            counts[r:r + patch_size, c:c + patch_size] += 1

    prediction /= np.maximum(counts, 1)
    return prediction


def run_top5_detectors(raw, processed, normalised, unet_model, unet_threshold=0.5):
    results = {}

    log_result = da.log_detect(raw, processed)
    otsu_result = da.otsu_detect(raw, processed)
    chambolle_result = da.chambolle_anisotropic_diffusion_detect(raw, processed)
    watershed_result = da.watershed_detect(raw, processed)

    results["LoG"] = log_result["after_height_filter"]["centres"]
    results["Otsu"] = otsu_result["after_height_filter"]["centres"]
    results["Chambolle"] = chambolle_result["after_height_filter"]["centres"]
    results["Watershed"] = watershed_result["after_height_filter"]["centres"]

    unet_prob = predict_full_scan(unet_model, normalised, patch_size=128)
    unet_mask = unet_prob >= unet_threshold
    unet_centres = qa.centres_from_binary_mask(unet_mask)

    results["U-Net"] = unet_centres

    return results, unet_prob, unet_mask


def merge_nearby_centres(centres, radius=3.0):
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 0:
        return np.empty((0, 2), dtype=float)

    tree = KDTree(centres)
    pairs = list(tree.query_pairs(r=radius))

    parent = np.arange(len(centres))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    groups = {}
    for i in range(len(centres)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged = np.array([centres[idxs].mean(axis=0) for idxs in groups.values()], dtype=float)
    return merged


def build_candidate_centres(detector_centres_dict, radius=3.0):
    all_centres = []

    for algorithm in top5_algorithms:
        centres = np.asarray(detector_centres_dict[algorithm], dtype=float).reshape(-1, 2)
        if len(centres):
            all_centres.append(centres)

    if not all_centres:
        return np.empty((0, 2), dtype=float)

    all_centres = np.vstack(all_centres)
    return merge_nearby_centres(all_centres, radius=radius)


def nearest_detection_flags(candidate_centres, detector_centres, tolerance=3.0):
    candidate_centres = np.asarray(candidate_centres, dtype=float).reshape(-1, 2)
    detector_centres = np.asarray(detector_centres, dtype=float).reshape(-1, 2)

    detected = np.zeros(len(candidate_centres), dtype=int)
    min_distance = np.full(len(candidate_centres), np.nan)

    if len(candidate_centres) == 0 or len(detector_centres) == 0:
        return detected, min_distance

    distances = cdist(candidate_centres, detector_centres)
    nearest = distances.min(axis=1)

    detected[nearest <= tolerance] = 1
    min_distance[:] = nearest

    return detected, min_distance


def label_candidates_against_gt(candidate_centres, gt_centres, tolerance=3.0):
    candidate_centres = np.asarray(candidate_centres, dtype=float).reshape(-1, 2)
    gt_centres = np.asarray(gt_centres, dtype=float).reshape(-1, 2)

    is_true_qd = np.zeros(len(candidate_centres), dtype=int)
    matched_gt_idx = np.full(len(candidate_centres), -1, dtype=int)
    gt_distance = np.full(len(candidate_centres), np.nan)

    if len(candidate_centres) == 0 or len(gt_centres) == 0:
        return is_true_qd, matched_gt_idx, gt_distance

    distances = cdist(candidate_centres, gt_centres)
    cand_idx, gt_idx = linear_sum_assignment(distances)

    for ci, gi in zip(cand_idx, gt_idx):
        d = distances[ci, gi]
        if d <= tolerance:
            is_true_qd[ci] = 1
            matched_gt_idx[ci] = gi
            gt_distance[ci] = d

    return is_true_qd, matched_gt_idx, gt_distance


def build_candidate_table(scan_name, detector_centres_dict, gt_centres, tolerance=3.0, merge_radius=3.0):
    candidate_centres = build_candidate_centres(detector_centres_dict, radius=merge_radius)

    candidate_df = pd.DataFrame({
        "scan_name": scan_name,
        "cy": candidate_centres[:, 0] if len(candidate_centres) else [],
        "cx": candidate_centres[:, 1] if len(candidate_centres) else [],
    })

    for algorithm in top5_algorithms:
        safe = qa.safe_name(algorithm)
        detected, distance = nearest_detection_flags(candidate_centres, detector_centres_dict[algorithm], tolerance=tolerance)
        candidate_df[safe] = detected
        candidate_df[f"{safe}_distance"] = distance

    y, matched_gt_idx, gt_distance = label_candidates_against_gt(candidate_centres, gt_centres, tolerance=tolerance)
    candidate_df["is_true_qd"] = y
    candidate_df["matched_gt_idx"] = matched_gt_idx
    candidate_df["gt_distance"] = gt_distance

    return candidate_df


def add_complementarity_features(candidate_df, rescue_matrix):
    candidate_df = candidate_df.copy()

    for missed_by in top5_algorithms:
        score = np.zeros(len(candidate_df), dtype=float)

        for detector in top5_algorithms:
            if detector == missed_by:
                continue
            score += candidate_df[qa.safe_name(detector)].to_numpy(dtype=float) * rescue_matrix.loc[detector, missed_by]

        candidate_df[f"rescue_{qa.safe_name(missed_by)}"] = score

    return candidate_df



def load_scan(scan_name):
    GT_ROOT = GROUND_TRUTH_DIR / "NPY-Ground-Truth"
    folder = GT_ROOT / scan_name

    raw = np.load(folder / "channel_00___0_data.npy")
    gt_mask = np.load(folder / "channel_00___0_data_mask.npy").astype(bool)

    processed, _ = bdp.preprocess_afm(
        raw,
        remove_plane=True,
        line_flatten=True,
        align_rows=True,
        remove_2d_background=False,
    )

    normalised = bdp.robust_normalise(processed).astype(np.float32)

    return {
        "raw": raw,
        "processed": processed,
        "normalised": normalised,
        "gt_mask": gt_mask,
        "gt_centres": qa.centres_from_binary_mask(gt_mask),
    }


def augment_scan(raw, processed, normalised, mask):
    variants = {}

    transforms = {
        "original": lambda x: x,
        "flip_h": np.fliplr,
        "flip_v": np.flipud,
        "rot90": lambda x: np.rot90(x, 1),
        "rot180": lambda x: np.rot90(x, 2),
        "rot270": lambda x: np.rot90(x, 3),
    }

    for name, transform in transforms.items():
        aug_raw = transform(raw)
        aug_processed = transform(processed)
        aug_normalised = transform(normalised)
        aug_mask = transform(mask)

        variants[name] = {
            "raw": np.ascontiguousarray(aug_raw),
            "processed": np.ascontiguousarray(aug_processed),
            "normalised": np.ascontiguousarray(aug_normalised),
            "gt_mask": np.ascontiguousarray(aug_mask),
            "gt_centres": qa.centres_from_binary_mask(aug_mask),
        }

    return variants


def build_meta_dataset(keys, rescue_matrix):
    tables = []

    for key in keys:
        scan = augmented_scans[key]

        candidate_df = build_candidate_table(
            scan_name=key,
            detector_centres_dict=responses_by_scan[key],
            gt_centres=scan["gt_centres"],
            tolerance=5.0,
            merge_radius=3.0,
        )

        candidate_df["original_scan"] = scan["original_scan"]
        candidate_df = add_complementarity_features(
            candidate_df,
            rescue_matrix,
        )

        tables.append(candidate_df)

    return pd.concat(tables, ignore_index=True)