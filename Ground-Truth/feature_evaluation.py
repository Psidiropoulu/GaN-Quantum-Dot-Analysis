import os, sys
from pathlib import Path
import os, sys
from IPython.display import display
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from scipy.spatial import KDTree
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

PROJECT_ROOT = Path.cwd()
PREPROCESSING_DIR = PROJECT_ROOT / "Image-Preprocessing"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PREPROCESSING_DIR))

import basic_data_preprocessing as bdp
import foreground_preprocessing as fp
import detection_algorithms as da


def evaluate_feature_masks(ground_truth_mask, predicted_mask, tolerance=3.0):
    ground_truth_mask, predicted_mask = np.asarray(ground_truth_mask, dtype=bool), np.asarray(predicted_mask, dtype=bool)

    gt_labeled, num_gt = ndimage.label(ground_truth_mask)
    pred_labeled, num_pred = ndimage.label(predicted_mask)
    if num_gt == 0 and num_pred == 0: 
        return {"TP": 0, "FP": 0, "FN": 0, "GT count": 0, "Predicted count": 0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0}
    if num_gt == 0: 
        return {"TP": 0, "FP": num_pred, "FN": 0, "GT count": 0, "Predicted count": num_pred, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    if num_pred == 0: 
        return {"TP": 0, "FP": 0, "FN": num_gt, "GT count": num_gt, "Predicted count": 0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    gt_centroids = np.array([ndimage.center_of_mass(ground_truth_mask, gt_labeled, i) for i in range(1, num_gt + 1)])
    pred_centroids = np.array([ndimage.center_of_mass(predicted_mask, pred_labeled, i) for i in range(1, num_pred + 1)])

    matches_gt_to_pred = KDTree(gt_centroids).query_ball_tree(KDTree(pred_centroids), r=tolerance)
    matched_gt, matched_pred = set(), set()

    for gt_idx, pred_indices in enumerate(matches_gt_to_pred):
        if not pred_indices:
            continue
        pred_indices = sorted(pred_indices, key=lambda p: np.linalg.norm(gt_centroids[gt_idx] - pred_centroids[p]))
        for pred_idx in pred_indices:
            if pred_idx not in matched_pred:
                matched_gt.add(gt_idx); matched_pred.add(pred_idx) 
                break
            
    tp = len(matched_gt); fp = num_pred - tp; fn = num_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "GT count": num_gt, "Predicted count": num_pred, "Precision": precision, "Recall": recall, "F1": f1}



def predict_full_scan(model, image, patch_size=128):
    height, width = image.shape
    prediction = np.zeros((height, width), dtype=np.float32)
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = image[row:row + patch_size, col:col + patch_size]
            if patch.shape != (patch_size, patch_size): 
                continue
            prediction[row:row + patch_size, col:col + patch_size] = model.predict(patch[np.newaxis, ..., np.newaxis], verbose=0)[0, :, :, 0]
    return prediction



# TP / FP / FN OVERLAY FUNCTION: GREEN=TP REGION, RED=FP, BLUE=FN
def make_error_overlay(gt_mask, pred_mask, tolerance=3.0):
    structure = np.ones((3,3), dtype=int)
    gt_labels, ngt = ndimage.label(gt_mask, structure=structure); pred_labels, npred = ndimage.label(pred_mask, structure=structure)
    gt_centres = np.array([ndimage.center_of_mass(gt_mask, gt_labels, i) for i in range(1, ngt+1)]) if ngt else np.empty((0,2))
    pred_centres = np.array([ndimage.center_of_mass(pred_mask, pred_labels, i) for i in range(1, npred+1)]) if npred else np.empty((0,2))
    matched_gt, matched_pred = set(), set()
    if ngt and npred:
        from scipy.spatial import KDTree
        candidates = KDTree(gt_centres).query_ball_tree(KDTree(pred_centres), r=tolerance)
        pairs = sorted([(np.linalg.norm(gt_centres[g]-pred_centres[p]), g, p) for g, ps in enumerate(candidates) for p in ps])
        for _, g, p in pairs:
            if g not in matched_gt and p not in matched_pred: matched_gt.add(g); matched_pred.add(p)
    overlay = np.zeros((*gt_mask.shape, 3), dtype=float)
    for g in matched_gt: overlay[gt_labels == g+1] = [0,1,0]
    for p in range(npred):
        if p not in matched_pred: overlay[pred_labels == p+1] = [1,0,0]
    for g in range(ngt):
        if g not in matched_gt: overlay[gt_labels == g+1] = [0,0.4,1]
    return overlay