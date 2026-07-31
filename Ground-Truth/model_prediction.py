import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


# IMPORT PROJECT PREPROCESSING
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import data_preprocessing as dp


# LOAD SAVED MODEL
model = tf.keras.models.load_model("Ground-Truth/qd_unet_fold1.keras", compile=False,)


# LOAD UNSEEN TEST SCAN
scan_folder = "Ground-Truth/NPY-Ground-Truth/Scan3"
image = np.load(os.path.join(scan_folder, "channel_00___0_data.npy"))
Y_test = np.load(os.path.join(scan_folder, "channel_00___0_data_mask.npy"))
labelled_ground_truth = label(Y_test, connectivity=2)
ground_truth_regions = regionprops(labelled_ground_truth)
number_of_ground_truth_qds = len(ground_truth_regions)
print("Ground-truth QDs:", number_of_ground_truth_qds)


# PREPROCESS EXACTLY LIKE TRAINING
image_preprocessed, _ = dp.preprocess_afm(
    image,
    remove_plane=True,
    line_flatten=True,
    align_rows=True,
    remove_2d_background=False,
)

X_test = dp.robust_normalise(image_preprocessed)


# PREDICT ON THE UNSEEN TEST SCAN
def predict_full_scan(model, image, patch_size=128):
    height, width = image.shape
    prediction = np.zeros((height, width), dtype=np.float32)

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = image[row:row + patch_size, col:col + patch_size]

            if patch.shape != (patch_size, patch_size):
                continue

            patch_input = patch[np.newaxis, ..., np.newaxis]
            patch_prediction = model.predict(patch_input, verbose=0)[0, :, :, 0]

            prediction[row:row + patch_size, col:col + patch_size] = patch_prediction

    return prediction


prediction = predict_full_scan(model, X_test, patch_size=128,)
predicted_mask = prediction >= 0.1


# REMOVE TINY OBJECTS
clean_mask = remove_small_objects(
    predicted_mask,
    min_size=1,
)


# COUNT QDS
labelled_prediction = label(clean_mask, connectivity=2)
regions = regionprops(labelled_prediction)
number_of_qds = len(regions)


# PLOT RESULTS
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(X_test, cmap="gray")
axes[0].set_title("Unseen AFM scan")

axes[1].imshow(Y_test, cmap="gray")
axes[1].set_title(f"Ground truth: {number_of_ground_truth_qds} QDs")

im = axes[2].imshow(prediction, cmap="viridis", vmin=0, vmax=1)
axes[2].set_title("Predicted QD probability")
fig.colorbar(im, ax=axes[2])

axes[3].imshow(clean_mask, cmap="gray")
axes[3].set_title(f"Predicted mask: {number_of_qds} QDs")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# COMPARE GROUND TRUTH AND PREDICTED MASK

ground_truth_flat = Y_test.astype(np.float32).ravel()
predicted_flat = clean_mask.astype(np.float32).ravel()

correlation = np.corrcoef(ground_truth_flat, predicted_flat)[0, 1]
intersection = np.logical_and(Y_test, clean_mask,).sum()
union = np.logical_or(Y_test,clean_mask,).sum()
dice = (2*intersection) / (Y_test.sum() + clean_mask.sum())
iou = intersection / union

print("Pixelwise correlation:", correlation)
print("Dice coefficient:", dice)
print("IoU:", iou)

probability_correlation = np.corrcoef(
    Y_test.astype(np.float32).ravel(),
    prediction.astype(np.float32).ravel(),
)[0, 1]

print(
    "Ground truth vs probability-map correlation:",
    probability_correlation,
)

thresholds = np.arange(0.1, 0.91, 0.05)

for threshold in thresholds:
    mask = prediction >= threshold

    intersection = np.logical_and(Y_test, mask).sum()
    union = np.logical_or(Y_test, mask).sum()

    dice = 2 * intersection / (Y_test.sum() + mask.sum())
    iou = intersection / union

    print(
        f"threshold={threshold:.2f}  "
        f"dice={dice:.3f}  "
        f"iou={iou:.3f}"
    )