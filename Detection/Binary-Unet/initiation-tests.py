import os
import numpy as np
from skimage.measure import label

ground_truth_root = "Ground-Truth/NPY-Ground-Truth"


# tests at the start of the project to check if there is actually any data loaded to the ground truth dataset
for scan_name in ["Scan1", "Scan2", "Scan3"]:

    scan_folder = os.path.join(ground_truth_root, scan_name)
    image_path = os.path.join(scan_folder,"channel_00___0_data.npy")
    mask_path = os.path.join(scan_folder, "channel_00___0_data_mask.npy")

    image = np.load(image_path)
    mask = np.load(mask_path)

    labelled_mask = label(mask, connectivity=2)
    number_of_qds = labelled_mask.max()

    print("\n", scan_name)
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", np.unique(mask))
    print("QD pixels:", np.sum(mask > 0))
    print("Labelled QDs:", number_of_qds)