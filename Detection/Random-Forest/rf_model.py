from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, uniform_filter
from skimage.morphology import remove_small_objects
import numpy as np

from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, remove_small_holes, disk
from skimage.measure import label, regionprops

def smooth_rf_mask(prob_map, threshold=0.55, prob_sigma=1.2, close_radius=2, open_radius=1,
                   min_area=6, max_area=120, min_solidity=0.55, max_eccentricity=0.95):
    prob_smooth = gaussian_filter(prob_map, sigma=prob_sigma)

    mask = prob_smooth >= threshold
    mask = binary_closing(mask, footprint=disk(close_radius))
    mask = binary_opening(mask, footprint=disk(open_radius))
    mask = remove_small_objects(mask, min_size=min_area)
    mask = remove_small_holes(mask, area_threshold=12)
    mask = binary_fill_holes(mask)

    cleaned = np.zeros_like(mask, dtype=bool)

    for region in regionprops(label(mask)):
        area = region.area
        solidity = region.solidity if region.area > 0 else 0
        eccentricity = region.eccentricity

        if area < min_area:
            continue
        if area > max_area:
            continue
        if solidity < min_solidity:
            continue
        if eccentricity > max_eccentricity:
            continue

        cleaned[label(mask) == region.label] = True

    return prob_smooth, cleaned


def rf_pixel_features(image):
    image = np.asarray(image, dtype=np.float32)
    gx, gy = sobel(image, axis=1), sobel(image, axis=0)
    local_mean = uniform_filter(image, size=9, mode="reflect")
    local_var = uniform_filter(image**2, size=9, mode="reflect") - local_mean**2

    features = [
        image,
        gaussian_filter(image, 1),
        gaussian_filter(image, 2),
        gaussian_filter(image, 4),
        -gaussian_laplace(image, 1),
        -gaussian_laplace(image, 2),
        np.sqrt(gx**2 + gy**2),
        local_mean,
        np.sqrt(np.maximum(local_var, 0)),
    ]

    return np.stack(features, axis=-1)


from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, uniform_filter, distance_transform_edt

def rf_pixel_features_mask(image, gan_mask):
    image = np.asarray(image, dtype=np.float32)
    gan_mask = np.asarray(gan_mask, dtype=bool)

    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)

    local_mean = uniform_filter(image, size=9, mode="reflect")
    local_var = uniform_filter(image**2, size=9, mode="reflect") - local_mean**2

    local_contrast = image - local_mean
    local_std = np.sqrt(np.maximum(local_var, 0))

    inside_distance = distance_transform_edt(gan_mask)
    outside_distance = distance_transform_edt(~gan_mask)
    gan_edge_distance = outside_distance - inside_distance

    features = [
        image,
        gaussian_filter(image, 1),
        gaussian_filter(image, 2),
        gaussian_filter(image, 4),
        -gaussian_laplace(image, 1),
        -gaussian_laplace(image, 2),
        np.sqrt(gx**2 + gy**2),
        local_contrast,
        local_std,
        gan_mask.astype(np.float32),
        gan_edge_distance.astype(np.float32),
    ]

    return np.stack(features, axis=-1)