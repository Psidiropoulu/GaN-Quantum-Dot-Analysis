from matplotlib import image
import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_laplace
from skimage.draw import disk
from skimage.exposure import rescale_intensity
from skimage.feature import SIFT, blob_log, canny, peak_local_max
from skimage.filters import gabor, gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk as morphology_disk
from skimage.morphology import remove_small_objects, white_tophat
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_chan_vese,
    morphological_geodesic_active_contour,
    watershed,
)
from skimage.transform import hough_circle, hough_circle_peaks
from detection_errors import evaluate_detection


def ground_truth_parameters_from_mask(binary_mask):
    labelled_mask = label(np.asarray(binary_mask, dtype=bool), connectivity=2)
    regions = regionprops(labelled_mask)

    centres = np.asarray([region.centroid for region in regions], dtype=float).reshape(-1, 2)
    areas = np.asarray([region.area for region in regions], dtype=float)
    radii = np.sqrt(areas / np.pi)

    return centres, radii, areas


# CONSISTENT OUTPUT HELPERS
def _empty_centres():
    return np.empty((0, 2), dtype=float)


def _empty_values():
    return np.empty(0, dtype=float)


def _normalise_centres(centres):
    centres = np.asarray(centres, dtype=float)
    return centres.reshape(-1, 2) if centres.size else _empty_centres()


def _normalise_values(values, count, fill_value=np.nan):
    if values is None:
        return np.full(count, fill_value, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != count:
        raise ValueError("The number of values must match the number of QD centres.")
    return values


def circles_to_mask(shape, centres, radii, default_radius=1.0):
    centres = _normalise_centres(centres)
    radii = _normalise_values(radii, len(centres))
    mask = np.zeros(shape, dtype=bool)

    for (row, column), radius in zip(centres, radii):
        radius = float(radius) if np.isfinite(radius) and radius > 0 else float(default_radius)
        rr, cc = disk((row, column), radius=radius, shape=shape)
        mask[rr, cc] = True

    return mask


def make_detection_stage(image_shape, centres, heights=None, radii=None, areas=None, mask=None):
    centres = _normalise_centres(centres)
    count = len(centres)
    heights = _normalise_values(heights, count)
    radii = _normalise_values(radii, count)

    if areas is None:
        areas = np.where(np.isfinite(radii), np.pi * radii**2, np.nan)
    else:
        areas = _normalise_values(areas, count)

    if mask is None:
        mask = circles_to_mask(image_shape, centres, radii)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image_shape:
            raise ValueError("Detection mask shape does not match the image shape.")

    return {
        "mask": mask,
        "centres": centres,
        "heights": heights,
        "radii": radii,
        "areas": areas,
    }



def make_ground_truth_result(mask, processed_image):
    mask = np.asarray(mask, dtype=bool)
    processed_image = np.asarray(processed_image, dtype=float)

    labelled_mask = label(mask, connectivity=2)
    regions = regionprops(labelled_mask)

    centres = []
    heights = []
    radii = []
    areas = []

    for region in regions:
        centre_y, centre_x = region.centroid
        area = float(region.area)
        radius = float(np.sqrt(area / np.pi))

        region_rows = region.coords[:, 0]
        region_columns = region.coords[:, 1]

        peak_index = np.argmax(processed_image[region_rows, region_columns])
        peak_y = region_rows[peak_index]
        peak_x = region_columns[peak_index]

        peak_value = processed_image[peak_y, peak_x]

        inner_radius = 5
        outer_radius = 10

        y0 = max(0, peak_y - outer_radius)
        y1 = min(processed_image.shape[0], peak_y + outer_radius + 1)
        x0 = max(0, peak_x - outer_radius)
        x1 = min(processed_image.shape[1], peak_x + outer_radius + 1)

        patch = processed_image[y0:y1, x0:x1]

        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_squared = (yy - peak_y) ** 2 + (xx - peak_x) ** 2

        ring = (
            (distance_squared >= inner_radius**2)
            & (distance_squared <= outer_radius**2)
            & np.isfinite(patch)
        )

        if np.count_nonzero(ring) >= 10:
            background = np.median(patch[ring])
            height = peak_value - background
        else:
            height = np.nan

        centres.append((peak_y, peak_x))
        heights.append(height)
        radii.append(radius)
        areas.append(area)

    return {
        "mask": mask,
        "centres": np.asarray(centres, dtype=float).reshape(-1, 2),
        "heights": np.asarray(heights, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "areas": np.asarray(areas, dtype=float),
    }


def filter_by_local_height(
    candidate_centres,
    z_physical,
    minimum_height,
    search_radius=2,
    background_inner_radius=5,
    background_outer_radius=10,
):
    candidate_centres = _normalise_centres(candidate_centres)
    z_physical = np.asarray(z_physical, dtype=float)
    rows, columns = z_physical.shape

    measured_centres = []
    measured_heights = []
    source_indices = []

    for candidate_index, (candidate_y, candidate_x) in enumerate(candidate_centres):
        candidate_y = int(round(candidate_y))
        candidate_x = int(round(candidate_x))

        y0 = max(0, candidate_y - search_radius)
        y1 = min(rows, candidate_y + search_radius + 1)
        x0 = max(0, candidate_x - search_radius)
        x1 = min(columns, candidate_x + search_radius + 1)
        search_patch = z_physical[y0:y1, x0:x1]

        if search_patch.size == 0 or not np.any(np.isfinite(search_patch)):
            continue

        local_y, local_x = np.unravel_index(np.nanargmax(search_patch), search_patch.shape)
        peak_y = y0 + local_y
        peak_x = x0 + local_x
        peak_value = z_physical[peak_y, peak_x]

        by0 = max(0, peak_y - background_outer_radius)
        by1 = min(rows, peak_y + background_outer_radius + 1)
        bx0 = max(0, peak_x - background_outer_radius)
        bx1 = min(columns, peak_x + background_outer_radius + 1)
        background_patch = z_physical[by0:by1, bx0:bx1]

        yy, xx = np.ogrid[by0:by1, bx0:bx1]
        distance_squared = (yy - peak_y) ** 2 + (xx - peak_x) ** 2
        ring_mask = (
            (distance_squared >= background_inner_radius**2)
            & (distance_squared <= background_outer_radius**2)
            & np.isfinite(background_patch)
        )

        if np.count_nonzero(ring_mask) < 10:
            continue

        local_background = np.median(background_patch[ring_mask])
        dot_height = peak_value - local_background
        measured_centres.append((peak_y, peak_x))
        measured_heights.append(dot_height)
        source_indices.append(candidate_index)

    measured_centres = _normalise_centres(measured_centres)
    measured_heights = np.asarray(measured_heights, dtype=float)
    source_indices = np.asarray(source_indices, dtype=int)

    if len(measured_centres):
        _, unique_indices = np.unique(measured_centres, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        measured_centres = measured_centres[unique_indices]
        measured_heights = measured_heights[unique_indices]
        source_indices = source_indices[unique_indices]

    accepted_mask = measured_heights >= minimum_height

    return {
        "measured_centres": measured_centres,
        "measured_heights": measured_heights,
        "source_indices": source_indices,
        "accepted_mask": accepted_mask,
        "accepted_indices": source_indices[accepted_mask],
        "rejected_indices": source_indices[~accepted_mask],
        "accepted_centres": measured_centres[accepted_mask],
        "accepted_heights": measured_heights[accepted_mask],
        "rejected_centres": measured_centres[~accepted_mask],
        "rejected_heights": measured_heights[~accepted_mask],
    }


def _filter_label_mask(label_mask, kept_region_indices):
    kept_region_indices = np.asarray(kept_region_indices, dtype=int)
    if kept_region_indices.size == 0:
        return np.zeros_like(label_mask, dtype=bool)
    kept_labels = kept_region_indices + 1
    return np.isin(label_mask, kept_labels)


def build_detection_result(
    processed_image,
    z_physical,
    candidate_centres,
    candidate_radii=None,
    candidate_areas=None,
    candidate_mask=None,
    candidate_label_mask=None,
    minimum_height=6e-10,
    search_radius=2,
    background_inner_radius=5,
    background_outer_radius=10,
    extra=None,
):
    z_physical = np.asarray(z_physical, dtype=float)
    candidate_centres = _normalise_centres(candidate_centres)
    count = len(candidate_centres)
    candidate_radii = _normalise_values(candidate_radii, count)

    if candidate_areas is None:
        candidate_areas = np.where(np.isfinite(candidate_radii), np.pi * candidate_radii**2, np.nan)
    else:
        candidate_areas = _normalise_values(candidate_areas, count)

    height_result = filter_by_local_height(
        candidate_centres=candidate_centres,
        z_physical=z_physical,
        minimum_height=minimum_height,
        search_radius=search_radius,
        background_inner_radius=background_inner_radius,
        background_outer_radius=background_outer_radius,
    )

    measured_indices = height_result["source_indices"]
    measured_radii = candidate_radii[measured_indices]
    measured_areas = candidate_areas[measured_indices]
    accepted_local = height_result["accepted_mask"]

    if candidate_label_mask is not None:
        before_mask = np.asarray(candidate_label_mask) > 0
        after_mask = _filter_label_mask(candidate_label_mask, height_result["accepted_indices"])
    else:
        before_mask = candidate_mask
        after_mask = None

    before = make_detection_stage(
        image_shape=z_physical.shape,
        centres=height_result["measured_centres"],
        heights=height_result["measured_heights"],
        radii=measured_radii,
        areas=measured_areas,
        mask=before_mask,
    )

    after = make_detection_stage(
        image_shape=z_physical.shape,
        centres=height_result["accepted_centres"],
        heights=height_result["accepted_heights"],
        radii=measured_radii[accepted_local],
        areas=measured_areas[accepted_local],
        mask=after_mask,
    )

    return {
        "processed_image": np.asarray(processed_image),
        "before_height_filter": before,
        "after_height_filter": after,
        "extra": {} if extra is None else extra,
    }



# DETECTORS

def log_detect(
    striped_image,
    destriped_image,
    destriped=True,
    smoothing=True,
    gaussian_sigma=1,
    min_sigma=2,
    max_sigma=8,
    num_sigma=20,
    threshold=0.4,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    image_normalised = (image - np.mean(image)) / np.std(image)
    processed_image = gaussian(image_normalised, sigma=gaussian_sigma, preserve_range=True) if smoothing else image_normalised.copy()
    blobs = blob_log(processed_image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    centres = blobs[:, :2] if len(blobs) else _empty_centres()
    radii = np.sqrt(2) * blobs[:, 2] if len(blobs) else _empty_values()
    return build_detection_result(processed_image, image, centres, radii, minimum_height=minimum_height, extra={"raw_blobs": blobs})


def chambolle_anisotropic_diffusion_detect(
    striped_image,
    destriped_image,
    destriped=True,
    smoothing=True,
    diffusion_weight=0.08,
    max_num_iter=200,
    min_sigma=2,
    max_sigma=8,
    num_sigma=20,
    threshold=0.4,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    image_normalised = (image - np.mean(image)) / np.std(image)
    processed_image = denoise_tv_chambolle(image_normalised, weight=diffusion_weight, max_num_iter=max_num_iter) if smoothing else image_normalised.copy()
    blobs = blob_log(processed_image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    centres = blobs[:, :2] if len(blobs) else _empty_centres()
    radii = np.sqrt(2) * blobs[:, 2] if len(blobs) else _empty_values()
    return build_detection_result(processed_image, image, centres, radii, minimum_height=minimum_height, extra={"raw_blobs": blobs})


def otsu_detect(
    striped_image,
    destriped_image,
    destriped=True,
    min_sigma=2,
    max_sigma=8,
    num_sigma=20,
    overlap=0.5,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    image_normalised = (image - np.mean(image)) / np.std(image)
    positive_responses = []

    for sigma in np.linspace(min_sigma, max_sigma, num_sigma):
        response = -gaussian_laplace(image_normalised, sigma=sigma) * sigma**2
        positive_part = response[response > 0]
        if positive_part.size:
            positive_responses.append(positive_part)

    if positive_responses:
        otsu_threshold = threshold_otsu(np.concatenate(positive_responses))
        blobs = blob_log(image_normalised, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=otsu_threshold, overlap=overlap)
    else:
        otsu_threshold = np.nan
        blobs = np.empty((0, 3), dtype=float)

    centres = blobs[:, :2] if len(blobs) else _empty_centres()
    radii = np.sqrt(2) * blobs[:, 2] if len(blobs) else _empty_values()
    return build_detection_result(image_normalised, image, centres, radii, minimum_height=minimum_height, extra={"raw_blobs": blobs, "otsu_threshold": otsu_threshold})


def hough_transform_detect(
    striped_image,
    destriped_image,
    destriped=True,
    canny_sigma=1,
    min_radius=2,
    max_radius=8,
    min_xdistance=5,
    min_ydistance=5,
    threshold_relative=0.73,
    normalize=True,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    median_height = np.median(image)
    mad = np.median(np.abs(image - median_height))
    robust_std = 1.4826 * mad
    if robust_std == 0:
        raise ValueError("Robust standard deviation is zero.")

    image_normalised = (image - median_height) / robust_std
    edges = canny(image_normalised, sigma=canny_sigma)
    radii = np.arange(min_radius, max_radius + 1)
    hough_result = hough_circle(edges, radii)

    if hough_result.size and np.max(hough_result) > 0:
        strengths, centre_x, centre_y, detected_radii = hough_circle_peaks(
            hough_result,
            radii,
            min_xdistance=min_xdistance,
            min_ydistance=min_ydistance,
            threshold=threshold_relative * np.max(hough_result),
            normalize=normalize,
        )
    else:
        strengths = _empty_values()
        centre_x = _empty_values()
        centre_y = _empty_values()
        detected_radii = _empty_values()

    centres = np.column_stack([centre_y, centre_x]) if len(centre_x) else _empty_centres()
    return build_detection_result(image_normalised, image, centres, detected_radii, minimum_height=minimum_height, extra={"strengths": strengths, "edges": edges, "hough_result": hough_result})


def peronamalik_anisotropic_diffusion_detect(
    striped_image,
    destriped_image,
    destriped=True,
    smoothing=True,
    low_sigma=5,
    high_sigma=40,
    n_iterations=15,
    kappa=0.4,
    gamma=0.15,
    option=1,
    min_sigma=2,
    max_sigma=8,
    num_sigma=20,
    threshold=0.4,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    rows, columns = image.shape
    frequency_y = np.fft.fftfreq(rows)
    frequency_x = np.fft.fftfreq(columns)
    frequency_x_grid, frequency_y_grid = np.meshgrid(frequency_x, frequency_y)
    frequency_squared = frequency_x_grid**2 + frequency_y_grid**2
    small_scale_low_pass = np.exp(-2 * np.pi**2 * low_sigma**2 * frequency_squared)
    large_scale_low_pass = np.exp(-2 * np.pi**2 * high_sigma**2 * frequency_squared)
    bandpass_filter = small_scale_low_pass - large_scale_low_pass
    image_filtered = np.real(np.fft.ifft2(np.fft.fft2(image) * bandpass_filter))
    image_standardised = (image_filtered - np.mean(image_filtered)) / np.std(image_filtered)
    processed_image = image_standardised.copy()

    if smoothing:
        for _ in range(n_iterations):
            north = np.zeros_like(processed_image)
            south = np.zeros_like(processed_image)
            east = np.zeros_like(processed_image)
            west = np.zeros_like(processed_image)
            north[:-1, :] = processed_image[1:, :] - processed_image[:-1, :]
            south[1:, :] = processed_image[:-1, :] - processed_image[1:, :]
            east[:, :-1] = processed_image[:, 1:] - processed_image[:, :-1]
            west[:, 1:] = processed_image[:, :-1] - processed_image[:, 1:]

            if option == 1:
                c_north = np.exp(-(north / kappa) ** 2)
                c_south = np.exp(-(south / kappa) ** 2)
                c_east = np.exp(-(east / kappa) ** 2)
                c_west = np.exp(-(west / kappa) ** 2)
            elif option == 2:
                c_north = 1 / (1 + (north / kappa) ** 2)
                c_south = 1 / (1 + (south / kappa) ** 2)
                c_east = 1 / (1 + (east / kappa) ** 2)
                c_west = 1 / (1 + (west / kappa) ** 2)
            else:
                raise ValueError("option must be 1 or 2")

            processed_image += gamma * (c_north * north + c_south * south + c_east * east + c_west * west)

    blobs = blob_log(processed_image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    centres = blobs[:, :2] if len(blobs) else _empty_centres()
    radii = np.sqrt(2) * blobs[:, 2] if len(blobs) else _empty_values()
    return build_detection_result(processed_image, image, centres, radii, minimum_height=minimum_height, extra={"raw_blobs": blobs, "filtered_image": image_filtered})


def phase_symmetry_mono(image, nscale=6, min_wavelength=1.5, mult=6.0, sigma_on_f=0.7, noise_k=2.5, polarity=1):
    image = np.asarray(image, dtype=np.float64)
    rows, cols = image.shape
    epsilon = np.finfo(np.float64).eps
    image_fft = np.fft.fft2(image)
    fx = np.fft.fftfreq(cols)
    fy = np.fft.fftfreq(rows)
    u, v = np.meshgrid(fx, fy)
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0
    riesz_x = 1j * u / radius
    riesz_y = 1j * v / radius
    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 20)
    total_energy = np.zeros_like(image)
    sum_amplitude = np.zeros_like(image)
    smallest_scale_amplitude = None
    denominator = 2.0 * np.log(sigma_on_f) ** 2

    for scale in range(nscale):
        wavelength = min_wavelength * mult**scale
        centre_frequency = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / centre_frequency) ** 2) / denominator)
        log_gabor *= lowpass
        log_gabor[0, 0] = 0.0
        filtered_fft = image_fft * log_gabor
        even_response = np.real(np.fft.ifft2(filtered_fft))
        odd_x = np.real(np.fft.ifft2(filtered_fft * riesz_x))
        odd_y = np.real(np.fft.ifft2(filtered_fft * riesz_y))
        odd_amplitude = np.sqrt(odd_x**2 + odd_y**2)
        amplitude = np.sqrt(even_response**2 + odd_amplitude**2)
        sum_amplitude += amplitude

        if polarity == 1:
            scale_energy = even_response - odd_amplitude
        elif polarity == -1:
            scale_energy = -even_response - odd_amplitude
        elif polarity == 0:
            scale_energy = np.abs(even_response) - odd_amplitude
        else:
            raise ValueError("polarity must be -1, 0 or 1")

        total_energy += scale_energy
        if scale == 0:
            smallest_scale_amplitude = amplitude.copy()

    tau = np.median(smallest_scale_amplitude) / np.sqrt(np.log(4.0))
    total_tau = tau * nscale if np.isclose(mult, 1.0) else tau * (1.0 - (1.0 / mult) ** nscale) / (1.0 - 1.0 / mult)
    noise_mean = total_tau * np.sqrt(np.pi / 2.0)
    noise_std = total_tau * np.sqrt((4.0 - np.pi) / 2.0)
    noise_threshold = noise_mean + noise_k * noise_std
    phase_symmetry = np.maximum(total_energy - noise_threshold, 0.0)
    phase_symmetry /= sum_amplitude + epsilon
    return phase_symmetry, total_energy, noise_threshold


def detect_qds_phase_symmetry(
    striped_image,
    destriped_image,
    destriped=True,
    nscale=6,
    min_wavelength=1.5,
    mult=9.0,
    noise_k=2.5,
    symmetry_threshold=0.08,
    min_distance=3,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    image_normalised = (image - np.mean(image)) / np.std(image)
    phase_map, total_energy, noise_threshold = phase_symmetry_mono(image_normalised, nscale=nscale, min_wavelength=min_wavelength, mult=mult, sigma_on_f=0.7, noise_k=noise_k, polarity=1)
    phase_map = gaussian_filter(phase_map, sigma=0.7)
    centres = peak_local_max(phase_map, min_distance=min_distance, threshold_abs=symmetry_threshold, exclude_border=False)
    radii = np.full(len(centres), np.nan)
    return build_detection_result(phase_map, image, centres, radii, minimum_height=minimum_height, extra={"total_energy": total_energy, "noise_threshold": noise_threshold})


def sift_detect(
    striped_image,
    destriped_image,
    destriped=True,
    background_sigma=10,
    qd_min_sigma=0.8,
    qd_max_sigma=4.0,
    upsampling=2,
    n_octaves=3,
    n_scales=5,
    sigma_min=2,
    sigma_in=0.5,
    c_dog=0.05,
    c_edge=15,
    lower_percentile=1,
    upper_percentile=99,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=np.float64)
    qd_signal = image - gaussian_filter(image, sigma=background_sigma)
    finite_values = qd_signal[np.isfinite(qd_signal)]
    if finite_values.size == 0:
        raise ValueError("The selected image contains no finite values.")

    low, high = np.percentile(finite_values, [lower_percentile, upper_percentile])
    if high <= low:
        raise ValueError("The selected image has insufficient intensity variation.")

    processed_image = np.clip((qd_signal - low) / (high - low), 0, 1).astype(np.float32)
    detector = SIFT(upsampling=upsampling, n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min, sigma_in=sigma_in, c_dog=c_dog, c_edge=c_edge)

    try:
        detector.detect_and_extract(processed_image)
        scale_mask = (detector.sigmas >= qd_min_sigma) & (detector.sigmas <= qd_max_sigma)
        centres = detector.positions[scale_mask]
        radii = np.sqrt(2) * detector.sigmas[scale_mask]
        descriptors = detector.descriptors[scale_mask]
    except RuntimeError:
        centres = _empty_centres()
        radii = _empty_values()
        descriptors = np.empty((0, 128), dtype=np.uint8)

    return build_detection_result(processed_image, image, centres, radii, minimum_height=minimum_height, extra={"descriptors": descriptors})



def regions_to_contiguous_label_mask(shape, regions):
    filtered_labels = np.zeros(shape, dtype=np.int32)

    for new_label, region in enumerate(regions, start=1):
        filtered_labels[region.coords[:, 0], region.coords[:, 1]] = new_label

    return filtered_labels



def geometric_contour_detect(
    striped_image,
    destriped_image,
    striped_centres,
    destriped_centres,
    destriped=True,
    initial_radius=3,
    alpha=100,
    gradient_sigma=1.0,
    num_iter=80,
    smoothing=1,
    balloon=-1,
    threshold="auto",
    min_area=1,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    centres = _normalise_centres(destriped_centres if destriped else striped_centres)
    initial_mask = circles_to_mask(image.shape, centres, np.full(len(centres), initial_radius))
    processed_image = inverse_gaussian_gradient(image, alpha=alpha, sigma=gradient_sigma)
    contour_mask = morphological_geodesic_active_contour(processed_image, num_iter=num_iter, init_level_set=initial_mask, smoothing=smoothing, balloon=balloon, threshold=threshold)
    labelled_mask = label(contour_mask)
    regions = [region for region in regionprops(labelled_mask) if region.area >= min_area]
    filtered_labels = regions_to_contiguous_label_mask(image.shape, regions)
    centres = np.asarray([region.centroid for region in regions], dtype=float).reshape(-1, 2) if regions else _empty_centres()
    areas = np.asarray([region.area for region in regions], dtype=float)
    radii = np.sqrt(areas / np.pi)
    return build_detection_result(processed_image, image, centres, radii, areas, candidate_label_mask=filtered_labels, minimum_height=minimum_height, extra={"initial_mask": initial_mask, "contour_mask": contour_mask})


def chan_vese_detect(
    striped_image,
    destriped_image,
    striped_centres,
    destriped_centres,
    destriped=True,
    initial_radius=3,
    num_iter=80,
    smoothing=1,
    lambda1=1,
    lambda2=2,
    min_area=1,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    centres = _normalise_centres(destriped_centres if destriped else striped_centres)
    initial_mask = circles_to_mask(image.shape, centres, np.full(len(centres), initial_radius))
    contour_mask = morphological_chan_vese(image, num_iter=num_iter, init_level_set=initial_mask, smoothing=smoothing, lambda1=lambda1, lambda2=lambda2)
    labelled_mask = label(contour_mask)
    regions = [region for region in regionprops(labelled_mask) if region.area >= min_area]
    centres = np.asarray([region.centroid for region in regions], dtype=float).reshape(-1, 2) if regions else _empty_centres()
    areas = np.asarray([region.area for region in regions], dtype=float)
    radii = np.sqrt(areas / np.pi)
    return build_detection_result(image, image, centres, radii, areas, candidate_label_mask=labelled_mask, minimum_height=minimum_height, extra={"initial_mask": initial_mask, "contour_mask": contour_mask})


def watershed_detect(
    striped_image,
    destriped_image,
    destriped=True,
    footprint_radius=6,
    min_size=1,
    min_distance=3,
    min_area=5,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    processed_image = white_tophat(image, footprint=morphology_disk(footprint_radius))
    threshold = threshold_otsu(processed_image)
    qd_mask = remove_small_objects(processed_image > threshold, min_size=min_size)
    distance = distance_transform_edt(qd_mask)
    peak_coordinates = peak_local_max(distance, min_distance=min_distance, labels=qd_mask, exclude_border=False)
    markers = np.zeros_like(distance, dtype=np.int32)

    for marker_number, (row, column) in enumerate(peak_coordinates, start=1):
        markers[row, column] = marker_number

    watershed_labels = watershed(-distance, markers, mask=qd_mask)
    filtered_labels = np.zeros_like(watershed_labels)
    regions = []
    next_label = 1

    for region in regionprops(watershed_labels, intensity_image=image):
        min_row, min_col, max_row, max_col = region.bbox
        touches_edge = min_row == 0 or min_col == 0 or max_row == image.shape[0] or max_col == image.shape[1]
        if region.area < min_area or touches_edge:
            continue
        filtered_labels[watershed_labels == region.label] = next_label
        regions.append(region)
        next_label += 1

    centres = np.asarray([region.centroid for region in regions], dtype=float).reshape(-1, 2) if regions else _empty_centres()
    areas = np.asarray([region.area for region in regions], dtype=float)
    radii = np.sqrt(areas / np.pi)
    return build_detection_result(processed_image, image, centres, radii, areas, candidate_label_mask=filtered_labels, minimum_height=minimum_height, extra={"qd_mask": qd_mask, "distance": distance, "markers": markers, "watershed_labels": filtered_labels, "threshold": threshold})


def gabor_detect(
    striped_image,
    destriped_image,
    destriped=True,
    smoothing_sigma=0.58,
    top_hat_radius=6,
    wavelengths=(2, 3, 4, 5, 6, 8),
    number_of_orientations=8,
    bandwidth=1.0,
    threshold_percentile=90,
    min_distance=2,
    minimum_height=6e-10,
):
    image = np.asarray(destriped_image if destriped else striped_image, dtype=float)
    median_height = np.nanmedian(image)
    image = np.nan_to_num(image, nan=median_height)
    image_nobackground = image - np.mean(image)
    image_smooth = gaussian_filter(image_nobackground, sigma=smoothing_sigma)
    image_normalised = rescale_intensity(image_smooth, in_range="image", out_range=(0.0, 1.0))
    small_feature_image = white_tophat(image_normalised, footprint=morphology_disk(top_hat_radius))
    small_feature_image = rescale_intensity(small_feature_image, in_range="image", out_range=(0.0, 1.0))
    orientations = np.linspace(0, np.pi, number_of_orientations, endpoint=False)
    combined_response = np.zeros_like(small_feature_image)
    response_maps = {}

    for wavelength in wavelengths:
        frequency = 1.0 / wavelength
        for theta in orientations:
            real_response, imaginary_response = gabor(small_feature_image, frequency=frequency, theta=theta, bandwidth=bandwidth)
            magnitude = np.sqrt(real_response**2 + imaginary_response**2)
            combined_response = np.maximum(combined_response, magnitude)
            if wavelength in (3, 5, 8) and np.isclose(theta, 0):
                response_maps[(wavelength, 0)] = magnitude

    threshold = np.percentile(combined_response, threshold_percentile)
    centres = peak_local_max(combined_response, min_distance=min_distance, threshold_abs=threshold, exclude_border=False)
    radii = np.full(len(centres), np.nan)
    return build_detection_result(combined_response, image, centres, radii, minimum_height=minimum_height, extra={"response_maps": response_maps, "threshold": threshold, "small_feature_image": small_feature_image})



def make_results_row(algorithm_name, stage_name, predicted_result, true_result):
    metrics = evaluate_detection(
        predicted_result=predicted_result,
        true_result=true_result,
        max_distance=5,
    )

    return {
        "Algorithm": algorithm_name,
        "Stage": stage_name,

        "Predicted count": len(predicted_result["centres"]),
        "True count": len(true_result["centres"]),
        "Count error": metrics["count_error"],
        "Absolute count error": metrics["absolute_count_error"],

        "Pixel precision": metrics["pixel_precision"],
        "Pixel recall": metrics["pixel_recall"],
        "Pixel F1": metrics["pixel_f1"],
        "Dice": metrics["dice"],
        "IoU": metrics["iou"],

        "Object precision": metrics["object_precision"],
        "Object recall": metrics["object_recall"],
        "Object F1": metrics["object_f1"],

        "Localisation mean (px)": metrics["localisation"]["mean"],
        "Localisation median (px)": metrics["localisation"]["median"],
        "Localisation RMSE (px)": metrics["localisation"]["rmse"],
        "Localisation maximum (px)": metrics["localisation"]["maximum"],

        "Height MAE": metrics["parameters"]["height"]["mae"],
        "Height RMSE": metrics["parameters"]["height"]["rmse"],

        "Radius MAE": metrics["parameters"]["radius"]["mae"],
        "Radius RMSE": metrics["parameters"]["radius"]["rmse"],

        "Area MAE": metrics["parameters"]["area"]["mae"],
        "Area RMSE": metrics["parameters"]["area"]["rmse"],
    }