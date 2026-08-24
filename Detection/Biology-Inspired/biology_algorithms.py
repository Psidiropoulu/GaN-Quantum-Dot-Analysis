import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from skimage.measure import regionprops

from stardist.models import StarDist2D
from cellpose import models as cellpose_models
from cellpose_omni import models as omnipose_models
from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )



def percentile_normalise(image, low=1, high=99):
    """
    Robustly rescale an AFM image to [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    lo, hi = np.nanpercentile(image, [low, high])

    if hi <= lo:
        return np.zeros_like(image)

    image = (image - lo) / (hi - lo)

    return np.clip(image, 0, 1)


def labels_to_qd_result(labels):
    """
    Convert any instance-label image into the common QD format.

    labels:
        0 = background
        1,2,... = individual objects
    """
    labels = np.asarray(labels)

    centres = []
    radii = []
    areas = []

    for region in regionprops(labels.astype(np.int32)):

        y, x = region.centroid
        area = float(region.area)

        # Equivalent-circle radius
        radius = np.sqrt(area / np.pi)

        centres.append((float(x), float(y)))
        radii.append(float(radius))
        areas.append(area)

    return {
        "labels": labels.astype(np.int32),
        "mask": labels > 0,
        "centres": centres,
        "radii": radii,
        "areas": areas,
        "n_qds": len(centres),
    }


def draw_qd_result(
    image,
    result,
    title=None,
):
    """
    Draw circles around every detected instance.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(image, cmap="gray")

    for (x, y), radius in zip(
        result["centres"],
        result["radii"],
    ):
        circle = Circle(
            (x, y),
            radius,
            fill=False,
            linewidth=1.0,
        )

        ax.add_patch(circle)

    if title is None:
        title = f"Detected QDs: {result['n_qds']}"

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.show()



def stardist_detect(
    image,
    prob_thresh=0.5,
    nms_thresh=0.4,
):
    """
    Detect QD-like objects using pretrained StarDist.

    Returns the common QD result dictionary.
    """
    from stardist.models import StarDist2D

    image_n = percentile_normalise(image)

    model = StarDist2D.from_pretrained(
        "2D_versatile_fluo"
    )

    labels, details = model.predict_instances(
        image_n,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    result = labels_to_qd_result(labels)

    result["details"] = details

    return result



def cellpose_detect(
    image,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    min_size=3,
    gpu=False,
):
    """
    Detect QD-like objects using Cellpose-SAM.
    """

    image_n = percentile_normalise(image)

    # Current Cellpose-SAM expects three channels.
    image_rgb = np.stack(
        [image_n, image_n, image_n],
        axis=-1,
    )

    model = cellpose_models.CellposeModel(
        gpu=gpu,
        pretrained_model="cpsam_v2",
    )

    masks, flows, styles = model.eval(
        image_rgb,
        channel_axis=-1,
        normalize=False,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )

    result = labels_to_qd_result(masks)

    result["flows"] = flows

    return result



def omnipose_detect(
    image,
    model_type="bact_fluor_omni",
    mask_threshold=0.4,
    flow_threshold=0.4,
    min_size=3,
    gpu=False,
):
    """
    Detect objects using Omnipose.
    """

    image_n = percentile_normalise(image)

    model = omnipose_models.Cellpose(
        gpu=gpu,
        model_type=model_type,
        nclasses=4,
        nchan=2,
        dim=2,
    )

    masks, flows, styles, diams = model.eval(
        [image_n],
        diameter=None,
        channels=[0, 0],
        threshold=mask_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        omni=True,
    )

    labels = masks[0]

    result = labels_to_qd_result(labels)

    result["flows"] = flows[0]

    return result



def microsam_detect(
    image,
    model_type="vit_b_lm",
    segmentation_mode="apg",
):
    """
    Automatic QD-like instance segmentation using microSAM.
    """

    image_n = percentile_normalise(image)

    # SAM-style models are generally happiest with image-like uint8 data.
    image_uint8 = (
        255 * image_n
    ).astype(np.uint8)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,
        segmentation_mode=segmentation_mode,
        is_tiled=False,
    )

    if segmentation_mode == "apg":

        generate_kwargs = {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "nms_threshold": 0.9,
        }

    elif segmentation_mode == "ais":

        generate_kwargs = {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "foreground_smoothing": 1.0,
            "distance_smoothing": 1.6,
            "min_size": 0,
        }

    elif segmentation_mode == "amg":

        generate_kwargs = {
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "crop_nms_thresh": 0.7,
            "min_mask_region_area": 0,
        }

    else:
        raise ValueError(
            "segmentation_mode must be "
            "'apg', 'ais', or 'amg'."
        )

    labels = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image_uint8,
        ndim=2,
        tile_shape=None,
        halo=None,
        **generate_kwargs,
    )

    return labels_to_qd_result(labels)


def maskrcnn_detect(
    image,
    score_threshold=0.5,
    mask_threshold=0.5,
    gpu=False,
):
    """
    Run pretrained TorchVision Mask R-CNN on an AFM image.

    NOTE:
    Pretrained weights are COCO-trained, not QD-trained.
    This is mainly a pipeline test until the model is fine-tuned.
    """
    import numpy as np
    import torch

    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    )

    image_n = percentile_normalise(image)

    # Mask R-CNN expects [C, H, W] float image in [0,1].
    image_rgb = np.stack(
        [image_n, image_n, image_n],
        axis=0,
    ).astype(np.float32)

    x = torch.from_numpy(image_rgb)

    device = torch.device(
        "mps"
        if gpu and torch.backends.mps.is_available()
        else "cpu"
    )

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn(
        weights=weights
    ).to(device)

    model.eval()

    with torch.no_grad():

        prediction = model(
            [x.to(device)]
        )[0]

    scores = prediction["scores"].detach().cpu().numpy()

    keep = scores >= score_threshold

    masks = (
        prediction["masks"]
        .detach()
        .cpu()
        .numpy()[keep, 0]
    )

    scores = scores[keep]

    H, W = image.shape

    labels = np.zeros(
        (H, W),
        dtype=np.int32,
    )

    # Highest-confidence instances first.
    order = np.argsort(scores)[::-1]

    instance_id = 1

    for idx in order:

        binary = masks[idx] >= mask_threshold

        # Don't overwrite pixels already assigned.
        binary &= labels == 0

        if np.any(binary):

            labels[binary] = instance_id
            instance_id += 1

    result = labels_to_qd_result(labels)

    result["scores"] = scores

    return result