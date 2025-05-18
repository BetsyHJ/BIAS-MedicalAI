import numpy as np

def bbox_to_mask(bbox, cam_shape, original_size):
    """
    Converts a bounding box from the original image size to a binary mask 
    of the same shape as the Grad-CAM heatmap.

    Args:
        bbox (tuple): The bounding box, either in (x, y, w, h) or (x1, y1, x2, y2) format,
                      defined on the original image scale.
        cam_shape (tuple): Shape of the Grad-CAM heatmap, e.g., (224, 224).
        original_size (tuple): Original image size, e.g., (1024, 1024).

    Returns:
        np.ndarray: A binary mask with shape cam_shape, with 1s inside the box area.
    """
    # Initialize an empty mask
    mask = np.zeros(cam_shape, dtype=np.uint8)

    # Unpack original and CAM dimensions
    H_orig, W_orig = original_size
    H_cam, W_cam = cam_shape

    # Compute scaling factors from original size to CAM size
    scale_x = W_cam / W_orig
    scale_y = H_cam / H_orig

    # Convert to (x1, y1, x2, y2) in CAM coordinates
    # Assume (x, y, w, h) format
    x, y, w, h = bbox
    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + w) * scale_x)
    y2 = int((y + h) * scale_y)

    # Clip coordinates to stay within CAM boundaries
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W_cam), min(y2, H_cam)

    # Set the region inside the bbox to 1
    mask[y1:y2, x1:x2] = 1
    return mask


def normalize_cam(cam):
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def pointing_game(cam, gt_mask):
    """
    Pointing Game metric.

    Args:
        cam (np.ndarray): Grad-CAM heatmap, shape (H, W).
        gt_mask (np.ndarray): Ground-truth binary mask, shape (H, W).

    Returns:
        bool: True if max point falls inside gt_mask, else False.
    """
    # Find the index of the maximum value in the CAM
    max_idx = np.unravel_index(np.argmax(cam), cam.shape)
    
    # Check whether it lies inside the ground truth mask
    return gt_mask[max_idx] == 1

def compute_iou(cam, gt_mask, threshold=0.5):
    """
    Compute IoU (Intersection over Union) between binarized CAM and ground truth.

    Args:
        cam (np.ndarray): Grad-CAM heatmap, shape (H, W), values in [0, 1].
        gt_mask (np.ndarray): Ground-truth binary mask, shape (H, W).
        threshold (float): Threshold for binarizing CAM.

    Returns:
        float: IoU score.
    """
    cam_binary = (cam >= threshold).astype(np.uint8)
    
    intersection = np.logical_and(cam_binary, gt_mask).sum()
    union = np.logical_or(cam_binary, gt_mask).sum()

    if union == 0:
        return float(intersection == 0)  # 1 if both are empty, else 0

    return intersection / union

def compute_dice(cam, gt_mask, threshold=0.5):
    """
    Compute Dice coefficient between binarized CAM and ground truth.

    Args:
        cam (np.ndarray): Grad-CAM heatmap, shape (H, W), values in [0, 1].
        gt_mask (np.ndarray): Ground-truth binary mask, shape (H, W).
        threshold (float): Threshold for binarizing CAM.

    Returns:
        float: Dice score.
    """
    cam_binary = (cam >= threshold).astype(np.uint8)
    
    intersection = np.logical_and(cam_binary, gt_mask).sum()
    size_sum = cam_binary.sum() + gt_mask.sum()

    if size_sum == 0:
        return float(intersection == 0)  # 1 if both are empty, else 0

    return 2 * intersection / size_sum
