import os
import torch
import numpy as np
import cv2

from src.utils.label_info import (
    coco_color_array, coco_color_array_, coco_label_array,
    coco_ids_2_labels, coco_ids_2_cont_ids,
    voc_color_array, voc_label_array
)
from detectron2.structures import Instances


def visualize_instances_d2(
    image: torch.Tensor,
    instances: Instances,
    data_type: str = "coco",
    num_labels: int = 91,
    save_name: str = None,
    to_bgr: bool = True,
    save: bool = False,
):
    """
    Visualize Detectron2 Instances (GT or predicted).
    Args:
        image: Tensor [3, H, W] in [0,255] or [0,1]
        instances: detectron2.structures.Instances (with gt_boxes, gt_classes)
        data_type: 'coco' or 'voc'
        num_labels: label count (91 for COCO)
        name: filename to save under ./demo_results/
        to_bgr: whether to convert RGB→BGR before saving
        save: whether to save file to disk
    """

    # ----------------------------------------------------------------------
    # Convert image tensor → numpy
    # ----------------------------------------------------------------------
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # [C, H, W] → [H, W, C]
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]

    # copy for visualization
    image_vis = image.copy()

    if to_bgr:
        # convert to OpenCV color space
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    # ----------------------------------------------------------------------
    # Extract boxes / labels / scores
    # ----------------------------------------------------------------------
    boxes = instances.gt_boxes.tensor.detach().cpu().numpy().astype(int) if instances.has("gt_boxes") else None
    labels = instances.gt_classes.detach().cpu().numpy() if instances.has("gt_classes") else None
    scores = instances.scores.detach().cpu().numpy() if instances.has("scores") else None

    if boxes is None or labels is None:
        print("[visualize_instances_d2] ⚠️ No boxes or labels found in Instances.")
        return

    # ----------------------------------------------------------------------
    # Label / color mapping
    # ----------------------------------------------------------------------
    if data_type == "voc":
        label_arr = voc_label_array
        color_arr = (voc_color_array * 255).astype(np.uint8)
        label_dict = None
    elif data_type == "coco":
        label_arr = coco_label_array
        color_arr = (coco_color_array_ * 255).astype(np.uint8) if num_labels == 91 else (coco_color_array * 255).astype(np.uint8)
        label_dict = coco_ids_2_labels if num_labels == 91 else None
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    # Reverse map for contiguous → COCO ids
    cont2coco = {v: k for k, v in coco_ids_2_cont_ids.items()}

    # ----------------------------------------------------------------------
    # Draw boxes and labels
    # ----------------------------------------------------------------------
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]

        color = tuple(int(c) for c in color_arr[labels[i] % len(color_arr)])

        # label name
        if label_dict is not None:
            coco_id = cont2coco.get(int(labels[i]), -1)
            label_name = label_dict.get(coco_id, f"id_{int(labels[i])}")
        else:
            label_name = str(label_arr[labels[i]]) if labels[i] < len(label_arr) else f"id_{int(labels[i])}"

        score_str = f": {scores[i]:.2f}" if scores is not None else ""
        text = f"{label_name}{score_str}"

        # draw rectangle
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)

        # label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image_vis, (x1, max(y1 - th - 4, 0)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(image_vis, text, (x1 + 2, max(y1 - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # ----------------------------------------------------------------------
    # Save / Show
    # ----------------------------------------------------------------------
    os.makedirs("./demo_results", exist_ok=True)
    save_path = f"./demo_results/{save_name or 'detectron2_instances.jpg'}"

    if save:
        cv2.imwrite(save_path, image_vis)
        print(f"[visualize_instances_d2] ✅ Saved: {save_path}")
    else:
        cv2.imshow("instances", image_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_vis
