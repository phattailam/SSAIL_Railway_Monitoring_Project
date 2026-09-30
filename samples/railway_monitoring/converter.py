import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.converter.yolo import compute_scale_and_pad


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """Convert YOLOv26 output [x1, y1, x2, y2, conf, class_id] to Savant bbox format."""

    def __init__(self, confidence_threshold: float = 0.25):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi,
    ):
        output = output_layers[0]

        # Remove batch dimension: [1, 300, 6] -> [300, 6]
        if output.ndim == 3:
            output = output[0]

        # YOLOv26 output:
        # [x1, y1, x2, y2, confidence, class_id]
        bboxes = output[:, :4].copy()
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(np.int32)

        # Filter detections by confidence
        mask = confidences >= self.confidence_threshold
        bboxes = bboxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(bboxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Convert xyxy -> xywh(center)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2

        # Scale coordinates from model input to original ROI/frame
        (scale_x, scale_y), (pad_x, pad_y) = compute_scale_and_pad(
            roi,
            model.input.width,
            model.input.height,
            model.input.maintain_aspect_ratio,
            model.input.symmetric_padding,
        )

        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        bboxes[:, 0] += pad_x
        bboxes[:, 1] += pad_y

        # Savant expected output:
        # [class_id, confidence, xc, yc, width, height]
        return np.concatenate(
            (
                class_ids.reshape(-1, 1).astype(np.float32),
                confidences.reshape(-1, 1).astype(np.float32),
                bboxes.astype(np.float32),
            ),
            axis=1,
        )
