import json

from datetime import datetime, timezone
from pathlib import Path

import cv2

from savant.deepstream.opencv_utils import nvds_to_gpu_mat


ALL_DETECTORS = {
    "slope_soil_failure_detector",
    "foreign_object_detector",
    "fire_detector",
}


VLM_TRIGGER_DETECTORS = {
    "slope_soil_failure_detector",
}


def append_jsonl(
    path,
    record,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def collect_detections(
    frame_meta,
):
    detections = []

    for obj_meta in frame_meta.objects:

        if obj_meta.element_name not in ALL_DETECTORS:
            continue

        bbox = obj_meta.bbox

        detections.append(
            {
                "detector_name":
                    obj_meta.element_name,

                "class_name":
                    obj_meta.label,

                "confidence":
                    float(
                        obj_meta.confidence
                    ),

                "bbox": {
                    "xc":
                        float(bbox.xc),

                    "yc":
                        float(bbox.yc),

                    "width":
                        float(bbox.width),

                    "height":
                        float(bbox.height),
                },
            }
        )

    return detections


def save_detection_image(
    results_root,
    source_id,
    frame_bgr,
    detections,
    frame_name,
):
    if frame_bgr is None:
        return

    output_dir = (
        Path(results_root)
        / source_id
        / "detection_images"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = frame_bgr.copy()

    for detection in detections:

        bbox = detection["bbox"]

        xc = bbox["xc"]
        yc = bbox["yc"]
        width = bbox["width"]
        height = bbox["height"]

        x1 = int(xc - width / 2)
        y1 = int(yc - height / 2)
        x2 = int(xc + width / 2)
        y2 = int(yc + height / 2)

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = (
        output_dir
        / f"{frame_name}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        image,
    )


def save_detection_result(
    results_root,
    source_id,
    detections,
):
    if not detections:
        return

    record = {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source_id":
            source_id,

        "detections":
            detections,
    }

    append_jsonl(
        Path(results_root)
        / source_id
        / "detections.jsonl",

        record,
    )


def select_vlm_candidate(
    detections,
):
    candidates = [
        detection
        for detection in detections
        if (
            detection["detector_name"]
            in VLM_TRIGGER_DETECTORS
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            item["confidence"],
    )


def get_frame_and_jpeg(
    buffer,
    frame_meta,
):
    with nvds_to_gpu_mat(
        buffer,
        frame_meta.frame_meta,
    ) as frame_mat:

        cpu_frame = (
            frame_mat.download()
        )

    frame_bgr = cv2.cvtColor(
        cpu_frame,
        cv2.COLOR_RGBA2BGR,
    )

    success, encoded_image = (
        cv2.imencode(
            ".jpg",
            frame_bgr,
        )
    )

    if not success:
        return None, None

    return (
        frame_bgr,
        encoded_image.tobytes(),
    )


def save_vlm_result(
    results_root,
    source_id,
    result,
):
    record = {
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "source_id":
            source_id,

        "result":
            result,
    }

    append_jsonl(
        Path(results_root)
        / source_id
        / "vlm_results.jsonl",

        record,
    )