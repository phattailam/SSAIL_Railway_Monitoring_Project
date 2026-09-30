import queue
import threading

from pathlib import Path

import requests

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst

from samples.railway_monitoring.vlm_client_utils import (
    collect_detections,
    save_detection_result,
    select_vlm_candidate,
    get_frame_and_jpeg,
    save_detection_image,
    save_vlm_result,
)


class VLMClient(NvDsPyFuncPlugin):

    def __init__(self, vlm_api_url="http://host.docker.internal:8000/analyze",**kwargs):
        super().__init__(**kwargs)

        self.vlm_api_url = (vlm_api_url)
        self.results_root = Path("/opt/savant/samples/railway_monitoring/results")
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.vlm_queue = (queue.Queue(maxsize=1))
        self.worker = (threading.Thread(target=self._vlm_worker, daemon=True))
        self.worker.start()


    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):

        source_id = (frame_meta.source_id)

        # ----------------------------------
        # Detection results
        # ----------------------------------

        detections = (collect_detections(frame_meta))

        if not detections:
            return

        save_detection_result(self.results_root, source_id, detections)

        frame_bgr, image_bytes = (get_frame_and_jpeg(buffer, frame_meta))

        save_detection_image(
            self.results_root,
            source_id,
            frame_bgr,
            detections,
            frame_name=f"frame_{frame_meta.frame_num:06d}",
        )

        # ----------------------------------
        # Select candidate for VLM
        # ----------------------------------

        detection = (select_vlm_candidate(detections))

        if detection is None:
            return

        if image_bytes is None:
            self.logger.error("Failed to encode frame as JPEG.")
            return

        # ----------------------------------
        # Add VLM job
        # ----------------------------------

        job = {
            "source_id": source_id,
            "image_bytes": image_bytes,
            "detection": detection,
        }

        try:
            self.vlm_queue.put_nowait(job)

        except queue.Full:
            pass


    def _vlm_worker(
        self,
    ):

        while True:

            job = (self.vlm_queue.get())

            try:
                source_id = (job["source_id"])

                detection = (job["detection"])

                files = {
                    "image": (
                        "frame.jpg",
                        job["image_bytes"],
                        "image/jpeg",
                    )
                }

                data = {
                    "detector_name":
                        detection[
                            "detector_name"
                        ],

                    "class_name":
                        detection[
                            "class_name"
                        ],

                    "confidence":
                        detection[
                            "confidence"
                        ],
                }

                response = (
                    requests.post(
                        self.vlm_api_url,
                        files=files,
                        data=data,
                        timeout=30,
                    )
                )

                response.raise_for_status()

                result = (response.json())

                save_vlm_result(self.results_root, source_id, result)

                print(
                    "[VLM RESULT]",
                    source_id,
                    result,
                )

            except Exception as exc:

                self.logger.error(
                    f"VLM request failed: "
                    f"{exc}"
                )

            finally:
                self.vlm_queue.task_done()