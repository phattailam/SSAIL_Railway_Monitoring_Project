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


PER_CAMERA_PENDING_MAXSIZE = 1
CONSECUTIVE_POSITIVE_FRAMES = 5


class VLMClient(NvDsPyFuncPlugin):

    def __init__(self, vlm_api_url="http://host.docker.internal:8000/analyze",**kwargs):
        super().__init__(**kwargs)

        # General configuration
        self.vlm_api_url = (vlm_api_url)
        self.results_root = Path("samples/railway_monitoring/results")
        self.results_root.mkdir(parents=True, exist_ok=True)

        # Per-camera temporal state
        self.source_states = {}

        # Per-camera pending VLM jobs
        self.pending_vlm_jobs = {}

        # Stores source IDs that currently have a job ready
        self.ready_sources = queue.Queue()

        # Prevent duplicate source IDs in ready_sources
        self.ready_sources_set = set()
        self.pending_lock = threading.Lock()

        # Single shared VLM worker
        self.worker = (threading.Thread(target=self._vlm_worker, daemon=True))
        self.worker.start()


    # ======================================================
    # Source state
    # ======================================================

    def _get_source_state(self, source_id):
        if source_id not in self.source_states:

            self.source_states[source_id] = {
                "positive_count": 0,
                "vlm_triggered": False,
                "vlm_active": False,
            }

        return self.source_states[source_id]


    def _reset_source_state(self, state):
        state["positive_count"] = 0
        state["vlm_triggered"] = False


    # ======================================================
    # Per-camera VLM queue
    # ======================================================

    def _get_camera_queue(self, source_id):
        with self.pending_lock:

            if source_id not in self.pending_vlm_jobs:

                self.pending_vlm_jobs[source_id] = (
                    queue.Queue(
                        maxsize=PER_CAMERA_PENDING_MAXSIZE
                    )
                )

            return self.pending_vlm_jobs[source_id]


    def _enqueue_vlm_job(self, source_id, job):
        
        camera_queue = self._get_camera_queue(
            source_id
        )

        try:
            camera_queue.put_nowait(job)

        except queue.Full:

            print(
                "[VLM PENDING]",
                source_id,
                "pending queue full",
            )

            return False

        # Tell the worker that this camera
        # has a job ready.
        with self.pending_lock:

            if source_id not in self.ready_sources_set:

                self.ready_sources.put(
                    source_id
                )

                self.ready_sources_set.add(
                    source_id
                )

        print(
            "[VLM PENDING]",
            source_id,
            "job stored",
        )

        return True


    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):

        source_id = frame_meta.source_id

        state = self._get_source_state(
            source_id
        )

        # --------------------------------------------------
        # 1. Collect YOLO detections
        # --------------------------------------------------

        detections = (collect_detections(frame_meta))

        # No detection at all.
        # This breaks the consecutive sequence.
        if not detections:

            self._reset_source_state(
                state
            )

            return


        # --------------------------------------------------
        # 2. Save detection metadata
        # --------------------------------------------------

        save_detection_result(
            self.results_root,
            source_id,
            detections,
        )
        
        # --------------------------------------------------
        # 3. Save annotated detection image
        # --------------------------------------------------

        frame_bgr, image_bytes = (
            get_frame_and_jpeg(
                buffer,
                frame_meta,
            )
        )

        frame_name = f"frame_{frame_meta.frame_num:06d}.jpg"

        if frame_bgr is not None:
            save_detection_image(
                self.results_root,
                source_id,
                frame_bgr,
                detections,
                frame_name=(frame_name),
            )

        # --------------------------------------------------
        # 4. Find slope-failure candidate
        # --------------------------------------------------

        detection = select_vlm_candidate(
            detections
        )

        # Other objects may exist, but no slope failure.
        if detection is None:
            self._reset_source_state(
                state
            )

            return

        # --------------------------------------------------
        # 5. Temporal confirmation
        # --------------------------------------------------
        
        state["positive_count"] = min(
            state["positive_count"] + 1,
            CONSECUTIVE_POSITIVE_FRAMES,
        )

        print(
            "[TEMPORAL]",
            source_id,
            f'positive_count={state["positive_count"]}/'
            f'{CONSECUTIVE_POSITIVE_FRAMES}',
        )

        # Need 5 consecutive positive processed frames.
        if (state["positive_count"] < CONSECUTIVE_POSITIVE_FRAMES):
            return

        # Current incident was already submitted.
        if state["vlm_triggered"]:
            return

        if state["vlm_active"]:
            print("[VLM]", source_id, "previous VLM job still active")
            return

        # --------------------------------------------------
        # 6. Prepare VLM job
        # --------------------------------------------------

        if image_bytes is None:

            self.logger.error(
                "Failed to encode frame as JPEG."
            )

            return



        job = {
            "source_id": source_id,
            "frame_name": frame_name,
            "image_bytes": image_bytes,
            "detection": detection,
        }

        # --------------------------------------------------
        # 7. Store one pending job for this camera
        # --------------------------------------------------

        queued = self._enqueue_vlm_job(
            source_id,
            job,
        )

        # Important:
        # Mark triggered only if job was actually stored.
        if queued:
            state["vlm_triggered"] = True
            state["vlm_active"] = True


    # ======================================================
    # Shared VLM worker
    # ======================================================

    def _vlm_worker(self):

        while True:

            # Wait until some camera has a pending job.
            source_id = self.ready_sources.get()

            camera_queue = self._get_camera_queue(
                source_id
            )

            job = camera_queue.get()

            try:

                self._process_vlm_job(job)

            except Exception as exc:

                self.logger.error(
                    f"VLM request failed: {exc}"
                )

            finally:

                camera_queue.task_done()

                state = self._get_source_state(source_id)
                state["vlm_active"] = False

                self._finish_camera_job(
                    source_id,
                    camera_queue,
                )

                self.ready_sources.task_done()


    # ======================================================
    # VLM HTTP request
    # ======================================================

    def _process_vlm_job(self, job):

        source_id = job["source_id"]
        detection = job["detection"]

        files = {
            "image": (
                job["frame_name"],
                job["image_bytes"],
                "image/jpeg",
            )
        }

        data = {
            "detector_name":
                detection["detector_name"],

            "class_name":
                detection["class_name"],

            "confidence":
                detection["confidence"],
        }

        response = requests.post(
            self.vlm_api_url,
            files=files,
            data=data,
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        save_vlm_result(
            self.results_root,
            source_id,
            result,
        )

        print(
            "[VLM RESULT]",
            source_id,
            result,
        )


    # ======================================================
    # Finish scheduling
    # ======================================================

    def _finish_camera_job(self, source_id, camera_queue):

        with self.pending_lock:

            self.ready_sources_set.discard(
                source_id
            )

            # Normally queue size is 0 because
            # maxsize=1 and one incident is submitted.
            # Keep this for future extensibility.
            if not camera_queue.empty():

                self.ready_sources.put(
                    source_id
                )

                self.ready_sources_set.add(
                    source_id
                )