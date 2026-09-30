import io
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

from PIL import Image
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
)

from vlm_service.vlm_model import (
    MASTER_ATTRIBUTES_FILE,
    TARGET_ATTRIBUTE_IDS,
    load_master_attributes,
    select_target_attributes,
    build_attribute_prompt,
    load_qwen_vlm,
    run_description_pass,
    run_attribute_pass,
    extract_json_object,
    validate_prediction,
    build_binary_vector,
)


# ============================================================
# CONFIGURATION
# ============================================================
DECISION_TREE_MODEL_PATH = Path(
    "/home/tailam/SSL/RAIL_PROJECT/Rule_discovery/Experiment_3/stage4C_semantic_classifier_results_400_two_stage_TRAIN_IMPORTANCE_RETRAIN/Feature_Set_B_slope_semantic_only/Decision_Tree_Importance_GT_0/decision_tree_reduced_FINAL_ALL_DATA.joblib"
)


RESIZE_DIVISOR = 3


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Railway VLM Service",
    version="2.0",
)


# ============================================================
# LOAD ONCE AT STARTUP
# ============================================================

vlm_model, vlm_processor = (
    load_qwen_vlm()
)


if not DECISION_TREE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Decision Tree model not found: "
        f"{DECISION_TREE_MODEL_PATH}"
    )

decision_tree = joblib.load(
    DECISION_TREE_MODEL_PATH
)

print("[Decision Tree] Model loaded.")


all_master_attributes = (
    load_master_attributes(
        MASTER_ATTRIBUTES_FILE
    )
)

master_attributes = (
    select_target_attributes(
        all_master_attributes
    )
)

attribute_prompt = (
    build_attribute_prompt(
        master_attributes
    )
)


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image_div3(
    image: Image.Image,
):
    width, height = image.size

    new_width = max(
        1,
        int(
            round(
                width / RESIZE_DIVISOR
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                height / RESIZE_DIVISOR
            )
        ),
    )

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )


def predict_failure(
    binary_vector,
):

    X = pd.DataFrame(
        [binary_vector],
        columns=TARGET_ATTRIBUTE_IDS,
    )

    pred_binary = int(
        decision_tree.predict(X)[0]
    )

    pred_label = (
        "failure"
        if pred_binary == 1
        else "no_failure"
    )

    failure_probability = float(
        decision_tree.predict_proba(X)[0, 1]
    )

    return {
        "pred_binary": pred_binary,
        "pred_label": pred_label,
        "failure_probability":
            failure_probability,
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "railway-vlm",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),

    detector_name: str = Form("unknown"),
    class_name: str = Form("unknown"),
    confidence: float = Form(0.0),
):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_bytes = await image.read()

    pil_image = (
        Image.open(
            io.BytesIO(image_bytes)
        )
        .convert("RGB")
    )

    original_size = pil_image.size

    # --------------------------------------------------------
    # Resize full image exactly as Stage 4A
    # --------------------------------------------------------

    vlm_image = resize_image_div3(
        pil_image
    )

    # --------------------------------------------------------
    # Stage 1: blind description
    # --------------------------------------------------------

    visual_description = (
        run_description_pass(
            vlm_model,
            vlm_processor,
            vlm_image,
        )
    )

    # --------------------------------------------------------
    # Stage 2: semantic feature extraction
    # --------------------------------------------------------

    raw_output = (
        run_attribute_pass(
            vlm_model,
            vlm_processor,
            vlm_image,
            visual_description,
            attribute_prompt,
        )
    )

    parsed = extract_json_object(
        raw_output
    )

    features = validate_prediction(
        parsed,
        master_attributes,
    )

    binary_vector = build_binary_vector(
        features
    )

    decision_result = predict_failure(
        binary_vector
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "status": "success",

        "image": {
            "filename": image.filename,

            "original_size": {
                "width": original_size[0],
                "height": original_size[1],
            },

            "vlm_size": {
                "width": vlm_image.size[0],
                "height": vlm_image.size[1],
            },
        },

        "detection": {
            "detector_name":
                detector_name,

            "class_name":
                class_name,

            "confidence":
                confidence,
        },

        "feature_order":
            TARGET_ATTRIBUTE_IDS,

        "visual_description":
            visual_description,

        "semantic_features":
            features,

        "binary_vector":
            binary_vector,

        "decision_tree": {
            "prediction":
                decision_result[
                    "pred_label"
                ],

            "pred_binary":
                decision_result[
                    "pred_binary"
                ],

            "failure_probability":
                round(
                    decision_result[
                        "failure_probability"
                    ],
                    4,
                ),
        },
    }