import joblib
import numpy as np
from pathlib import Path
from PIL import Image

from stage4A_qwen_master_attribute_annotation_100_TP import (
    MASTER_ATTRIBUTES_FILE,
    TARGET_ATTRIBUTE_IDS,
    load_master_attributes,
    select_target_attributes,
    build_attribute_prompt,
    resize_image_div3,
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

DECISION_TREE_MODEL = Path(
    "/home/tailam/SSL/RAIL_PROJECT/Rule_discovery/Experiment_3/stage4C_semantic_classifier_results_400_two_stage_TRAIN_IMPORTANCE_RETRAIN/Feature_Set_B_slope_semantic_only/Decision_Tree_Importance_GT_0/decision_tree_reduced_FINAL_ALL_DATA.joblib"
)

IMAGE_PATH = Path(
    "/home/tailam/SSL/RAIL_PROJECT/Savant/samples/railway_monitoring/prompt_testing/Data/Slope_soil_failure/images_TP/C10d_S02315_preserve.jpg"
)


# ============================================================
# DECISION TREE
# ============================================================

def load_decision_tree():

    if not DECISION_TREE_MODEL.exists():
        raise FileNotFoundError(
            f"Decision Tree model not found: "
            f"{DECISION_TREE_MODEL}"
        )

    return joblib.load(
        DECISION_TREE_MODEL
    )


def predict_failure(
    decision_tree,
    binary_vector,
):

    X = np.array(
        [binary_vector],
        dtype=np.int32,
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image_path,
    qwen_model,
    qwen_processor,
    master_attributes,
    attribute_prompt,
):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = (
        Image.open(image_path)
        .convert("RGB")
    )

    image = resize_image_div3(
        image
    )

    # --------------------------------------------------------
    # Stage 1: blind visual description
    # --------------------------------------------------------

    visual_description = (
        run_description_pass(
            qwen_model,
            qwen_processor,
            image,
        )
    )

    # --------------------------------------------------------
    # Stage 2: evaluate the 6 selected attributes
    # --------------------------------------------------------

    raw_output = (
        run_attribute_pass(
            qwen_model,
            qwen_processor,
            image,
            visual_description,
            attribute_prompt,
        )
    )

    parsed = extract_json_object(
        raw_output
    )

    predictions = validate_prediction(
        parsed,
        master_attributes,
    )

    binary_vector = build_binary_vector(
        predictions
    )

    return (
        predictions,
        binary_vector,
        visual_description,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load semantic attribute definitions
    # --------------------------------------------------------

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

    print(
        "Feature order:",
        TARGET_ATTRIBUTE_IDS,
    )

    # --------------------------------------------------------
    # Load Qwen
    # --------------------------------------------------------

    print("\nLoading Qwen3-VL...")

    qwen_model, qwen_processor = (
        load_qwen_vlm()
    )

    # --------------------------------------------------------
    # Load Decision Tree
    # --------------------------------------------------------

    print("Loading Decision Tree...")

    decision_tree = (
        load_decision_tree()
    )

    # --------------------------------------------------------
    # Extract semantic features
    # --------------------------------------------------------

    (
        predictions,
        binary_vector,
        visual_description,
    ) = extract_features(
        image_path=IMAGE_PATH,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
        master_attributes=master_attributes,
        attribute_prompt=attribute_prompt,
    )

    # --------------------------------------------------------
    # Decision Tree prediction
    # --------------------------------------------------------

    result = predict_failure(
        decision_tree,
        binary_vector,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 100)

    print(
        f"Image: {IMAGE_PATH.name}"
    )

    print()
    print("[BLIND DESCRIPTION]")
    print(
        visual_description
    )

    print()
    print("[SEMANTIC FEATURES]")

    for feature in TARGET_ATTRIBUTE_IDS:
        print(
            f"{feature}: "
            f"{int(predictions[feature])}"
        )

    print()
    print(
        "Binary vector:",
        binary_vector,
    )

    print()
    print("[DECISION TREE]")

    print(
        "Prediction:",
        result["pred_label"],
    )

    print(
        "Failure probability:",
        f'{result["failure_probability"]:.4f}',
    )

    print("=" * 100)


if __name__ == "__main__":
    main()