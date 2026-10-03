import json
import time
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

MASTER_ATTRIBUTES_FILE = Path(
    "/home/tailam/SSL/RAIL_PROJECT/Rule_discovery/Experiment_3/"
    "stage2C_global_consolidation_input/"
    "master_attributes_Qwen_Phrase_Only.txt"
)

DESCRIPTION_MAX_NEW_TOKENS = 120
ATTRIBUTE_MAX_NEW_TOKENS = 768

# TARGET_ATTRIBUTE_IDS = [
#     "M05",
#     "M07",
#     "M01",
#     "M13",
#     "M11",
#     "M04",
# ]

TARGET_ATTRIBUTE_IDS = [
    "M07",
    "M05",
    "M13",
    "M01",
    "M11",
    "M04",
    "M12",
]


# ============================================================
# TWO-STAGE PROMPTS
# ============================================================

DESCRIPTION_SYSTEM_MESSAGE = """
You are a visual inspection assistant describing railway tunnel portal scenes.

Your task is only to describe clearly visible physical characteristics
of the slope regions in the image.

Do not classify the image.
Do not infer the ground-truth category.
Do not infer causes, previous conditions, temporal changes, or future risk.
Use only directly visible evidence.
""".strip()


DESCRIPTION_USER_PROMPT = """
Look at the slope and the ground beside the railway near the tunnel portal.

Describe only what is clearly visible in the image. Focus on the slope condition and nearby ground.

If some parts look different from the surrounding slope or ground, describe those differences. If not, describe the overall visible slope and ground.

Do not decide whether the scene is normal or damaged. Do not explain causes, past changes, movement, or future risk. If something is unclear, leave it out.

Ignore weather, seasons, snow, lighting, people, vehicles, camera quality, and unrelated railway equipment.

Return one short paragraph of 2 to 3 sentences.
""".strip()


ATTRIBUTE_SYSTEM_MESSAGE = """
You inspect railway tunnel slope images and evaluate predefined visual attributes.

A short blind visual description of the same image has already been produced in a previous step.
Use that description as an intermediate visual summary to keep the interpretation consistent, but also inspect the image directly because the short description may omit visible details.

For each visual attribute:
- return true if it is clearly visible;
- return false if it is absent, unclear, or uncertain.

Use only directly visible evidence from the image and the blind description.
Do not infer slope failure, causes, risk, ground-truth category, or past events.

For the final attribute-evaluation response, return JSON only.
Use the exact keys provided in the final user prompt.
Do not rename, omit, or add any keys.
""".strip()



# ============================================================
# MASTER ATTRIBUTE LOADING
# ============================================================

def load_json_like(path):
    text = Path(
        path
    ).read_text(
        encoding="utf-8"
    ).strip()

    if text.startswith("```"):
        lines = (
            text.splitlines()[1:]
        )

        if (
            lines
            and lines[-1]
            .strip()
            .startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    return json.loads(
        text
    )


def load_master_attributes(path):
    data = load_json_like(
        path
    )

    attrs = data.get(
        "master_attributes",
        [],
    )

    if not attrs:
        raise ValueError(
            f'No "master_attributes" found in {path}'
        )

    return attrs


def select_target_attributes(
    master_attributes,
):
    attribute_by_id = {
        attr["master_attribute_id"]: attr
        for attr in master_attributes
    }

    missing = [
        attr_id
        for attr_id in TARGET_ATTRIBUTE_IDS
        if attr_id not in attribute_by_id
    ]

    if missing:
        raise ValueError(
            f"Target attributes not found: {missing}"
        )

    return [
        attribute_by_id[attr_id]
        for attr_id in TARGET_ATTRIBUTE_IDS
    ]



def make_output_key(attr):
    return (
        f'{attr["master_attribute_id"]}_'
        f'{attr["master_name"]}'
    )


def build_attribute_prompt(
    master_attributes,
):
    lines = [
        "Inspect the image and evaluate each visual attribute below.",
        "",
        "For each attribute:",
        "- return true if the described visual attribute is clearly visible;",
        "- return false if it is absent, unclear, or uncertain.",
        "",
        "Use only directly visible evidence from the image.",
        "",
    ]

    for attr in master_attributes:
        key = make_output_key(
            attr
        )

        definition = str(
            attr[
                "master_definition"
            ]
        ).strip()

        phrases = (
            attr.get(
                "representative_phrases",
                [],
            )[:3]
        )

        lines.append(
            f"For {key}, evaluate the following visual attribute:"
        )

        lines.append("")
        lines.append(
            definition
        )

        if phrases:
            lines.append("")
            lines.append(
                "Representative examples include:"
            )

            for phrase in phrases:
                lines.append(
                    f'- "{phrase}"'
                )

        lines.append("")

        lines.append(
            "Return true if this visual attribute is clearly visible "
            "in the image; otherwise return false."
        )

        lines.append("")
        lines.append(
            "-" * 60
        )
        lines.append("")

    expected = {
        make_output_key(
            attr
        ): False
        for attr
        in master_attributes
    }

    lines.append(
        "Return one JSON object only using exactly these keys:"
    )

    lines.append(
        json.dumps(
            expected,
            ensure_ascii=False,
            indent=2,
        )
    )

    return "\n".join(
        lines
    )



def run_description_pass(
    model,
    processor,
    image,
):
    """Stage-1-style blind visual description for the current image."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": DESCRIPTION_SYSTEM_MESSAGE,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": DESCRIPTION_USER_PROMPT,
                },
            ],
        },
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(model.device)
            if hasattr(value, "to")
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=DESCRIPTION_MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, input_length:]

    return processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def run_attribute_pass(
    model,
    processor,
    image,
    visual_description,
    attribute_prompt,
):
    """Evaluate master attributes after the Stage-1-style description pass."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": ATTRIBUTE_SYSTEM_MESSAGE,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": DESCRIPTION_USER_PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": visual_description,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Now evaluate the predefined master visual attributes for "
                        "the same image. Use the blind description above to keep "
                        "your visual interpretation consistent, but verify every "
                        "decision against the image itself.\n\n"
                        + attribute_prompt
                    ),
                }
            ],
        },
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(model.device)
            if hasattr(value, "to")
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=ATTRIBUTE_MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, input_length:]

    return processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def extract_json_object(
    text,
):
    cleaned = (
        text.strip()
    )

    if cleaned.startswith(
        "```"
    ):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        pass

    start = cleaned.find(
        "{"
    )

    end = cleaned.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "No JSON object found in Qwen output."
        )

    return json.loads(
        cleaned[
            start:end + 1
        ]
    )


def normalize_bool(
    value,
):
    if isinstance(
        value,
        bool,
    ):
        return value

    if (
        isinstance(
            value,
            int,
        )
        and value
        in (
            0,
            1,
        )
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        str,
    ):
        value = (
            value.strip()
            .lower()
        )

        if value == "true":
            return True

        if value == "false":
            return False

    raise ValueError(
        f"Expected true/false, got: {value!r}"
    )


def validate_prediction(
    parsed,
    master_attributes,
):
    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "Qwen output is not a JSON object."
        )

    expected_mapping = {
        make_output_key(
            attr
        ):
        attr[
            "master_attribute_id"
        ]
        for attr
        in master_attributes
    }

    expected_keys = set(
        expected_mapping.keys()
    )

    received_keys = set(
        parsed.keys()
    )

    missing = sorted(
        expected_keys
        - received_keys
    )

    extra = sorted(
        received_keys
        - expected_keys
    )

    if missing:
        raise ValueError(
            f"Missing attributes: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected attributes: {extra}"
        )

    normalized = {}

    for (
        long_key,
        compact_id,
    ) in expected_mapping.items():

        normalized[
            compact_id
        ] = normalize_bool(
            parsed[
                long_key
            ]
        )

    return normalized


def build_binary_vector(predictions):
    return [
        int(predictions[attr_id])
        for attr_id in TARGET_ATTRIBUTE_IDS
    ]



def load_qwen_vlm(
    model_id: str = MODEL_ID,
):

    print(
        f"[VLM] Loading model: {model_id}"
    )

    vlm_model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=False,
        )
        .eval()
    )

    vlm_processor = (
        AutoProcessor.from_pretrained(
            model_id
        )
    )

    print("[VLM] Model loaded.")

    return vlm_model, vlm_processor
