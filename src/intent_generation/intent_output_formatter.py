"""Parse free-form model responses and package teacher-forced token tensors.

Annotation CSVs use underscore keys; the Qwen JSON schema uses spaced keys
("communicative intent", "reasoning"). Only those two fields are supervised.
Evidence and other annotation columns are ignored here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


# Must be a subset of model.pC.JSON_FIELDS and match prepare_inputs output_fields.
EXPECTED_FIELDS = ("communicative intent", "reasoning")
ANNOTATION_KEY_MAP = {
    "communicative_intent": "communicative intent",
    "communicative intent": "communicative intent",
    "reasoning": "reasoning",
}


@dataclass
class FormattedTokenOutput:
    """Teacher-forced logits aligned with annotation labels and per-field masks."""

    logits: Tensor
    labels: Tensor
    field_masks: Mapping[str, Tensor]


@dataclass
class FormattedIntentOutput:
    """One free-generation parse used for metrics and the missing-field penalty."""

    fields: dict[str, str]
    missing_fields: tuple[str, ...]
    valid_json: bool
    raw_output: str


def annotation_targets(batch_targets: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Convert collated underscore annotation columns into per-sample output JSON dicts."""
    if not batch_targets:
        raise ValueError("batch targets are empty")
    mapped = {}
    for key, values in batch_targets.items():
        field = ANNOTATION_KEY_MAP.get(key)
        if field is None:
            continue  # ignore evidence / movement / gesture / unknown columns
        if field not in EXPECTED_FIELDS:
            continue
        mapped[field] = values
    missing = [field for field in EXPECTED_FIELDS if field not in mapped]
    if missing:
        raise ValueError(f"Annotation batch is missing supervised fields: {missing}")
    batch_size = len(next(iter(mapped.values())))
    if any(len(values) != batch_size for values in mapped.values()):
        raise ValueError("All annotation target lists must share the batch size")
    return [{field: mapped[field][i] for field in EXPECTED_FIELDS} for i in range(batch_size)]


def format_token_output(logits: Tensor, labels: Tensor | None,
                        field_masks: Mapping[str, Tensor] | None) -> FormattedTokenOutput:
    """Keep logits attached; refuse detached or incomplete supervision tensors."""
    if labels is None or field_masks is None:
        raise ValueError("Teacher-forced labels and field_masks are required for training loss")
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [B,L,V] and labels [B,L]")
    if logits.device != labels.device:
        raise ValueError("logits and labels must be on the same device")
    masks = {}
    for field in EXPECTED_FIELDS:
        if field not in field_masks:
            raise ValueError(f"field_masks missing supervised field {field!r}")
        mask = field_masks[field]
        if mask.shape != labels.shape or mask.dtype != torch.bool:
            raise ValueError(f"field mask for {field!r} must be a BoolTensor shaped like labels")
        if mask.device != labels.device:
            raise ValueError("field masks must match the labels device")
        masks[field] = mask
    return FormattedTokenOutput(logits=logits, labels=labels, field_masks=masks)


def intent_output_formatter(text: str) -> FormattedIntentOutput:
    """Parse one generated response into expected JSON fields without inventing text."""
    raw = text if isinstance(text, str) else str(text)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    fields = {field: "" for field in EXPECTED_FIELDS}
    valid_json = False
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            valid_json = True
            for field in EXPECTED_FIELDS:
                value = obj.get(field, obj.get(field.replace(" ", "_")))
                if isinstance(value, str):
                    fields[field] = value.strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        valid_json = False
    missing = tuple(field for field in EXPECTED_FIELDS if not fields[field])
    return FormattedIntentOutput(fields=fields, missing_fields=missing,
                                 valid_json=valid_json, raw_output=raw)
