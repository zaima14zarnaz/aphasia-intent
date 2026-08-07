"""
Train LandmarkGestureClassifier on pre-extracted Aphasia gesture segments.

Loads stored MediaPipe pose landmarks for positive (gest_*) and negative
(neg_*) segments and trains a spatiotemporal landmark classifier for binary
gesture detection (POS=1, NEG=0).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from losses import (
    alignment_warmup_factor,
    compute_alignment_target_probability,
    compute_classification_loss,
    compute_masked_alignment_loss,
    get_batch_alignment_targets,
)
from metrics import GestureClassificationMetrics, format_metrics, print_per_class_accuracy
from model.custom_mamba import LandmarkGestureClassifier
# from model.llm_based import LLMBasedGestureClassifier

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment, misc]

_SRC_DIR = Path(__file__).resolve().parent
_PREPROCESS_DIR = _SRC_DIR / "pre_process"
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))

from collate import gesture_segment_collate_fn
from dataloder import (
    LABEL_NAME_BY_VALUE,
    NEGATIVE_SEGMENT_LABEL,
    NEGATIVE_SEGMENT_LABEL_NAME,
    NUM_MEDIAPIPE_FACE_UPPER_BODY_LANDMARKS,
    POSITIVE_SEGMENT_LABEL,
    POSITIVE_SEGMENT_LABEL_NAME,
)

_DATASET_MODULE_CANDIDATES = (
    "dataloder",
    "dataset",
    "aphasia_gesture_dataset",
    "dataloader",
)

LANDMARKS_DIR = (
    "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments_landmarks/mediapipe"
)
NEGATIVE_LANDMARKS_DIR = (
    "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments_landmarks/mediapipe"
)
TRANSCRIPT_DIR = (
    "/data/quantization/zaima/aphasiabank/neural2/control/annotations/aphasia"
)
SAMPLE_OUTPUT_DIR = _SRC_DIR / "training_outputs" / "sample_train_segment"
BEST_MODEL_PATH = _SRC_DIR / "best_landmark_gesture_classifier.pt"
FREQUENT_MISCLASSIFICATION_PATH = (
    _SRC_DIR / "training_outputs" / "frequently_misclassified_val_samples.txt"
)
RESULTS_PER_EPOCH_PATH = _SRC_DIR / "training_outputs" / "results_per_epoch.csv"
RESULTS_PER_EPOCH_FIELDNAMES = (
    "epoch",
    "train_acc",
    "train_precision",
    "train_recall",
    "train_f1",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_pos_acc",
    "val_neg_acc"
)
FREQUENT_MISCLASSIFICATION_THRESHOLD = 0.9

FIXED_TRAIN_VIDEO_KEYS = frozenset(
    {
        "65_1",
        "81_1",
        "92_1",
        "101_1",
        "104_1",
        "105_1",
        "121_1",
        "123_1",
        "128_1",
    }
)
FIXED_VAL_VIDEO_KEYS = frozenset(
    {
        "133_1",
        "171_1",
        "202_1",
        "208_1",
    }
)

NUM_EPOCHS = 120
BATCH_SIZE = 16
NUM_FRAMES = 128
NUM_JOINTS = NUM_MEDIAPIPE_FACE_UPPER_BODY_LANDMARKS
IN_CHANNELS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

print(f"NUM_EPOCHS: {NUM_EPOCHS}")

_SAMPLE_LABEL_KEYS = (
    "label",
    
    "gesture",
    "gesture_label",
    "class_name",
    "annotation",
    "target",
)
_ANNOTATION_LABEL_KEYS = (
    "label_name",
    "gesture_type",
    "label",
    "gesture",
    "gesture_label",
    "class_name",
    "target",
)
_BATCH_LABEL_KEYS = (
    "label",
    "labels",
    "gesture",
    "gesture_label",
    "class_name",
    "target",
    "targets",
    "gesture_types",
    "gesture_type",
)


def _import_aphasia_gesture_dataset():
    """Import gesture segment dataset from the dataset module in pre_process/."""
    for module_name in _DATASET_MODULE_CANDIDATES:
        module_path = _PREPROCESS_DIR / f"{module_name}.py"
        if not module_path.is_file():
            continue

        module = importlib.import_module(module_name)
        dataset_cls = getattr(module, "AphasiaGestureSegmentDataset", None)
        if dataset_cls is not None:
            return dataset_cls

        raise ImportError(
            f"Found {_PREPROCESS_DIR / f'{module_name}.py'} but it does not define "
            "AphasiaGestureSegmentDataset."
        )

    searched = ", ".join(f"{name}.py" for name in _DATASET_MODULE_CANDIDATES)
    raise ImportError(
        f"Could not find a dataset module in {_PREPROCESS_DIR}. "
        f"Expected one of: {searched}"
    )


AphasiaGestureSegmentDataset = _import_aphasia_gesture_dataset()


def save_segment_frames(
    frames: torch.Tensor,
    output_dir: str | Path,
    annotation: dict | None = None,
) -> None:
    """
    Save a single gesture segment's frames as PNG files.

    Args:
        frames: torch.Tensor [T, C, H, W] in RGB, values typically in [0, 1].
        output_dir: directory where frame_0000.png, ... will be written.
        annotation: optional metadata written to annotation.txt.
    """
    if frames.ndim != 4:
        raise ValueError(
            f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}"
        )

    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not write to {output_path}. Use a writable path or "
            "run from a directory where you have permissions."
        ) from exc

    video = frames.detach().cpu().clamp(0.0, 1.0)
    video_uint8 = (video * 255.0).round().to(torch.uint8)
    video_hwc = video_uint8.permute(0, 2, 3, 1).numpy()

    for frame_idx in range(video_hwc.shape[0]):
        frame_bgr = cv2.cvtColor(video_hwc[frame_idx], cv2.COLOR_RGB2BGR)
        frame_path = output_path / f"frame_{frame_idx:04d}.png"
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise RuntimeError(f"Failed to write frame image: {frame_path}")

    if annotation is not None:
        annotation_path = output_path / "annotation.txt"
        lines = [
            f"video_filename: {annotation.get('video_filename')}",
            f"annotation_filename: {annotation.get('annotation_filename')}",
            f"start_ms: {annotation.get('start_ms')}",
            f"end_ms: {annotation.get('end_ms')}",
            f"duration_ms: {annotation.get('duration_ms')}",
            f"task: {annotation.get('task')}",
            f"gesture_type: {annotation.get('gesture_type')}",
            f"function: {annotation.get('function')}",
            f"left_hand_label: {annotation.get('left_hand_label')}",
            f"left_hand_shape: {annotation.get('left_hand_shape')}",
            f"right_hand_label: {annotation.get('right_hand_label')}",
            f"right_hand_shape: {annotation.get('right_hand_shape')}",
        ]
        try:
            annotation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except PermissionError as exc:
            raise PermissionError(
                f"Could not write to {output_path}. Use a writable path or "
                "run from a directory where you have permissions."
            ) from exc


def get_label_from_sample(sample: dict[str, Any]) -> int | str:
    """Extract the class label from a single dataset item."""
    if not isinstance(sample, dict):
        raise TypeError(
            f"Expected sample to be a dict, got {type(sample)!r}"
        )

    if "label" in sample and isinstance(sample["label"], (int, np.integer)):
        return int(sample["label"])

    for key in _SAMPLE_LABEL_KEYS:
        if key not in sample:
            continue

        value = sample[key]
        if key == "annotation":
            if isinstance(value, dict):
                annotation_label = value.get("label")
                if isinstance(annotation_label, (int, np.integer)):
                    return int(annotation_label)

                for annotation_key in _ANNOTATION_LABEL_KEYS:
                    annotation_value = value.get(annotation_key)
                    if annotation_value is None:
                        continue
                    if annotation_key == "label" and isinstance(
                        annotation_value, (int, np.integer)
                    ):
                        return int(annotation_value)
                    return str(annotation_value)
            continue

        if isinstance(value, (int, np.integer)):
            return int(value)

        if value is not None:
            return str(value)

    raise KeyError(
        "Could not find a gesture label in sample. "
        f"Available sample keys: {sorted(sample.keys())}"
    )


def get_label_from_dataset_item(dataset: Dataset, index: int) -> int | str:
    """
    Read a class label without loading segment frames when possible.
    """
    if hasattr(dataset, "get_annotation"):
        metadata = dataset.get_annotation(index)
        if isinstance(metadata, dict):
            return get_label_from_sample(metadata)

        label = getattr(metadata, "label", None)
        if isinstance(label, (int, np.integer)):
            return int(label)

        gesture_type = getattr(metadata, "gesture_type", None)
        if gesture_type is not None:
            return str(gesture_type)

        annotation = getattr(metadata, "annotation", None)
        if isinstance(annotation, dict):
            return get_label_from_sample({"annotation": annotation})

    return get_label_from_sample(dataset[index])


def build_label_mapping(
    dataset: Dataset,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build fixed binary label mappings for POS/NEG gesture detection."""
    label_to_idx = {
        NEGATIVE_SEGMENT_LABEL_NAME: NEGATIVE_SEGMENT_LABEL,
        POSITIVE_SEGMENT_LABEL_NAME: POSITIVE_SEGMENT_LABEL,
    }
    idx_to_label = dict(LABEL_NAME_BY_VALUE)

    seen_labels: set[int] = set()
    for index in range(len(dataset)):
        label = get_label_from_dataset_item(dataset, index)
        if isinstance(label, int):
            seen_labels.add(label)
        elif isinstance(label, str):
            if label not in label_to_idx:
                raise RuntimeError(
                    f"Unknown label {label!r} at dataset index {index}. "
                    f"Expected {NEGATIVE_SEGMENT_LABEL_NAME!r} or "
                    f"{POSITIVE_SEGMENT_LABEL_NAME!r}."
                )
            seen_labels.add(label_to_idx[label])
        else:
            raise RuntimeError(
                f"Unsupported label type at dataset index {index}: {type(label)!r}"
            )

    if not seen_labels:
        raise RuntimeError("No gesture labels were found in the dataset.")

    return label_to_idx, idx_to_label


def count_dataset_labels(
    dataset: Dataset,
    label_to_idx: dict[str, int] | None = None,
) -> dict[int, int]:
    """Count samples per integer class label."""
    if label_to_idx is None:
        label_to_idx = {
            NEGATIVE_SEGMENT_LABEL_NAME: NEGATIVE_SEGMENT_LABEL,
            POSITIVE_SEGMENT_LABEL_NAME: POSITIVE_SEGMENT_LABEL,
        }

    counts: dict[int, int] = {}
    for index in range(len(dataset)):
        label = get_label_from_dataset_item(dataset, index)
        if isinstance(label, str):
            label = label_to_idx[label]
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def _label_value_to_index(
    value: Any,
    label_to_idx: dict[str, int],
) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Expected a scalar label tensor, got shape {tuple(value.shape)}"
            )
        value = value.item()

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)

    if isinstance(value, str):
        if value not in label_to_idx:
            raise KeyError(
                f"Unknown gesture label {value!r}. "
                f"Known labels: {sorted(label_to_idx)}"
            )
        return label_to_idx[value]

    raise TypeError(f"Unsupported label value type: {type(value)!r}")


def get_batch_labels(
    batch: dict[str, Any],
    label_to_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Extract integer class labels from a collated batch."""
    for key in _BATCH_LABEL_KEYS:
        if key not in batch:
            continue

        raw_labels = batch[key]

        if isinstance(raw_labels, torch.Tensor):
            if raw_labels.dtype in (torch.int64, torch.int32, torch.long):
                return raw_labels.long().to(device)
            raw_labels = raw_labels.tolist()

        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]

        encoded = []
        for value in raw_labels:
            if value is None:
                raise ValueError(
                    "Encountered None label in batch. "
                    "Filter unlabeled samples before training."
                )
            encoded.append(_label_value_to_index(value, label_to_idx))

        return torch.tensor(encoded, dtype=torch.long, device=device)

    print(f"Available batch keys: {sorted(batch.keys())}")
    raise KeyError(
        "Could not find labels in batch. "
        f"Checked label keys: {_BATCH_LABEL_KEYS}"
    )


def sample_id_from_annotation(annotation: dict[str, Any]) -> str:
    """Build ``{video_filename}/{gest|neg}_{idx}`` from batch annotation metadata."""
    video_filename = annotation.get("video_filename")
    if not video_filename:
        raise KeyError(
            "Annotation is missing 'video_filename', which is required to build "
            "a validation sample id."
        )

    segment_dir = annotation.get("segment_dir")
    if segment_dir:
        segment_name = Path(str(segment_dir)).name
        if segment_name:
            return f"{video_filename}/{segment_name}"

    segment_index = annotation.get("segment_index")
    if segment_index is None:
        raise KeyError(
            "Annotation must include 'segment_dir' or 'segment_index' to build "
            "a validation sample id."
        )

    label_name = annotation.get("label_name")
    if label_name == POSITIVE_SEGMENT_LABEL_NAME:
        segment_prefix = "gest"
    elif label_name == NEGATIVE_SEGMENT_LABEL_NAME:
        segment_prefix = "neg"
    else:
        gesture_video_index = annotation.get("gesture_video_index")
        negative_video_index = annotation.get("negative_video_index")
        if gesture_video_index is not None:
            segment_prefix = "gest"
            segment_index = gesture_video_index
        elif negative_video_index is not None:
            segment_prefix = "neg"
            segment_index = negative_video_index
        else:
            raise KeyError(
                "Could not infer gest/neg segment prefix from annotation "
                f"for video {video_filename!r}."
            )

    return f"{video_filename}/{segment_prefix}_{segment_index}"


def record_val_misclassifications(
    logits: torch.Tensor,
    labels: torch.Tensor,
    annotations: list[dict[str, Any]],
    misclassification_counts: dict[str, int],
) -> None:
    """Increment per-sample misclassification counts for incorrect val predictions."""
    predictions = logits.argmax(dim=1).detach().cpu()
    labels_cpu = labels.detach().cpu()

    if predictions.shape[0] != labels_cpu.shape[0]:
        raise ValueError(
            "Batch size mismatch between predictions and labels: "
            f"predictions={predictions.shape[0]}, labels={labels_cpu.shape[0]}"
        )
    if len(annotations) != predictions.shape[0]:
        raise ValueError(
            "Batch size mismatch between predictions and annotations: "
            f"predictions={predictions.shape[0]}, annotations={len(annotations)}"
        )

    for prediction, label, annotation in zip(predictions, labels_cpu, annotations):
        if int(prediction.item()) == int(label.item()):
            continue

        sample_id = sample_id_from_annotation(annotation)
        misclassification_counts[sample_id] = misclassification_counts.get(sample_id, 0) + 1


def print_top_misclassified_val_samples(
    misclassification_counts: dict[str, int],
    *,
    num_epochs: int,
    top_k: int = 20,
) -> None:
    """Print the validation samples misclassified most often across training."""
    if not misclassification_counts:
        print("No validation misclassifications were recorded.")
        return

    ranked_samples = sorted(
        misclassification_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]

    print(f"Top {len(ranked_samples)} most frequently misclassified validation samples:")
    for rank, (sample_id, frequency) in enumerate(ranked_samples, start=1):
        epoch_rate = frequency / num_epochs if num_epochs > 0 else 0.0
        print(
            f"  {rank:2d}. {sample_id}  "
            f"misclassified {frequency}/{num_epochs} epochs ({epoch_rate:.1%})"
        )


def save_frequently_misclassified_val_samples(
    misclassification_counts: dict[str, int],
    *,
    output_path: Path,
    num_epochs: int,
    threshold: float = FREQUENT_MISCLASSIFICATION_THRESHOLD,
) -> list[tuple[str, int, float]]:
    """
    Save validation sample names misclassified in more than ``threshold`` of epochs.

    Each line contains:
        sample_id<TAB>misclassified_epochs/total_epochs<TAB>rate
    """
    if num_epochs <= 0:
        return []

    frequent_samples = [
        (sample_id, count, count / num_epochs)
        for sample_id, count in misclassification_counts.items()
        if (count / num_epochs) > threshold
    ]
    frequent_samples.sort(key=lambda item: (-item[1], item[0]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        f"# Validation samples misclassified in > {threshold:.0%} of epochs",
        f"# num_epochs: {num_epochs}",
        f"# saved_count: {len(frequent_samples)}",
        "",
    ]
    body_lines = [
        f"{sample_id}\t{count}/{num_epochs}\t{rate:.4f}"
        for sample_id, count, rate in frequent_samples
    ]
    output_path.write_text(
        "\n".join(header_lines + body_lines) + ("\n" if body_lines else ""),
        encoding="utf-8",
    )
    return frequent_samples


def print_frequently_misclassified_val_samples(
    frequent_samples: list[tuple[str, int, float]],
    *,
    output_path: Path,
    num_epochs: int,
    threshold: float = FREQUENT_MISCLASSIFICATION_THRESHOLD,
) -> None:
    """Print a summary of persistently misclassified validation samples."""
    if not frequent_samples:
        print(
            "No validation samples were misclassified in "
            f"> {threshold:.0%} of epochs."
        )
        print(f"Saved empty report to: {output_path}")
        return

    print(
        f"Validation samples misclassified in > {threshold:.0%} of epochs "
        f"({len(frequent_samples)} total):"
    )
    for sample_id, count, rate in frequent_samples:
        print(
            f"  {sample_id}  "
            f"misclassified {count}/{num_epochs} epochs ({rate:.1%})"
        )
    print(f"Saved sample names to: {output_path}")


def init_results_per_epoch_csv(output_path: Path) -> None:
    """Create or overwrite the per-epoch metrics CSV with a header row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULTS_PER_EPOCH_FIELDNAMES)
        writer.writeheader()


def append_results_per_epoch_row(
    output_path: Path,
    *,
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    """Append one epoch of train/val classification metrics to the CSV."""
    row = {
        "epoch": epoch,
        "train_acc": float(train_metrics["accuracy"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_f1": float(train_metrics["f1"]),
        "val_acc": float(val_metrics["accuracy"]),
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_f1": float(val_metrics["f1"]),
        "val_pos_acc": float(val_metrics["per_class_accuracy"].get("POS")),
        "val_neg_acc": float(val_metrics["per_class_accuracy"].get("NEG")),
    }
    with output_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULTS_PER_EPOCH_FIELDNAMES)
        writer.writerow(row)


def landmarks_to_tensor(
    landmarks: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(landmarks, dtype=torch.float32, device=device)


def get_batch_transcripts(batch: dict[str, Any]) -> list[str]:
    """Extract per-sample transcript strings from a collated batch."""
    if "transcripts" in batch:
        return [str(transcript) for transcript in batch["transcripts"]]
    raise KeyError("Batch is missing 'transcripts'.")


def get_batch_transcript_chunks(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[list[list[str]], torch.Tensor | None]:
    """Extract chunked transcripts and their validity mask from a collated batch."""
    if "transcript_chunks" not in batch:
        raise KeyError("Batch is missing 'transcript_chunks'.")

    transcript_chunks = [
        [str(chunk) for chunk in sample_chunks]
        for sample_chunks in batch["transcript_chunks"]
    ]
    chunk_mask = batch.get("transcript_chunk_mask")
    if chunk_mask is None:
        return transcript_chunks, None
    return transcript_chunks, chunk_mask.to(device=device, dtype=torch.bool)


def _iter_with_progress(
    loader: DataLoader,
    desc: str,
):
    if tqdm is None:
        return loader

    # Force the bar on even when stderr is not a TTY (e.g. captured IDE terminals).
    return tqdm(
        loader,
        desc=desc,
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        file=sys.stderr,
        disable=False,
        mininterval=0.5,
    )


def _update_progress_bar(
    progress_bar,
    total_loss: float,
    total_samples: int,
    metrics_tracker: GestureClassificationMetrics,
) -> None:
    if tqdm is None or progress_bar is None:
        return

    running_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    running_metrics = metrics_tracker.compute()
    progress_bar.set_postfix(
        loss=f"{running_loss:.4f}",
        acc=f"{running_metrics['accuracy']:.4f}",
        refresh=True,
    )


def train_one_epoch(
    model: LandmarkGestureClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_to_idx: dict[str, int],
    num_classes: int,
    idx_to_label: dict[int, str],
    *,
    epoch: int,
    expected_num_chunks: int,
    alignment_loss_weight: float,
    alignment_label_smoothing: float,
    alignment_warmup_epochs: int,
    use_alignment_supervision: bool,
) -> dict[str, float]:
    model.train()
    metrics_tracker = GestureClassificationMetrics(
        num_classes=num_classes,
        idx_to_label=idx_to_label,
    )
    metrics_tracker.reset()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_alignment_loss = 0.0
    total_alignment_target_probability = 0.0
    total_annotated_alignment_samples = 0
    total_samples = 0

    warmup_factor = alignment_warmup_factor(epoch, alignment_warmup_epochs)
    effective_alignment_weight = (
        alignment_loss_weight * warmup_factor
        if use_alignment_supervision
        else 0.0
    )

    progress_bar = _iter_with_progress(loader, desc="train")
    for batch in progress_bar:
        landmarks = landmarks_to_tensor(batch["landmarks"], device)
        labels = get_batch_labels(batch, label_to_idx, device)
        transcripts = get_batch_transcripts(batch)
        transcript_chunks, transcript_chunk_mask = get_batch_transcript_chunks(
            batch,
            device,
        )

        if use_alignment_supervision:
            alignment_targets = get_batch_alignment_targets(
                batch,
                device,
                expected_num_chunks,
            )
            (
                logits,
                attention_weights,
                _modality_gate,
                returned_chunk_mask,
            ) = model(
                landmarks,
                transcript_chunks=transcript_chunks,
                transcript_chunk_mask=transcript_chunk_mask,
                transcripts=transcripts,
                return_attention=True,
            )
            if not torch.isfinite(logits).all():
                print("logits:", logits)
                print("logits min/max:", logits.nan_to_num().min(), logits.nan_to_num().max())
                raise RuntimeError("Model produced NaN/Inf logits")

            if returned_chunk_mask is None:
                if transcript_chunk_mask is None:
                    raise ValueError(
                        "Model returned no transcript_chunk_mask and batch did not "
                        "provide one."
                    )
                returned_chunk_mask = transcript_chunk_mask

            classification_loss = compute_classification_loss(logits, labels)
            alignment_loss, num_annotated, mean_attention = (
                compute_masked_alignment_loss(
                    attention_weights,
                    alignment_targets,
                    returned_chunk_mask,
                    label_smoothing=alignment_label_smoothing,
                )
            )
            loss = classification_loss + effective_alignment_weight * alignment_loss
        else:
            logits = model(
                landmarks,
                transcript_chunks=transcript_chunks,
                transcript_chunk_mask=transcript_chunk_mask,
                transcripts=transcripts,
                return_attention=False,
            )
            if not torch.isfinite(logits).all():
                print("logits:", logits)
                print("logits min/max:", logits.nan_to_num().min(), logits.nan_to_num().max())
                raise RuntimeError("Model produced NaN/Inf logits")

            classification_loss = compute_classification_loss(logits, labels)
            alignment_loss = logits.new_tensor(0.0)
            num_annotated = 0
            mean_attention = None
            loss = classification_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_classification_loss += classification_loss.item() * batch_size
        total_alignment_loss += alignment_loss.item() * batch_size
        total_samples += batch_size
        if use_alignment_supervision and num_annotated > 0:
            batch_target_probability = compute_alignment_target_probability(
                mean_attention,
                alignment_targets,
            )
            total_alignment_target_probability += (
                batch_target_probability.item() * num_annotated
            )
            total_annotated_alignment_samples += num_annotated
        metrics_tracker.update(logits, labels)
        _update_progress_bar(progress_bar, total_loss, total_samples, metrics_tracker)

    average_total_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    average_classification_loss = (
        0.0 if total_samples == 0 else total_classification_loss / total_samples
    )
    average_alignment_loss = (
        0.0 if total_samples == 0 else total_alignment_loss / total_samples
    )
    average_alignment_target_probability = (
        0.0
        if total_annotated_alignment_samples == 0
        else total_alignment_target_probability / total_annotated_alignment_samples
    )

    metrics = metrics_tracker.compute()
    metrics.update(
        {
            "total_loss": average_total_loss,
            "classification_loss": average_classification_loss,
            "alignment_loss": average_alignment_loss,
            "annotated_alignment_samples": float(total_annotated_alignment_samples),
            "alignment_target_probability": average_alignment_target_probability,
            "effective_alignment_weight": effective_alignment_weight,
        }
    )
    return metrics


def validate_one_epoch(
    model: LandmarkGestureClassifier,
    loader: DataLoader,
    device: torch.device,
    label_to_idx: dict[str, int],
    num_classes: int,
    idx_to_label: dict[int, str],
    *,
    expected_num_chunks: int,
    alignment_loss_weight: float,
    alignment_label_smoothing: float,
    alignment_warmup_epochs: int,
    epoch: int,
    use_alignment_supervision: bool,
    misclassification_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    model.eval()
    metrics_tracker = GestureClassificationMetrics(
        num_classes=num_classes,
        idx_to_label=idx_to_label,
    )
    metrics_tracker.reset()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_alignment_loss = 0.0
    total_alignment_target_probability = 0.0
    total_annotated_alignment_samples = 0
    total_samples = 0

    warmup_factor = alignment_warmup_factor(epoch, alignment_warmup_epochs)
    effective_alignment_weight = (
        alignment_loss_weight * warmup_factor
        if use_alignment_supervision
        else 0.0
    )

    progress_bar = _iter_with_progress(loader, desc="val")
    with torch.no_grad():
        for batch in progress_bar:
            landmarks = landmarks_to_tensor(batch["landmarks"], device)
            labels = get_batch_labels(batch, label_to_idx, device)
            transcripts = get_batch_transcripts(batch)
            transcript_chunks, transcript_chunk_mask = get_batch_transcript_chunks(
                batch,
                device,
            )

            if use_alignment_supervision:
                alignment_targets = get_batch_alignment_targets(
                    batch,
                    device,
                    expected_num_chunks,
                )
                (
                    logits,
                    attention_weights,
                    _modality_gate,
                    returned_chunk_mask,
                ) = model(
                    landmarks,
                    transcript_chunks=transcript_chunks,
                    transcript_chunk_mask=transcript_chunk_mask,
                    transcripts=transcripts,
                    return_attention=True,
                )

                if returned_chunk_mask is None:
                    if transcript_chunk_mask is None:
                        raise ValueError(
                            "Model returned no transcript_chunk_mask and batch did not "
                            "provide one."
                        )
                    returned_chunk_mask = transcript_chunk_mask

                classification_loss = compute_classification_loss(logits, labels)
                alignment_loss, num_annotated, mean_attention = (
                    compute_masked_alignment_loss(
                        attention_weights,
                        alignment_targets,
                        returned_chunk_mask,
                        label_smoothing=alignment_label_smoothing,
                    )
                )
                loss = classification_loss + effective_alignment_weight * alignment_loss
            else:
                logits = model(
                    landmarks,
                    transcript_chunks=transcript_chunks,
                    transcript_chunk_mask=transcript_chunk_mask,
                    transcripts=transcripts,
                    return_attention=False,
                )
                classification_loss = compute_classification_loss(logits, labels)
                alignment_loss = logits.new_tensor(0.0)
                num_annotated = 0
                mean_attention = None
                loss = classification_loss

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_classification_loss += classification_loss.item() * batch_size
            total_alignment_loss += alignment_loss.item() * batch_size
            total_samples += batch_size
            if use_alignment_supervision and num_annotated > 0:
                batch_target_probability = compute_alignment_target_probability(
                    mean_attention,
                    alignment_targets,
                )
                total_alignment_target_probability += (
                    batch_target_probability.item() * num_annotated
                )
                total_annotated_alignment_samples += num_annotated
            metrics_tracker.update(logits, labels)
            if misclassification_counts is not None:
                if "annotations" not in batch:
                    raise KeyError(
                        "Validation batch is missing 'annotations', which are "
                        "required to track misclassified samples."
                    )
                record_val_misclassifications(
                    logits,
                    labels,
                    batch["annotations"],
                    misclassification_counts,
                )
            _update_progress_bar(
                progress_bar,
                total_loss,
                total_samples,
                metrics_tracker,
            )

    average_total_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    average_classification_loss = (
        0.0 if total_samples == 0 else total_classification_loss / total_samples
    )
    average_alignment_loss = (
        0.0 if total_samples == 0 else total_alignment_loss / total_samples
    )
    average_alignment_target_probability = (
        0.0
        if total_annotated_alignment_samples == 0
        else total_alignment_target_probability / total_annotated_alignment_samples
    )

    metrics = metrics_tracker.compute()
    metrics.update(
        {
            "total_loss": average_total_loss,
            "classification_loss": average_classification_loss,
            "alignment_loss": average_alignment_loss,
            "annotated_alignment_samples": float(total_annotated_alignment_samples),
            "alignment_target_probability": average_alignment_target_probability,
            "effective_alignment_weight": effective_alignment_weight,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LandmarkGestureClassifier on gesture landmark segments."
    )
    parser.add_argument(
        "--per_video_samples",
        type=int,
        default=3,
        help=(
            "Randomly sample this many gestures per video each training epoch. "
            "Use <=0 to train on all gestures."
        ),
    )
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--gcn_dim", type=int, default=256)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--graph_top_k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default=TRANSCRIPT_DIR,
        help="Directory containing CHAT .cha transcript files.",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name for transcript encoding.",
    )
    parser.add_argument(
        "--transcript_embed_dim",
        type=int,
        default=256,
        help="Projected BERT embedding dimension fused with frame tokens.",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze BERT weights and only train the projection head.",
    )
    parser.add_argument(
        "--no_transcripts",
        action="store_true",
        help="Disable transcript encoding and fusion in the model.",
    )
    parser.add_argument(
        "--transcript_offset_seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds of transcript context to include before/after each gesture "
            "when building temporal transcript chunks."
        ),
    )
    parser.add_argument(
        "--num_transcript_chunks",
        type=int,
        default=5,
        help="Number of equal-duration transcript chunks in the expanded window.",
    )
    parser.add_argument(
        "--max_transcript_chunks",
        type=int,
        default=16,
        help="Maximum supported transcript chunks for position embeddings.",
    )
    parser.add_argument(
        "--transcript_dropout_probability",
        type=float,
        default=0.2,
        help=(
            "Training-only probability of dropping all transcript context for a "
            "sample during modality dropout."
        ),
    )
    parser.add_argument(
        "--alignment_loss_weight",
        type=float,
        default=0.01,
        help="Weight for auxiliary pose-to-chunk alignment loss.",
    )
    parser.add_argument(
        "--alignment_label_smoothing",
        type=float,
        default=0.00,
        help="Label smoothing applied to alignment targets across valid chunks.",
    )
    parser.add_argument(
        "--alignment_warmup_epochs",
        type=int,
        default=5,
        help="Linear warmup epochs for the alignment loss weight.",
    )
    parser.add_argument(
        "--no_alignment_supervision",
        action="store_true",
        help=(
            "Disable alignment annotation targets and auxiliary alignment loss "
            "during training and validation."
        ),
    )
    args = parser.parse_args()

    if args.alignment_loss_weight < 0:
        parser.error("--alignment_loss_weight must be non-negative.")
    if not 0.0 <= args.alignment_label_smoothing < 1.0:
        parser.error("--alignment_label_smoothing must be in [0, 1).")
    if args.alignment_warmup_epochs < 0:
        parser.error("--alignment_warmup_epochs must be non-negative.")

    return args


def print_model_config(args: argparse.Namespace, *, num_classes: int) -> None:
    print("Model config:")
    print(f"  num_classes: {num_classes}")
    print(f"  num_frames: {NUM_FRAMES}")
    print(f"  num_joints: {NUM_JOINTS}")
    print(f"  in_channels: {IN_CHANNELS}")
    print(f"  model_dim: {args.model_dim}")
    print(f"  gcn_dim: {args.gcn_dim}")
    print(f"  gcn_layers: {args.gcn_layers}")
    print(f"  graph_top_k: {args.graph_top_k}")
    print(f"  dropout: {args.dropout}")
    print(f"  mamba_layers: {args.mamba_layers}")
    print(f"  mamba_d_state: {args.mamba_d_state}")
    print(f"  mamba_d_conv: {args.mamba_d_conv}")
    print(f"  mamba_expand: {args.mamba_expand}")
    print(f"  transcript_dir: {args.transcript_dir}")
    print(f"  use_transcripts: {not args.no_transcripts}")
    print(f"  bert_model_name: {args.bert_model_name}")
    print(f"  transcript_embed_dim: {args.transcript_embed_dim}")
    print(f"  freeze_bert: {args.freeze_bert}")
    print(f"  transcript_offset_seconds: {args.transcript_offset_seconds}")
    print(f"  num_transcript_chunks: {args.num_transcript_chunks}")
    print(f"  max_transcript_chunks: {args.max_transcript_chunks}")
    print(f"  transcript_dropout_probability: {args.transcript_dropout_probability}")
    print(f"  alignment_loss_weight: {args.alignment_loss_weight}")
    print(f"  alignment_label_smoothing: {args.alignment_label_smoothing}")
    print(f"  alignment_warmup_epochs: {args.alignment_warmup_epochs}")
    print(f"  alignment_supervision: {not args.no_alignment_supervision}")
    print()


def split_train_val_indices(
    dataset: AphasiaGestureSegmentDataset,
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split all gesture samples into global train/val subsets.

    Shuffles every sample index once, then assigns about ``train_fraction`` of
    the full dataset to train and the remainder to validation.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be between 0 and 1, got {train_fraction}"
        )

    all_indices = list(range(len(dataset.samples)))
    if len(all_indices) < 2:
        raise RuntimeError(
            "Need at least 2 gesture samples for an 80/20 train/val split, "
            f"got {len(all_indices)}"
        )

    rng = random.Random(seed)
    rng.shuffle(all_indices)

    num_samples = len(all_indices)
    train_count = int(num_samples * train_fraction)
    if train_count < 1:
        train_count = 1
    elif train_count >= num_samples:
        train_count = num_samples - 1

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count:]
    return train_indices, val_indices


def _video_key_from_filename(video_filename: str) -> str:
    """Map ``65-1.mp4`` to the compact key ``65_1``."""
    stem = Path(video_filename).stem
    subject_id, separator, day = stem.partition("-")
    if separator:
        return f"{subject_id}_{day}"
    return stem.replace("-", "_")


def _video_filename_from_key(video_key: str) -> str:
    """Map ``65_1`` to ``65-1.mp4``."""
    normalized = video_key.strip()
    if normalized.endswith(".mp4"):
        return normalized
    subject_id, separator, day = normalized.partition("_")
    if separator:
        return f"{subject_id}-{day}.mp4"
    return f"{normalized}.mp4"


def split_train_val_indices_by_video(
    dataset: AphasiaGestureSegmentDataset,
    *,
    fixed_train_video_keys: frozenset[str] | set[str],
    fixed_val_video_keys: frozenset[str] | set[str],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int], dict[str, list[str]]]:
    """
    Split gesture samples by video, keeping fixed train/val videos and randomly
    assigning the remaining videos.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be between 0 and 1, got {train_fraction}"
        )

    overlap = set(fixed_train_video_keys) & set(fixed_val_video_keys)
    if overlap:
        raise ValueError(
            "Fixed train and val video keys overlap: "
            + ", ".join(sorted(overlap))
        )

    video_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        video_to_indices[sample.video_filename].append(index)

    available_keys = {
        _video_key_from_filename(video_filename)
        for video_filename in video_to_indices
    }

    missing_fixed_train = sorted(fixed_train_video_keys - available_keys)
    missing_fixed_val = sorted(fixed_val_video_keys - available_keys)
    if missing_fixed_train:
        print(
            "Warning: fixed train videos not found in dataset: "
            + ", ".join(missing_fixed_train)
        )
    if missing_fixed_val:
        print(
            "Warning: fixed val videos not found in dataset: "
            + ", ".join(missing_fixed_val)
        )

    train_video_filenames: set[str] = set()
    val_video_filenames: set[str] = set()

    for video_key in fixed_train_video_keys:
        video_filename = _video_filename_from_key(video_key)
        if video_filename in video_to_indices:
            train_video_filenames.add(video_filename)

    for video_key in fixed_val_video_keys:
        video_filename = _video_filename_from_key(video_key)
        if video_filename in video_to_indices:
            val_video_filenames.add(video_filename)

    remaining_video_filenames = sorted(
        video_filename
        for video_filename in video_to_indices
        if video_filename not in train_video_filenames
        and video_filename not in val_video_filenames
    )

    if len(remaining_video_filenames) >= 2:
        rng = random.Random(seed)
        rng.shuffle(remaining_video_filenames)
        remaining_train_count = int(len(remaining_video_filenames) * train_fraction)
        if remaining_train_count < 1:
            remaining_train_count = 1
        elif remaining_train_count >= len(remaining_video_filenames):
            remaining_train_count = len(remaining_video_filenames) - 1

        train_video_filenames.update(
            remaining_video_filenames[:remaining_train_count]
        )
        val_video_filenames.update(
            remaining_video_filenames[remaining_train_count:]
        )
    elif len(remaining_video_filenames) == 1:
        train_video_filenames.add(remaining_video_filenames[0])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for video_filename in sorted(train_video_filenames):
        train_indices.extend(video_to_indices[video_filename])
    for video_filename in sorted(val_video_filenames):
        val_indices.extend(video_to_indices[video_filename])

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Video-level train/val split produced an empty partition: "
            f"train_samples={len(train_indices)}, val_samples={len(val_indices)}"
        )

    split_summary = {
        "fixed_train_videos": sorted(
            _video_key_from_filename(video_filename)
            for video_filename in train_video_filenames
            if _video_key_from_filename(video_filename) in fixed_train_video_keys
        ),
        "fixed_val_videos": sorted(
            _video_key_from_filename(video_filename)
            for video_filename in val_video_filenames
            if _video_key_from_filename(video_filename) in fixed_val_video_keys
        ),
        "random_train_videos": sorted(
            _video_key_from_filename(video_filename)
            for video_filename in train_video_filenames
            if _video_key_from_filename(video_filename) not in fixed_train_video_keys
        ),
        "random_val_videos": sorted(
            _video_key_from_filename(video_filename)
            for video_filename in val_video_filenames
            if _video_key_from_filename(video_filename) not in fixed_val_video_keys
        ),
    }
    return train_indices, val_indices, split_summary


def run_sampling_sanity_check(
    train_dataset: AphasiaGestureSegmentDataset,
    val_dataset: AphasiaGestureSegmentDataset,
    per_video_samples: int,
) -> None:
    expected_train = train_dataset.num_videos * per_video_samples
    actual_train = len(train_dataset)
    if actual_train != expected_train:
        raise RuntimeError(
            "Train sampling sanity check failed: "
            f"expected {expected_train} samples "
            f"({train_dataset.num_videos} videos * {per_video_samples}), "
            f"got {actual_train}"
        )

    if len(val_dataset) != len(val_dataset.sample_indices):
        raise RuntimeError(
            "Validation sanity check failed: "
            f"expected all {len(val_dataset.sample_indices)} val samples, "
            f"got active length {len(val_dataset)}"
        )

    train_dataset.resample_epoch(1)
    epoch_one_len = len(train_dataset)
    train_dataset.resample_epoch(2)
    epoch_two_len = len(train_dataset)
    if epoch_one_len != expected_train or epoch_two_len != expected_train:
        raise RuntimeError(
            "Epoch resampling sanity check failed: "
            f"expected {expected_train}, got epoch1={epoch_one_len}, epoch2={epoch_two_len}"
        )

    print(
        "Sampling sanity check passed: "
        f"train={actual_train} samples/epoch from {train_dataset.num_videos} videos "
        f"({per_video_samples}/video), val={len(val_dataset)} full samples"
    )


def main() -> None:
    args = parse_args()
    per_video_samples = (
        None if args.per_video_samples <= 0 else int(args.per_video_samples)
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=LANDMARKS_DIR,
        negative_landmarks_dir=NEGATIVE_LANDMARKS_DIR,
        num_frames=NUM_FRAMES,
        return_tensor=True,
        transcript_dir=args.transcript_dir,
        strict_transcripts=False,
        transcript_offset_seconds=args.transcript_offset_seconds,
        num_transcript_chunks=args.num_transcript_chunks,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No gesture samples were found. Check the following:\n"
            f"  - positive landmarks directory: {LANDMARKS_DIR}\n"
            f"  - negative landmarks directory: {NEGATIVE_LANDMARKS_DIR}\n"
            "  - expected layout: {video}.mp4/gest_{idx} or neg_{idx} with "
            "landmarks.json + annotation.txt"
        )

    label_to_idx, idx_to_label = build_label_mapping(dataset)
    num_classes = len(idx_to_label)
    label_counts = count_dataset_labels(dataset, label_to_idx)

    print(f"Gesture-segment samples: {len(dataset)}")
    print(
        "Label counts: "
        f"{NEGATIVE_SEGMENT_LABEL_NAME}={label_counts.get(NEGATIVE_SEGMENT_LABEL, 0)}, "
        f"{POSITIVE_SEGMENT_LABEL_NAME}={label_counts.get(POSITIVE_SEGMENT_LABEL, 0)}"
    )
    print(f"Positive landmarks root: {dataset.landmarks_root}")
    print(f"Negative landmarks root: {dataset.negative_landmarks_root}")
    if dataset.transcript_dir is not None:
        print(f"Transcript root: {dataset.transcript_dir}")
    print()

    print("First gesture samples:")
    dataset.print_gesture_samples(max_rows=10)
    print()

    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {idx_to_label}")
    print(f"Name to index: {label_to_idx}")
    print()

    train_indices, val_indices, split_summary = split_train_val_indices_by_video(
        dataset,
        fixed_train_video_keys=(
            frozenset()
            if args.no_alignment_supervision
            else FIXED_TRAIN_VIDEO_KEYS
        ),
        fixed_val_video_keys=(
            frozenset()
            if args.no_alignment_supervision
            else FIXED_VAL_VIDEO_KEYS
        ),
        train_fraction=0.8,
        seed=42,
    )

    train_dataset = AphasiaGestureSegmentDataset.from_existing(
        dataset,
        sample_indices=train_indices,
        per_video_samples=per_video_samples,
    )
    val_dataset = AphasiaGestureSegmentDataset.from_existing(
        dataset,
        sample_indices=val_indices,
        per_video_samples=None,
    )

    print("Dataset loaded successfully")
    print(f"Full gesture samples: {len(dataset)}")
    if args.no_alignment_supervision:
        print(
            "Split strategy: random 80/20 video split "
            "(fixed train/val videos disabled with --no_alignment_supervision)"
        )
    else:
        print(
            "Split strategy: fixed train/val videos with random 80/20 split "
            "for remaining videos"
        )
        print(f"Fixed train videos: {split_summary['fixed_train_videos']}")
        print(f"Fixed val videos: {split_summary['fixed_val_videos']}")
    print(f"Random train videos: {split_summary['random_train_videos']}")
    print(f"Random val videos: {split_summary['random_val_videos']}")
    print(f"Train split samples: {len(train_indices)}")
    print(f"Val split samples: {len(val_indices)}")
    if per_video_samples is not None:
        print(
            f"Train videos: {train_dataset.num_videos} | "
            f"per_video_samples: {per_video_samples} | "
            f"train samples/epoch: {len(train_dataset)}"
        )
        print(
            f"Expected train samples/epoch ~= "
            f"{train_dataset.num_videos * per_video_samples}"
        )
    else:
        print(f"Train videos: {train_dataset.num_videos} | train samples/epoch: {len(train_dataset)}")
    print(
        f"Val videos: {val_dataset.num_videos} | "
        f"val samples (full): {len(val_dataset)}"
    )
    if per_video_samples is not None:
        run_sampling_sanity_check(train_dataset, val_dataset, per_video_samples)
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=gesture_segment_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=gesture_segment_collate_fn,
    )

    

    print_model_config(args, num_classes=num_classes)

    model = LandmarkGestureClassifier(
        num_classes=num_classes,
        num_frames=NUM_FRAMES,
        num_joints=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        temporal_embed_dim=args.model_dim,
        embed_dim=args.gcn_dim,
        gcn_layers=args.gcn_layers,
        gcn_top_k=args.graph_top_k,
        dropout=args.dropout,
        mamba_layers=args.mamba_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        return_probabilities=False,
        use_gcn = True,
        use_mamba_encoder = True,
        use_transcripts=not args.no_transcripts,
        bert_model_name=args.bert_model_name,
        transcript_embed_dim=args.transcript_embed_dim,
        freeze_bert=args.freeze_bert,
        num_transcript_chunks=args.num_transcript_chunks,
        max_transcript_chunks=args.max_transcript_chunks,
        transcript_dropout_probability=args.transcript_dropout_probability,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    sanity_batch = next(iter(train_loader))
    sanity_landmarks = landmarks_to_tensor(sanity_batch["landmarks"], device)
    sanity_labels = get_batch_labels(sanity_batch, label_to_idx, device)
    sanity_transcripts = get_batch_transcripts(sanity_batch)
    sanity_transcript_chunks, sanity_transcript_chunk_mask = (
        get_batch_transcript_chunks(sanity_batch, device)
    )
    use_alignment_supervision = not args.no_alignment_supervision
    if use_alignment_supervision:
        sanity_alignment_targets = get_batch_alignment_targets(
            sanity_batch,
            device,
            args.num_transcript_chunks,
        )
        (
            sanity_logits,
            sanity_attention_weights,
            _sanity_modality_gate,
            sanity_returned_chunk_mask,
        ) = model(
            sanity_landmarks,
            transcript_chunks=sanity_transcript_chunks,
            transcript_chunk_mask=sanity_transcript_chunk_mask,
            transcripts=sanity_transcripts,
            return_attention=True,
        )
        if sanity_returned_chunk_mask is None:
            sanity_returned_chunk_mask = sanity_transcript_chunk_mask
        sanity_classification_loss = compute_classification_loss(
            sanity_logits,
            sanity_labels,
        )
        sanity_alignment_loss, sanity_num_annotated, sanity_mean_attention = (
            compute_masked_alignment_loss(
                sanity_attention_weights,
                sanity_alignment_targets,
                sanity_returned_chunk_mask,
                label_smoothing=args.alignment_label_smoothing,
            )
        )
        sanity_effective_alignment_weight = (
            args.alignment_loss_weight
            * alignment_warmup_factor(1, args.alignment_warmup_epochs)
        )
        sanity_combined_loss = (
            sanity_classification_loss
            + sanity_effective_alignment_weight * sanity_alignment_loss
        )
        sanity_target_probability = compute_alignment_target_probability(
            sanity_mean_attention,
            sanity_alignment_targets,
        )
    else:
        sanity_logits = model(
            sanity_landmarks,
            transcript_chunks=sanity_transcript_chunks,
            transcript_chunk_mask=sanity_transcript_chunk_mask,
            transcripts=sanity_transcripts,
            return_attention=False,
        )
        sanity_classification_loss = compute_classification_loss(
            sanity_logits,
            sanity_labels,
        )
        sanity_alignment_loss = sanity_logits.new_tensor(0.0)
        sanity_num_annotated = 0
        sanity_effective_alignment_weight = 0.0
        sanity_combined_loss = sanity_classification_loss
        sanity_target_probability = sanity_logits.new_tensor(0.0)

    print("One-batch sanity check:")
    print(f"  landmarks shape: {tuple(sanity_landmarks.shape)}")
    print(f"  labels shape: {tuple(sanity_labels.shape)}")
    print(f"  labels: {sanity_labels.tolist()}")
    print(f"  transcripts: {len(sanity_transcripts)} strings")
    if sanity_transcripts:
        preview = sanity_transcripts[0].replace("\n", " ")[:120]
        print(f"  first transcript preview: {preview!r}")
    if use_alignment_supervision:
        print(f"  alignment target shape: {tuple(sanity_alignment_targets.shape)}")
        print(f"  attention shape: {tuple(sanity_attention_weights.shape)}")
        print(f"  annotated alignment samples: {sanity_num_annotated}")
    else:
        print("  alignment supervision: disabled")
    print(f"  logits shape: {tuple(sanity_logits.shape)}")
    print(f"  classification loss: {sanity_classification_loss.item():.4f}")
    print(f"  alignment loss: {sanity_alignment_loss.item():.4f}")
    print(f"  effective alignment weight: {sanity_effective_alignment_weight:.4f}")
    print(f"  combined loss: {sanity_combined_loss.item():.4f}")
    if use_alignment_supervision:
        print(
            "  alignment target probability: "
            f"{sanity_target_probability.item():.4f}"
        )
    print()

    best_val_acc = 0.0
    val_misclassification_counts: dict[str, int] = defaultdict(int)
    init_results_per_epoch_csv(RESULTS_PER_EPOCH_PATH)
    print(f"Per-epoch metrics CSV: {RESULTS_PER_EPOCH_PATH}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        if hasattr(train_dataset, "resample_epoch"):
            train_dataset.resample_epoch(epoch)
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: resampled train samples={len(train_dataset)} "
                    f"(videos={train_dataset.num_videos}, "
                    f"per_video_samples={per_video_samples})"
                )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_to_idx=label_to_idx,
            num_classes=num_classes,
            idx_to_label=idx_to_label,
            epoch=epoch,
            expected_num_chunks=args.num_transcript_chunks,
            alignment_loss_weight=args.alignment_loss_weight,
            alignment_label_smoothing=args.alignment_label_smoothing,
            alignment_warmup_epochs=args.alignment_warmup_epochs,
            use_alignment_supervision=use_alignment_supervision,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            label_to_idx=label_to_idx,
            num_classes=num_classes,
            idx_to_label=idx_to_label,
            expected_num_chunks=args.num_transcript_chunks,
            alignment_loss_weight=args.alignment_loss_weight,
            alignment_label_smoothing=args.alignment_label_smoothing,
            alignment_warmup_epochs=args.alignment_warmup_epochs,
            epoch=epoch,
            use_alignment_supervision=use_alignment_supervision,
            misclassification_counts=val_misclassification_counts,
        )

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"train total loss {train_metrics['total_loss']:.4f} | "
            f"train cls loss {train_metrics['classification_loss']:.4f} | "
            f"train align loss {train_metrics['alignment_loss']:.4f} | "
            f"train annotated {int(train_metrics['annotated_alignment_samples'])} | "
            f"train align prob {train_metrics['alignment_target_probability']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | "
            f"train precision {train_metrics['precision']:.4f} | "
            f"train recall {train_metrics['recall']:.4f} | "
            f"train f1 {train_metrics['f1']:.4f} | "
            f"val total loss {val_metrics['total_loss']:.4f} | "
            f"val cls loss {val_metrics['classification_loss']:.4f} | "
            f"val align loss {val_metrics['alignment_loss']:.4f} | "
            f"val annotated {int(val_metrics['annotated_alignment_samples'])} | "
            f"val align prob {val_metrics['alignment_target_probability']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"val precision {val_metrics['precision']:.4f} | "
            f"val recall {val_metrics['recall']:.4f} | "
            f"val f1 {val_metrics['f1']:.4f}"
        )
        print_per_class_accuracy(
            val_metrics,
            prefix="Per-class val accuracy:",
        )
        append_results_per_epoch_row(
            RESULTS_PER_EPOCH_PATH,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        print()

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_metrics["accuracy"],
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "alignment_loss_weight": args.alignment_loss_weight,
                    "alignment_label_smoothing": args.alignment_label_smoothing,
                    "alignment_warmup_epochs": args.alignment_warmup_epochs,
                    "no_alignment_supervision": args.no_alignment_supervision,
                    "num_transcript_chunks": args.num_transcript_chunks,
                },
                BEST_MODEL_PATH,
            )

    print()
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Saved per-epoch metrics to: {RESULTS_PER_EPOCH_PATH}")
    print()
    print_top_misclassified_val_samples(
        val_misclassification_counts,
        num_epochs=NUM_EPOCHS,
        top_k=20,
    )
    frequent_misclassified = save_frequently_misclassified_val_samples(
        val_misclassification_counts,
        output_path=FREQUENT_MISCLASSIFICATION_PATH,
        num_epochs=NUM_EPOCHS,
    )
    print()
    print_frequently_misclassified_val_samples(
        frequent_misclassified,
        output_path=FREQUENT_MISCLASSIFICATION_PATH,
        num_epochs=NUM_EPOCHS,
    )


if __name__ == "__main__":
    main()
