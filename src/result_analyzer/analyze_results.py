#!/usr/bin/env python3
"""Compare pose-only (vis) and multimodal (multi) gesture-classifier training results."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import subprocess
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent

DEFAULT_VIS_CSV = _SCRIPT_DIR / "results_per_epoch_vis.csv"
DEFAULT_MULTI_CSV = _SCRIPT_DIR / "results_per_epoch_multi.csv"
DEFAULT_VIS_MISCLASS_PATH = (
    _SRC_DIR / "training_outputs" / "frequently_misclassified_val_samples_vis.txt"
)
DEFAULT_MULTI_MISCLASS_PATH = (
    _SRC_DIR / "training_outputs" / "frequently_misclassified_val_samples_multi.txt"
)
DEFAULT_LANDMARKS_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/"
    "gest_segments_landmarks/mediapipe"
)
DEFAULT_NEGATIVE_LANDMARKS_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/"
    "neg_gest_segments_landmarks/mediapipe"
)
DEFAULT_GEST_SEGMENTS_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments"
)
DEFAULT_NEG_GEST_SEGMENTS_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments"
)
DEFAULT_VIDEOS_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/videos"
)
DEFAULT_MISCLASSIFIED_SAMPLES_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/variant_analysis"
)

# Match train_multi.py split_train_val_indices defaults.
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_SPLIT_SEED = 42
CLIP_CONTEXT_PAD_MS = 3000

GESTURE_SEGMENT_DIR_RE = re.compile(r"^gest_(\d+)$")
NEGATIVE_SEGMENT_DIR_RE = re.compile(r"^neg_(\d+)$")

CATEGORY_DIRS = {
    "pose_fail_multi_success": "pose_fail_multi_success",
    "both_success": "both_success",
    "both_fail": "both_fail",
    "multi_fail_pose_succeed": "multi_fail_pose_succeed",
    "multi_fail": "multi_fail",
}

REQUIRED_COLUMNS = (
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
    "val_neg_acc",
)

VAL_ACC_REPORT_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pose-only and multimodal validation metrics and "
            "frequently misclassified validation samples."
        )
    )
    parser.add_argument(
        "--vis-csv",
        type=Path,
        default=DEFAULT_VIS_CSV,
        help=f"Pose-only per-epoch CSV (default: {DEFAULT_VIS_CSV})",
    )
    parser.add_argument(
        "--multi-csv",
        type=Path,
        default=DEFAULT_MULTI_CSV,
        help=f"Multimodal per-epoch CSV (default: {DEFAULT_MULTI_CSV})",
    )
    parser.add_argument(
        "--vis-misclass",
        type=Path,
        default=DEFAULT_VIS_MISCLASS_PATH,
        help=(
            "Pose-only frequently misclassified validation samples "
            f"(default: {DEFAULT_VIS_MISCLASS_PATH})"
        ),
    )
    parser.add_argument(
        "--multi-misclass",
        type=Path,
        default=DEFAULT_MULTI_MISCLASS_PATH,
        help=(
            "Multimodal frequently misclassified validation samples "
            f"(default: {DEFAULT_MULTI_MISCLASS_PATH})"
        ),
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=DEFAULT_LANDMARKS_DIR,
        help=(
            "Positive landmarks root used to rebuild the train_multi val split "
            f"(default: {DEFAULT_LANDMARKS_DIR})"
        ),
    )
    parser.add_argument(
        "--negative-landmarks-dir",
        type=Path,
        default=DEFAULT_NEGATIVE_LANDMARKS_DIR,
        help=(
            "Negative landmarks root used to rebuild the train_multi val split "
            f"(default: {DEFAULT_NEGATIVE_LANDMARKS_DIR})"
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help=(
            "Train fraction for the sample-level split matching train_multi.py "
            f"(default: {DEFAULT_TRAIN_FRACTION})"
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=(
            "RNG seed for the sample-level split matching train_multi.py "
            f"(default: {DEFAULT_SPLIT_SEED})"
        ),
    )
    parser.add_argument(
        "--gest-segments-dir",
        type=Path,
        default=DEFAULT_GEST_SEGMENTS_DIR,
        help=(
            "Positive gesture segment root with annotation.txt files "
            f"(default: {DEFAULT_GEST_SEGMENTS_DIR})"
        ),
    )
    parser.add_argument(
        "--neg-gest-segments-dir",
        type=Path,
        default=DEFAULT_NEG_GEST_SEGMENTS_DIR,
        help=(
            "Negative gesture segment root with annotation.txt files "
            f"(default: {DEFAULT_NEG_GEST_SEGMENTS_DIR})"
        ),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Full source video directory (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--misclassified-samples-dir",
        type=Path,
        default=DEFAULT_MISCLASSIFIED_SAMPLES_DIR,
        help=(
            "Output root for exported category video clips "
            f"(default: {DEFAULT_MISCLASSIFIED_SAMPLES_DIR})"
        ),
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Print comparisons without exporting category video clips.",
    )
    return parser.parse_args()


def read_results(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): " + ", ".join(missing)
            )

        for line_number, row in enumerate(reader, start=2):
            parsed_row: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "").strip()
                if not raw_value:
                    raise ValueError(
                        f"Missing value in column '{column}' at "
                        f"{csv_path} line {line_number}."
                    )
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid numeric value {raw_value!r} in column "
                        f"'{column}' at {csv_path} line {line_number}."
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value {raw_value!r} in column '{column}' "
                        f"at {csv_path} line {line_number}."
                    )
                parsed_row[column] = value
            results.append(parsed_row)

    if not results:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")

    return results


def format_epoch(epoch: float) -> str:
    return str(int(epoch)) if epoch.is_integer() else f"{epoch:g}"


def print_epoch_validation_summary(
    row: dict[str, float],
    *,
    label: str,
) -> None:
    print(f"{label} at epoch: {format_epoch(row['epoch'])}")
    print(f"\tValidation accuracy: {row['val_acc']}")
    print(f"\tValidation F1: {row['val_f1']}")
    print(f"\tValidation precision: {row['val_precision']}")
    print(f"\tValidation recall: {row['val_recall']}")
    print(f"\tValidation positive accuracy: {row['val_pos_acc']}")
    print(f"\tValidation negative accuracy: {row['val_neg_acc']}")


def print_high_val_acc_epochs(
    results: list[dict[str, float]],
    *,
    model_label: str,
    threshold: float = VAL_ACC_REPORT_THRESHOLD,
) -> None:
    high_val_epochs = [row for row in results if row["val_acc"] > threshold]
    high_val_epochs.sort(key=lambda row: (row["epoch"], row["val_acc"]))

    print(
        f"{model_label}: epochs with validation accuracy > {threshold:.2f}: "
        f"{len(high_val_epochs)}"
    )
    if not high_val_epochs:
        print()
        return

    print()
    for index, row in enumerate(high_val_epochs):
        print_epoch_validation_summary(row, label="Epoch")
        if index < len(high_val_epochs) - 1:
            print()


def print_best_validation_epoch(
    results: list[dict[str, float]],
    *,
    model_label: str,
) -> dict[str, float]:
    best_val_row = max(results, key=lambda row: (row["val_acc"], -row["epoch"]))
    average_class_gap = sum(
        abs(row["val_pos_acc"] - row["val_neg_acc"]) for row in results
    ) / len(results)

    print(f"{model_label}: overall best validation epoch")
    print_epoch_validation_summary(best_val_row, label="Best epoch")
    print(
        f"{model_label}: average absolute difference between positive and "
        f"negative validation accuracies: {average_class_gap:.6f}"
    )
    print()
    return best_val_row


def read_frequently_misclassified(path: Path) -> dict[str, str]:
    """
    Parse a frequently-misclassified report.

    Returns a mapping from sample id (``video.mp4/gest_1``) to the raw line
    body after the sample id.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Misclassification report not found: {path}")

    samples: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if not parts or not parts[0]:
            raise ValueError(
                f"Invalid misclassification line at {path}:{line_number}: "
                f"{raw_line!r}"
            )
        sample_id = parts[0].strip()
        samples[sample_id] = line
    return samples


def _list_labeled_segment_sample_ids(
    landmarks_root: Path,
    segment_dir_re: re.Pattern[str],
) -> list[str]:
    """
    Mirror AphasiaGestureSegmentDataset._scan_labeled_segments sample order.

    Sample ids use ``{video_filename}/{segment_dir_name}``, matching training
    misclassification reports.
    """
    if not landmarks_root.is_dir():
        raise FileNotFoundError(f"Landmarks directory not found: {landmarks_root}")

    sample_ids: list[str] = []
    for video_dir in sorted(landmarks_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_filename = video_dir.name
        for segment_dir in sorted(video_dir.iterdir()):
            if not segment_dir.is_dir():
                continue
            if segment_dir_re.match(segment_dir.name) is None:
                continue
            if not (segment_dir / "landmarks.json").is_file():
                continue
            if not (segment_dir / "annotation.txt").is_file():
                continue
            sample_ids.append(f"{video_filename}/{segment_dir.name}")
    return sample_ids


def list_dataset_sample_ids(
    landmarks_dir: Path,
    negative_landmarks_dir: Path,
) -> list[str]:
    """Rebuild the full dataset sample-id list in train_multi scan order."""
    positive_ids = _list_labeled_segment_sample_ids(
        landmarks_dir,
        GESTURE_SEGMENT_DIR_RE,
    )
    negative_ids = _list_labeled_segment_sample_ids(
        negative_landmarks_dir,
        NEGATIVE_SEGMENT_DIR_RE,
    )
    return positive_ids + negative_ids


def split_train_val_indices(
    num_samples: int,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """
    Match train_multi.py ``split_train_val_indices`` sample-level split.

    Shuffles every sample index once, then assigns about ``train_fraction`` of
    the full dataset to train and the remainder to validation.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be between 0 and 1, got {train_fraction}"
        )

    all_indices = list(range(num_samples))
    if len(all_indices) < 2:
        raise RuntimeError(
            "Need at least 2 gesture samples for an 80/20 train/val split, "
            f"got {len(all_indices)}"
        )

    rng = random.Random(seed)
    rng.shuffle(all_indices)

    train_count = int(num_samples * train_fraction)
    if train_count < 1:
        train_count = 1
    elif train_count >= num_samples:
        train_count = num_samples - 1

    train_indices = all_indices[:train_count]
    val_indices = all_indices[train_count:]
    return train_indices, val_indices


def load_train_val_sample_ids(
    landmarks_dir: Path,
    negative_landmarks_dir: Path,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[str], list[str]]:
    """Rebuild train/val sample-id lists matching train_multi.py."""
    all_sample_ids = list_dataset_sample_ids(landmarks_dir, negative_landmarks_dir)
    if not all_sample_ids:
        raise RuntimeError(
            "No gesture samples found while rebuilding the train/val split. Check:\n"
            f"  - positive landmarks directory: {landmarks_dir}\n"
            f"  - negative landmarks directory: {negative_landmarks_dir}"
        )

    train_indices, val_indices = split_train_val_indices(
        len(all_sample_ids),
        train_fraction=train_fraction,
        seed=seed,
    )
    train_sample_ids = [all_sample_ids[index] for index in train_indices]
    val_sample_ids = [all_sample_ids[index] for index in val_indices]
    return train_sample_ids, val_sample_ids


def video_filename_from_sample_id(sample_id: str) -> str:
    return sample_id.split("/", 1)[0]


def unique_video_filenames(sample_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sample_id in sorted(sample_ids):
        video_filename = video_filename_from_sample_id(sample_id)
        if video_filename not in seen:
            seen.add(video_filename)
            ordered.append(video_filename)
    return ordered


def print_sample_group(
    title: str,
    sample_ids: list[str],
    *,
    description: str,
) -> None:
    print(title)
    print(description)
    print(f"Sample count: {len(sample_ids)}")
    # video_filenames = unique_video_filenames(sample_ids)
    # print(f"Unique video filenames ({len(video_filenames)}):")
    # if video_filenames:
    #     for video_filename in video_filenames:
    #         print(f"  - {video_filename}")
    # else:
    #     print("  (none)")
    # if sample_ids:
    #     print("Sample ids:")
    #     for sample_id in sorted(sample_ids):
    #         print(f"  - {sample_id}")
    print()


def segment_name_from_sample_id(sample_id: str) -> str:
    if "/" not in sample_id:
        raise ValueError(f"Invalid sample id (expected video/segment): {sample_id!r}")
    return sample_id.split("/", 1)[1]


def resolve_sample_annotation_path(
    sample_id: str,
    *,
    gest_segments_dir: Path,
    neg_gest_segments_dir: Path,
) -> Path:
    video_filename = video_filename_from_sample_id(sample_id)
    segment_name = segment_name_from_sample_id(sample_id)
    if segment_name.startswith("neg_"):
        root = neg_gest_segments_dir
    else:
        root = gest_segments_dir
    return root / video_filename / segment_name / "annotation.txt"


def parse_annotation_timestamps(annotation_path: Path) -> tuple[int, int]:
    """Read ``start_ms`` and ``end_ms`` from a segment ``annotation.txt``."""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    start_ms: int | None = None
    end_ms: int | None = None
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("start_ms:"):
            start_ms = int(line.split(":", 1)[1].strip())
        elif line.startswith("end_ms:"):
            end_ms = int(line.split(":", 1)[1].strip())

    if start_ms is None or end_ms is None:
        raise ValueError(
            f"Annotation is missing start_ms/end_ms: {annotation_path}"
        )
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError(
            f"Invalid timestamps in {annotation_path}: "
            f"start_ms={start_ms}, end_ms={end_ms}"
        )
    return start_ms, end_ms


def resolve_ffmpeg_executable() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    candidate = Path(
        "/home/z/zaimazarnaz/miniconda3/envs/aphasia_project/bin/ffmpeg"
    )
    if candidate.is_file():
        return str(candidate)

    raise RuntimeError(
        "ffmpeg was not found on PATH. Activate an environment with ffmpeg "
        "(e.g. aphasia_project) before exporting video clips."
    )


def resolve_ffprobe_executable(ffmpeg_executable: str) -> str:
    ffmpeg_path = Path(ffmpeg_executable)
    sibling = ffmpeg_path.with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path

    raise RuntimeError(
        "ffprobe was not found next to ffmpeg or on PATH. "
        "Needed to read full-video duration for clip padding."
    )


def get_video_duration_ms(
    source_video_path: Path,
    *,
    ffprobe_executable: str,
) -> int:
    """Return full-video duration in milliseconds via ffprobe."""
    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source_video_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(
            f"ffprobe failed while reading duration for {source_video_path}: "
            f"{stderr}"
        )

    raw_duration = (completed.stdout or "").strip()
    try:
        duration_sec = float(raw_duration)
    except ValueError as error:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration {raw_duration!r} for "
            f"{source_video_path}"
        ) from error

    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise RuntimeError(
            f"Invalid video duration {duration_sec} for {source_video_path}"
        )
    return int(round(duration_sec * 1000.0))


def expand_clip_timestamps(
    start_ms: int,
    end_ms: int,
    *,
    video_duration_ms: int,
    pad_ms: int = CLIP_CONTEXT_PAD_MS,
) -> tuple[int, int]:
    """
    Expand a segment window by ``pad_ms`` on each side.

    ``start_ms = max(0, start_ms - pad_ms)``
    ``end_ms = min(video_duration_ms, end_ms + pad_ms)``
    """
    if video_duration_ms <= 0:
        raise ValueError(f"video_duration_ms must be positive, got {video_duration_ms}")

    expanded_start_ms = max(0, start_ms - pad_ms)
    expanded_end_ms = min(video_duration_ms, end_ms + pad_ms)
    if expanded_end_ms <= expanded_start_ms:
        raise ValueError(
            "Expanded clip window is empty after padding: "
            f"start_ms={expanded_start_ms}, end_ms={expanded_end_ms}, "
            f"video_duration_ms={video_duration_ms}"
        )
    return expanded_start_ms, expanded_end_ms


def cut_video_segment(
    *,
    source_video_path: Path,
    destination_video_path: Path,
    start_ms: int,
    end_ms: int,
    ffmpeg_executable: str,
) -> None:
    """Cut ``[start_ms, end_ms]`` from ``source_video_path`` into an mp4 with audio."""
    destination_video_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_video_path.exists():
        if destination_video_path.is_dir():
            shutil.rmtree(destination_video_path)
        else:
            destination_video_path.unlink()

    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    command = [
        ffmpeg_executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(source_video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(destination_video_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg failed while cutting "
            f"{source_video_path} [{start_ms}, {end_ms}] -> "
            f"{destination_video_path}: {stderr}"
        )


def export_sample_videos(
    sample_ids: list[str],
    *,
    category_dir: Path,
    gest_segments_dir: Path,
    neg_gest_segments_dir: Path,
    videos_dir: Path,
    ffmpeg_executable: str,
) -> tuple[int, list[str]]:
    """
    Export one mp4 clip per sample.

    Destination layout:
      ``{category_dir}/{video_filename}/{segment_name}.mp4``

    Each clip is padded by ``CLIP_CONTEXT_PAD_MS`` on both sides, clamped to
    ``[0, video_duration_ms]``.
    """
    category_dir.mkdir(parents=True, exist_ok=True)
    exported_count = 0
    missing_samples: list[str] = []
    ffprobe_executable = resolve_ffprobe_executable(ffmpeg_executable)
    video_duration_cache: dict[Path, int] = {}

    for sample_id in sample_ids:
        video_filename = video_filename_from_sample_id(sample_id)
        segment_name = segment_name_from_sample_id(sample_id)
        annotation_path = resolve_sample_annotation_path(
            sample_id,
            gest_segments_dir=gest_segments_dir,
            neg_gest_segments_dir=neg_gest_segments_dir,
        )
        source_video_path = videos_dir / video_filename
        destination_video_path = (
            category_dir / video_filename / f"{segment_name}.mp4"
        )

        if not annotation_path.is_file() or not source_video_path.is_file():
            missing_samples.append(sample_id)
            continue

        try:
            start_ms, end_ms = parse_annotation_timestamps(annotation_path)
            if source_video_path not in video_duration_cache:
                video_duration_cache[source_video_path] = get_video_duration_ms(
                    source_video_path,
                    ffprobe_executable=ffprobe_executable,
                )
            start_ms, end_ms = expand_clip_timestamps(
                start_ms,
                end_ms,
                video_duration_ms=video_duration_cache[source_video_path],
            )
            cut_video_segment(
                source_video_path=source_video_path,
                destination_video_path=destination_video_path,
                start_ms=start_ms,
                end_ms=end_ms,
                ffmpeg_executable=ffmpeg_executable,
            )
        except (OSError, ValueError, RuntimeError):
            missing_samples.append(sample_id)
            continue

        exported_count += 1

    return exported_count, missing_samples


def copy_category_samples(
    categories: dict[str, dict[str, list[str]]],
    *,
    output_root: Path,
    gest_segments_dir: Path,
    neg_gest_segments_dir: Path,
    videos_dir: Path,
) -> None:
    ffmpeg_executable = resolve_ffmpeg_executable()

    print("Exporting category video clips")
    print(f"Output root: {output_root}")
    print(f"Source videos root: {videos_dir}")
    print(f"Positive segments root: {gest_segments_dir}")
    print(f"Negative segments root: {neg_gest_segments_dir}")
    print(f"ffmpeg: {ffmpeg_executable}")
    print()

    for category_key, split_samples in categories.items():
        category_name = CATEGORY_DIRS[category_key]
        category_dir = output_root / category_name
        sample_ids = sorted(
            set(split_samples["train"]) | set(split_samples["val"])
        )
        exported_count, missing_samples = export_sample_videos(
            sample_ids,
            category_dir=category_dir,
            gest_segments_dir=gest_segments_dir,
            neg_gest_segments_dir=neg_gest_segments_dir,
            videos_dir=videos_dir,
            ffmpeg_executable=ffmpeg_executable,
        )
        print(f"{category_name}/")
        print(f"  train samples: {len(split_samples['train'])}")
        print(f"  val samples: {len(split_samples['val'])}")
        print(f"  requested samples: {len(sample_ids)}")
        print(f"  exported videos: {exported_count}")
        if missing_samples:
            print(f"  missing/failed samples: {len(missing_samples)}")
            for sample_id in missing_samples:
                print(f"    - {sample_id}")
        print()


def build_split_misclassification_categories(
    vis_samples: dict[str, str],
    multi_samples: dict[str, str],
    split_sample_ids: list[str],
) -> dict[str, list[str]]:
    """Categorize samples within one split (train or val)."""
    split_ids = set(split_sample_ids)
    vis_fail = set(vis_samples) & split_ids
    multi_fail = set(multi_samples) & split_ids

    return {
        "pose_fail_multi_success": sorted(vis_fail - multi_fail),
        "both_success": sorted(split_ids - vis_fail - multi_fail),
        "both_fail": sorted(vis_fail & multi_fail),
        "multi_fail_pose_succeed": sorted(multi_fail - vis_fail),
        "multi_fail": sorted(multi_fail),
    }


def build_misclassification_categories(
    vis_samples: dict[str, str],
    multi_samples: dict[str, str],
    train_sample_ids: list[str],
    val_sample_ids: list[str],
) -> dict[str, dict[str, list[str]]]:
    train_categories = build_split_misclassification_categories(
        vis_samples,
        multi_samples,
        train_sample_ids,
    )
    val_categories = build_split_misclassification_categories(
        vis_samples,
        multi_samples,
        val_sample_ids,
    )
    return {
        category_key: {
            "train": train_categories[category_key],
            "val": val_categories[category_key],
        }
        for category_key in CATEGORY_DIRS
    }


def compare_frequently_misclassified_samples(
    categories: dict[str, dict[str, list[str]]],
    *,
    train_sample_ids: list[str],
    val_sample_ids: list[str],
    train_fraction: float,
    split_seed: int,
    vis_report_count: int,
    multi_report_count: int,
) -> None:
    print("Frequently misclassified sample comparison")
    print(
        "Split: sample-level "
        f"{train_fraction:.0%}/{(1.0 - train_fraction):.0%} "
        f"(seed={split_seed}, matching train_multi.py)"
    )
    print(f"Rebuilt train samples: {len(train_sample_ids)}")
    print(f"Rebuilt val samples: {len(val_sample_ids)}")
    print(f"Pose-only report samples: {vis_report_count}")
    print(f"Multimodal report samples: {multi_report_count}")
    print()

    for category_key, category_name in CATEGORY_DIRS.items():
        train_ids = categories[category_key]["train"]
        val_ids = categories[category_key]["val"]
        print(f"{category_name}/")
        print(f"  train samples: {len(train_ids)}")
        print(f"  val samples: {len(val_ids)}")
        print()


def main() -> None:
    args = parse_args()

    vis_results = read_results(args.vis_csv)
    multi_results = read_results(args.multi_csv)

    print("=" * 72)
    print("Best validation epochs")
    print("=" * 72)
    print()

    print_high_val_acc_epochs(vis_results, model_label="Pose-only (vis)")
    print()
    print_best_validation_epoch(vis_results, model_label="Pose-only (vis)")

    print_high_val_acc_epochs(multi_results, model_label="Multimodal (multi)")
    print()
    print_best_validation_epoch(multi_results, model_label="Multimodal (multi)")

    train_sample_ids, val_sample_ids = load_train_val_sample_ids(
        args.landmarks_dir,
        args.negative_landmarks_dir,
        train_fraction=args.train_fraction,
        seed=args.split_seed,
    )
    vis_samples = read_frequently_misclassified(args.vis_misclass)
    multi_samples = read_frequently_misclassified(args.multi_misclass)
    categories = build_misclassification_categories(
        vis_samples,
        multi_samples,
        train_sample_ids,
        val_sample_ids,
    )

    print("=" * 72)
    print("Frequently misclassified sample comparison")
    print("=" * 72)
    print()
    compare_frequently_misclassified_samples(
        categories,
        train_sample_ids=train_sample_ids,
        val_sample_ids=val_sample_ids,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        vis_report_count=len(vis_samples),
        multi_report_count=len(multi_samples),
    )

    if not args.skip_copy:
        print("=" * 72)
        copy_category_samples(
            categories,
            output_root=args.misclassified_samples_dir,
            gest_segments_dir=args.gest_segments_dir,
            neg_gest_segments_dir=args.neg_gest_segments_dir,
            videos_dir=args.videos_dir,
        )


if __name__ == "__main__":
    main()
