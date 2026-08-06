"""
Extract gesture-annotated video segments as PNG frame sequences.

Loads AphasiaGestureEAFDataset, reads each gesture interval from the matched
videos, and writes frames to disk under OUTPUT_ROOT.

Example output path:
    .../gest_segments/65-1.mp4/gest_0/0000.png
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment, misc]

_PREPROCESS_DIR = Path(__file__).resolve().parent
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))

_DATASET_MODULE_CANDIDATES = (
    "dataloder",
    "dataset",
    "aphasia_gesture_dataset",
    "dataloader",
)


def _import_aphasia_gesture_dataset():
    """Import AphasiaGestureEAFDataset from the dataset module in pre_process/."""
    for module_name in _DATASET_MODULE_CANDIDATES:
        module_path = _PREPROCESS_DIR / f"{module_name}.py"
        if not module_path.is_file():
            continue

        module = importlib.import_module(module_name)
        dataset_cls = getattr(module, "AphasiaGestureEAFDataset", None)
        if dataset_cls is not None:
            return dataset_cls

        raise ImportError(
            f"Found {module_path} but it does not define AphasiaGestureEAFDataset."
        )

    searched = ", ".join(f"{name}.py" for name in _DATASET_MODULE_CANDIDATES)
    raise ImportError(
        f"Could not find a dataset module in {_PREPROCESS_DIR}. "
        f"Expected one of: {searched}"
    )


AphasiaGestureEAFDataset = _import_aphasia_gesture_dataset()

VIDEO_DIR = "/data/quantization/zaima/aphasiabank/neural2/control/videos"

ANNOTATION_DIRS = [
    "/data/quantization/zaima/aphasiabank/neural2/control/annotations/aphasia",
    "/data/quantization/zaima/aphasiabank/neural2/control/annotations/NBD",
]

OUTPUT_ROOT = "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments"

OVERWRITE = False


def _list_segment_frame_paths(output_dir: Path) -> list[Path]:
    """Return sorted PNG frame files named with numeric indices."""
    return sorted(
        (
            path
            for path in output_dir.glob("*.png")
            if path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )


def save_frames_as_png(
    frames: torch.Tensor,
    output_dir: str | Path,
    overwrite: bool = False,
) -> int:
    """
    Save a [T, C, H, W] RGB tensor as zero-padded {idx}.png files.

    Returns the number of frames saved, or the number of existing frames when
    skipping because output already exists and overwrite is False.
    """
    if frames.ndim != 4:
        raise ValueError(
            f"Expected frames with shape [T, C, H, W], got {tuple(frames.shape)}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_frames = _list_segment_frame_paths(output_path)
    if existing_frames and not overwrite:
        return len(existing_frames)

    video = frames.detach().cpu().clamp(0.0, 1.0)
    video_uint8 = (video * 255.0).round().to(torch.uint8)
    video_hwc = video_uint8.permute(0, 2, 3, 1).numpy()

    num_saved = 0
    for frame_idx in range(video_hwc.shape[0]):
        frame_bgr = cv2.cvtColor(video_hwc[frame_idx], cv2.COLOR_RGB2BGR)
        frame_path = output_path / f"{frame_idx:04d}.png"
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise RuntimeError(f"Failed to write frame image: {frame_path}")
        num_saved += 1

    return num_saved


def write_annotation_metadata(annotation: dict, output_dir: str | Path) -> None:
    """Write gesture metadata for one segment to annotation.txt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_tier_values = annotation.get("raw_tier_values", {})
    if not isinstance(raw_tier_values, dict):
        raw_tier_values = {}

    lines = [
        f"video_path: {annotation.get('video_path')}",
        f"annotation_path: {annotation.get('annotation_path')}",
        f"video_filename: {annotation.get('video_filename')}",
        f"annotation_filename: {annotation.get('annotation_filename')}",
        f"start_ms: {annotation.get('start_ms')}",
        f"end_ms: {annotation.get('end_ms')}",
        f"duration_ms: {annotation.get('duration_ms')}",
        f"gesture_video_index: {annotation.get('gesture_video_index')}",
        f"marker: {annotation.get('marker')}",
        f"task: {annotation.get('task')}",
        f"gesture_type: {annotation.get('gesture_type')}",
        f"function: {annotation.get('function')}",
        f"left_hand_label: {annotation.get('left_hand_label')}",
        f"left_hand_shape: {annotation.get('left_hand_shape')}",
        f"right_hand_label: {annotation.get('right_hand_label')}",
        f"right_hand_shape: {annotation.get('right_hand_shape')}",
        f"raw_tier_values: {json.dumps(raw_tier_values, ensure_ascii=True)}",
    ]
    (output_path / "annotation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_video_gesture_indices(dataset) -> dict[int, int]:
    """
    Map each global dataset index to the gesture index within its video.

    Gestures for the same video are ordered by start time, then dataset index.
    """
    gestures_by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for global_idx, sample in enumerate(dataset.samples):
        gestures_by_video[sample.video_path.name].append(
            (sample.start_ms, global_idx)
        )

    index_map: dict[int, int] = {}
    for gestures in gestures_by_video.values():
        for gesture_idx, (_, global_idx) in enumerate(
            sorted(gestures, key=lambda item: (item[0], item[1]))
        ):
            index_map[global_idx] = gesture_idx

    return index_map


def build_segment_output_dir(annotation: dict, gesture_idx: int) -> Path:
    """Return the output directory for one gesture segment."""
    return Path(OUTPUT_ROOT) / annotation["video_filename"] / f"gest_{gesture_idx}"


def _segment_has_existing_frames(output_dir: Path) -> bool:
    return output_dir.exists() and bool(_list_segment_frame_paths(output_dir))


def main() -> None:
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = AphasiaGestureEAFDataset(
        video_dir=VIDEO_DIR,
        annotation_dir=ANNOTATION_DIRS,
        recursive=False,
        strict=False,
        allow_multiple_annotations_per_video=True,
        num_frames=None,
        frame_size=None,
        rgb=True,
        return_tensor=True,
    )

    if hasattr(dataset, "pairs"):
        print(f"Matched video/EAF pairs: {len(dataset.pairs)}")
    print(f"Gesture segments: {len(dataset)}")
    print(f"Output root: {output_root}")
    print(f"Overwrite existing segments: {OVERWRITE}")
    print()

    video_gesture_indices = build_video_gesture_indices(dataset)

    total_segments = 0
    saved_segments = 0
    skipped_existing_segments = 0
    failed_segments = 0

    def _log(message: str) -> None:
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    segment_indices = range(len(dataset))
    progress_bar = (
        tqdm(segment_indices, desc="Extracting gesture segments", unit="segment")
        if tqdm is not None
        else segment_indices
    )

    for idx in progress_bar:
        total_segments += 1

        try:
            sample = dataset[idx]
            frames = sample["frames"]
            annotation = sample["annotation"]
            gesture_video_index = video_gesture_indices[idx]
            annotation["gesture_video_index"] = gesture_video_index
            output_dir = build_segment_output_dir(annotation, gesture_video_index)

            if _segment_has_existing_frames(output_dir) and not OVERWRITE:
                skipped_existing_segments += 1
            else:
                save_frames_as_png(frames, output_dir, overwrite=OVERWRITE)
                write_annotation_metadata(annotation, output_dir)
                saved_segments += 1

        except Exception as exc:
            failed_segments += 1
            video_filename = "unknown"
            start_ms = "unknown"
            end_ms = "unknown"
            try:
                segment_annotation = dataset.get_annotation(idx)
                video_filename = segment_annotation.video_path.name
                start_ms = segment_annotation.start_ms
                end_ms = segment_annotation.end_ms
            except Exception:
                pass

            _log(
                f"Failed segment idx={idx} "
                f"video={video_filename} "
                f"start_ms={start_ms} end_ms={end_ms}: {exc}"
            )

        if tqdm is not None:
            progress_bar.set_postfix(
                saved=saved_segments,
                skipped=skipped_existing_segments,
                failed=failed_segments,
                refresh=False,
            )

    print()
    print("Extraction summary:")
    print(f"  Total segments: {total_segments}")
    print(f"  Saved segments: {saved_segments}")
    print(f"  Skipped existing segments: {skipped_existing_segments}")
    print(f"  Failed segments: {failed_segments}")
    print(f"  Output root: {output_root}")


if __name__ == "__main__":
    main()
