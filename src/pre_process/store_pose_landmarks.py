"""
Extract and store OpenPose BODY_25 pose landmarks for pre-extracted gesture segments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment, misc]

_PREPROCESS_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PREPROCESS_DIR.parent / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from openpose_extractor import (
    DEFAULT_OPENPOSE_ROOT,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    extract_openpose_landmarks,
)

GESTURE_SEGMENT_DIR_RE = re.compile(r"^(gest|neg)_\d+$")
DEFAULT_SOURCE_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments"
)
DEFAULT_DEST_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/"
    "gest_segments_landmarks/openpose"
)
DEFAULT_NEG_SOURCE_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments"
)
DEFAULT_NEG_DEST_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/"
    "neg_gest_segments_landmarks/openpose"
)
LANDMARK_NAMES = ["x", "y", "z", "confidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and store OpenPose BODY_25 landmarks for pre-extracted "
            "gesture segment PNG directories."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(DEFAULT_NEG_SOURCE_ROOT),
        help="Root directory containing extracted segment frame PNGs.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(DEFAULT_NEG_DEST_ROOT),
        help="Root directory where landmarks.json files will be written.",
    )
    parser.add_argument(
        "--openpose-root",
        type=Path,
        default=DEFAULT_OPENPOSE_ROOT,
        help="Path to the compiled OpenPose repository.",
    )
    parser.add_argument(
        "--num-gpu-start",
        type=int,
        default=1,
        help="GPU index for OpenPose (default: 1; use 0 if GPU 0 is free).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run OpenPose on CPU instead of GPU.",
    )
    parser.add_argument(
        "--net-resolution",
        default="-1x368",
        help="OpenPose net resolution (lower uses less VRAM).",
    )
    parser.add_argument(
        "--scale-number",
        type=int,
        default=1,
        help="OpenPose scale_number (1 uses least GPU memory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing landmarks.json files.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional limit on number of segments to process (for debugging).",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Also extract OpenPose hand/finger landmarks (21 keypoints per hand).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-segment skip and progress details.",
    )
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort frame PNGs numerically when possible."""
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    if stem.startswith("frame_"):
        suffix = stem.removeprefix("frame_")
        if suffix.isdigit():
            return (1, int(suffix))
    return (2, stem.lower())


def list_segment_frame_paths(segment_dir: Path) -> list[Path]:
    frame_paths = [
        path
        for path in segment_dir.glob("*.png")
        if path.is_file()
    ]
    return sorted(frame_paths, key=frame_sort_key)


def load_segment_frames(frame_paths: list[Path]) -> np.ndarray:
    """Load PNG frames into an RGB array with shape [T, H, W, 3]."""
    frames: list[np.ndarray] = []

    cv2 = None
    for frame_path in frame_paths:
        frame_rgb: np.ndarray | None = None

        if cv2 is not False:
            try:
                if cv2 is None:
                    import cv2 as cv2_module

                    cv2 = cv2_module
                frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            except ImportError:
                cv2 = False

        if frame_rgb is None:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "Need opencv-python or pillow to load segment PNG frames. "
                    "Install one with: pip install opencv-python  OR  pip install pillow"
                ) from exc

            with Image.open(frame_path) as image:
                frame_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        frames.append(frame_rgb)

    if not frames:
        raise ValueError(f"No frames loaded from: {segment_dir_display(frame_paths)}")

    return np.stack(frames, axis=0)


def segment_dir_display(frame_paths: list[Path]) -> str:
    if frame_paths:
        return str(frame_paths[0].parent)
    return "<empty>"


def json_safe(value: Any) -> Any:
    """Convert values to JSON-safe Python types, mapping NaN/Inf to null."""
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        if math.isnan(float_value) or math.isinf(float_value):
            return None
        return float_value

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    return value


def build_landmarks_payload(
    *,
    video_filename: str,
    segment_name: str,
    source_segment_dir: Path,
    frame_paths: list[Path],
    landmarks: np.ndarray,
    num_joints: int,
    hand_left: np.ndarray | None = None,
    hand_right: np.ndarray | None = None,
) -> dict[str, Any]:
    if landmarks.ndim != 3 or landmarks.shape[1:] != (num_joints, 4):
        raise ValueError(
            f"Expected landmarks with shape [T, {num_joints}, 4], "
            f"got {tuple(landmarks.shape)}"
        )

    if landmarks.shape[0] != len(frame_paths):
        raise ValueError(
            "Frame count mismatch between loaded frames and landmarks: "
            f"frames={len(frame_paths)}, landmarks={landmarks.shape[0]}"
        )

    if hand_left is not None:
        if hand_left.shape != (len(frame_paths), NUM_HAND_LANDMARKS, 4):
            raise ValueError(
                "Expected hand_left with shape "
                f"[T, {NUM_HAND_LANDMARKS}, 4], got {tuple(hand_left.shape)}"
            )
    if hand_right is not None:
        if hand_right.shape != (len(frame_paths), NUM_HAND_LANDMARKS, 4):
            raise ValueError(
                "Expected hand_right with shape "
                f"[T, {NUM_HAND_LANDMARKS}, 4], got {tuple(hand_right.shape)}"
            )

    frames_payload = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame_entry = {
            "frame_index": frame_index,
            "frame_file": frame_path.name,
            "landmarks": json_safe(landmarks[frame_index].tolist()),
        }
        if hand_left is not None:
            frame_entry["hand_left"] = json_safe(hand_left[frame_index].tolist())
        if hand_right is not None:
            frame_entry["hand_right"] = json_safe(hand_right[frame_index].tolist())
        frames_payload.append(frame_entry)

    payload = {
        "video_filename": video_filename,
        "segment_name": segment_name,
        "source_segment_dir": str(source_segment_dir),
        "num_frames": len(frame_paths),
        "landmark_shape": [len(frame_paths), num_joints, 4],
        "landmark_format": LANDMARK_NAMES,
        "extractor": (
            "openpose_body25_hand21" if hand_left is not None else "openpose_body25"
        ),
        "frames": frames_payload,
    }
    if hand_left is not None:
        payload["hand_landmark_shape"] = [len(frame_paths), NUM_HAND_LANDMARKS, 4]
    return payload


def discover_segment_dirs(source_root: Path) -> list[tuple[Path, str, str]]:
    """
    Return (segment_dir, video_filename, segment_name) for all gesture segments.
    """
    segment_entries: list[tuple[Path, str, str]] = []

    for video_dir in sorted(source_root.iterdir(), key=lambda path: path.name.lower()):
        if not video_dir.is_dir():
            continue

        video_filename = video_dir.name
        for segment_dir in sorted(video_dir.iterdir(), key=lambda path: path.name.lower()):
            if not segment_dir.is_dir():
                continue
            if not GESTURE_SEGMENT_DIR_RE.match(segment_dir.name):
                continue

            segment_entries.append(
                (segment_dir, video_filename, segment_dir.name)
            )

    return segment_entries


def copy_annotation_file(
    source_segment_dir: Path,
    dest_segment_dir: Path,
    verbose: bool,
) -> None:
    source_annotation = source_segment_dir / "annotation.txt"
    dest_annotation = dest_segment_dir / "annotation.txt"

    if not source_annotation.is_file():
        print(
            f"Warning: missing annotation.txt in source segment: {source_segment_dir}"
        )
        return

    shutil.copy2(source_annotation, dest_annotation)
    if verbose:
        print(f"Copied annotation: {dest_annotation}")


def process_segment(
    *,
    source_segment_dir: Path,
    dest_segment_dir: Path,
    video_filename: str,
    segment_name: str,
    openpose_root: Path,
    num_gpu: int,
    num_gpu_start: int,
    net_resolution: str,
    scale_number: int,
    enable_hand: bool,
    overwrite: bool,
    verbose: bool,
) -> str:
    """
    Process one gesture segment.

    Returns:
        One of: "processed", "skipped", "failed_no_frames"
    """
    dest_landmarks_path = dest_segment_dir / "landmarks.json"

    if dest_landmarks_path.exists() and not overwrite:
        if verbose:
            print(f"Skipping existing landmarks: {dest_landmarks_path}")
        return "skipped"

    frame_paths = list_segment_frame_paths(source_segment_dir)
    if not frame_paths:
        print(f"Warning: no PNG frames found in {source_segment_dir}")
        return "failed_no_frames"

    frames = load_segment_frames(frame_paths)
    extracted = extract_openpose_landmarks(
        frames,
        openpose_root=openpose_root,
        enable_hand=enable_hand,
        num_gpu=num_gpu,
        num_gpu_start=num_gpu_start,
        net_resolution=net_resolution,
        scale_number=scale_number,
        verbose=verbose,
    )
    landmarks = extracted["pose_body25"]
    assert isinstance(landmarks, np.ndarray)

    payload = build_landmarks_payload(
        video_filename=video_filename,
        segment_name=segment_name,
        source_segment_dir=source_segment_dir,
        frame_paths=frame_paths,
        landmarks=landmarks,
        num_joints=NUM_POSE_LANDMARKS,
        hand_left=extracted.get("hand_left"),
        hand_right=extracted.get("hand_right"),
    )

    dest_segment_dir.mkdir(parents=True, exist_ok=True)
    with dest_landmarks_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(payload), output_file, indent=2)
        output_file.write("\n")

    copy_annotation_file(source_segment_dir, dest_segment_dir, verbose=verbose)

    if verbose:
        print(f"Saved landmarks: {dest_landmarks_path}")

    return "processed"


def run_batch(
    *,
    source_root: Path,
    dest_root: Path,
    openpose_root: Path,
    num_gpu: int,
    num_gpu_start: int,
    net_resolution: str,
    scale_number: int,
    enable_hand: bool,
    overwrite: bool,
    max_segments: int | None,
    verbose: bool,
) -> None:
    source_root = source_root.expanduser().resolve()
    dest_root = dest_root.expanduser().resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)

    segment_entries = discover_segment_dirs(source_root)
    if max_segments is not None:
        segment_entries = segment_entries[:max_segments]

    video_dirs = sorted(
        {video_filename for _, video_filename, _ in segment_entries},
        key=str.lower,
    )

    print(f"Source root: {source_root}")
    print(f"Destination root: {dest_root}")
    print(f"OpenPose root: {openpose_root}")
    print(f"GPU mode: {'CPU' if num_gpu == 0 else f'CUDA gpu={num_gpu_start}'}")
    print(f"Landmarks per frame: {NUM_POSE_LANDMARKS} body (BODY_25)")
    if enable_hand:
        print(f"Hand landmarks per frame: {NUM_HAND_LANDMARKS} per hand")
    print(f"Video directories found: {len(video_dirs)}")
    print(f"Segment directories found: {len(segment_entries)}")
    print()

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    iterator = segment_entries
    if tqdm is not None:
        iterator = tqdm(segment_entries, desc="Storing OpenPose landmarks", unit="segment")

    for index, (source_segment_dir, video_filename, segment_name) in enumerate(
        iterator, start=1
    ):
        dest_segment_dir = dest_root / video_filename / segment_name
        if verbose or tqdm is None:
            print(
                f"Processing segment {index}/{len(segment_entries)}: "
                f"{source_segment_dir}"
            )

        try:
            status = process_segment(
                source_segment_dir=source_segment_dir,
                dest_segment_dir=dest_segment_dir,
                video_filename=video_filename,
                segment_name=segment_name,
                openpose_root=openpose_root,
                num_gpu=num_gpu,
                num_gpu_start=num_gpu_start,
                net_resolution=net_resolution,
                scale_number=scale_number,
                enable_hand=enable_hand,
                overwrite=overwrite,
                verbose=verbose,
            )
        except Exception as exc:
            failed_count += 1
            print(f"Failed segment: {source_segment_dir}")
            print(f"  Error: {exc}")
            continue

        if status == "processed":
            processed_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

        if tqdm is not None:
            iterator.set_postfix(
                processed=processed_count,
                skipped=skipped_count,
                failed=failed_count,
                refresh=False,
            )

    print()
    print("Summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total segments scanned: {len(segment_entries)}")


def main() -> None:
    args = parse_args()

    run_batch(
        source_root=args.source_root,
        dest_root=args.dest_root,
        openpose_root=args.openpose_root.expanduser().resolve(),
        num_gpu=0 if args.cpu else 1,
        num_gpu_start=args.num_gpu_start,
        net_resolution=args.net_resolution,
        scale_number=args.scale_number,
        enable_hand=args.hand,
        overwrite=args.overwrite,
        max_segments=args.max_segments,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
