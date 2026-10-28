"""
Extract and store MediaPipe pose landmarks for pre-extracted gesture segments.

Temporarily supports annotation-only mode: skip landmark extraction and refresh
``annotation.txt`` files with EAF ``ANNOTATION_ID`` values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment, misc]

_PREPROCESS_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PREPROCESS_DIR.parent / "model"
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from dataloder import parse_segment_annotation_file  # noqa: E402

from mediapipe_extractor import NUM_POSE_LANDMARKS, extract_pose_landmarks  # noqa: E402

GESTURE_SEGMENT_DIR_RE = re.compile(r"^(gest|neg)_\d+$")
DIRECT_CLIP_FILENAME_RE = re.compile(
    r"^(?P<video_key>\d+-\d+)_(?P<gest_idx>\d+)_(?P<label>[A-Za-z]+)"
    r"\.(?P<ext>mp4|mov|avi|mkv|webm)$",
    re.IGNORECASE,
)
DIRECT_CLIP_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_SOURCE_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/unaligned_samples/gest_clips"
)
DEFAULT_DEST_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/unaligned_samples/landmarks"
)
DEFAULT_NEG_SOURCE_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments"
)
DEFAULT_NEG_DEST_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/"
    "neg_gest_segments_landmarks/mediapipe"
)
LANDMARK_NAMES = ["x", "y", "z", "visibility"]

# Temporary: skip MediaPipe landmark extraction and only refresh annotation.txt.
ANNOTATION_ONLY_MODE = False

POSITIVE_ANNOTATION_FIELD_ORDER = (
    "video_path",
    "annotation_path",
    "video_filename",
    "annotation_filename",
    "annotation_id",
    "start_ms",
    "end_ms",
    "duration_ms",
    "gesture_video_index",
    "marker",
    "task",
    "gesture_type",
    "function",
    "left_hand_label",
    "left_hand_shape",
    "right_hand_label",
    "right_hand_shape",
    "raw_tier_values",
)

NEGATIVE_ANNOTATION_FIELD_ORDER = (
    "video_path",
    "annotation_path",
    "video_filename",
    "annotation_filename",
    "annotation_id",
    "start_ms",
    "end_ms",
    "duration_ms",
    "negative_video_index",
    "marker",
    "task",
    "gesture_type",
    "function",
    "left_hand_label",
    "left_hand_shape",
    "right_hand_label",
    "right_hand_shape",
    "source",
    "positive_overlap",
    "length_frames",
    "fps",
    "gap_start_ms",
    "gap_end_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and store MediaPipe pose landmarks for pre-extracted "
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
        "--model-complexity",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="MediaPipe pose model complexity: 0=lite, 1=full, 2=heavy.",
    )
    parser.add_argument(
        "--smooth-landmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth landmarks across frames (legacy MediaPipe solutions API).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe pose detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe pose tracking confidence.",
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
        "--direct-clip-parse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Read pre-cut gesture clip MP4 files directly from gest_clips "
            "instead of PNG frame directories under neg_gest_segments."
        ),
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

    frames_payload = []
    for frame_index, frame_path in enumerate(frame_paths):
        frames_payload.append(
            {
                "frame_index": frame_index,
                "frame_file": frame_path.name,
                "landmarks": json_safe(landmarks[frame_index].tolist()),
            }
        )

    return {
        "video_filename": video_filename,
        "segment_name": segment_name,
        "source_segment_dir": str(source_segment_dir),
        "num_frames": len(frame_paths),
        "landmark_shape": [len(frame_paths), num_joints, 4],
        "landmark_format": LANDMARK_NAMES,
        "extractor": "mediapipe",
        "frames": frames_payload,
    }


def discover_direct_clips(source_root: Path) -> list[tuple[Path, str, str]]:
    """
    Return (clip_path, video_filename, segment_name) for direct MP4 gesture clips.
    """
    clip_entries: list[tuple[Path, str, str]] = []

    for clip_path in sorted(source_root.iterdir(), key=lambda path: path.name.lower()):
        if not clip_path.is_file():
            continue
        if clip_path.suffix.lower() not in DIRECT_CLIP_VIDEO_EXTS:
            continue

        match = DIRECT_CLIP_FILENAME_RE.match(clip_path.name)
        if match is not None:
            video_filename = f"{match.group('video_key')}.mp4"
            segment_name = f"gest_{match.group('gest_idx')}"
        else:
            video_filename = f"{clip_path.stem.split('_', 1)[0]}.mp4"
            segment_name = clip_path.stem

        clip_entries.append((clip_path, video_filename, segment_name))

    return clip_entries


def load_direct_clip_frames(clip_path: Path) -> tuple[np.ndarray, list[Path]]:
    """Load all RGB frames from a pre-cut gesture clip MP4."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required to read gesture clip MP4 files. "
            "Install with: pip install opencv-python"
        ) from exc

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    frames: list[np.ndarray] = []
    frame_paths: list[Path] = []
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_paths.append(
                clip_path.with_name(f"{clip_path.stem}_frame_{frame_index:06d}.png")
            )
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames read from clip: {clip_path}")

    return np.stack(frames, axis=0), frame_paths


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


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _annotation_line_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _parse_marker_gesture_annotation_ids(
    eaf_path: Path,
) -> dict[tuple[int, int], str]:
    """Map ``(start_ms, end_ms)`` to EAF ``ANNOTATION_ID`` for marker GESTURE tiers."""
    text = _read_text_with_fallback(eaf_path)
    root = ET.fromstring(text)

    time_slots: dict[str, int] = {}
    for time_slot in root.findall(".//TIME_SLOT"):
        slot_id = time_slot.attrib.get("TIME_SLOT_ID")
        time_value = time_slot.attrib.get("TIME_VALUE")
        if slot_id is None or time_value is None:
            continue
        time_slots[slot_id] = int(time_value)

    annotation_ids: dict[tuple[int, int], str] = {}
    for tier in root.findall(".//TIER"):
        if tier.attrib.get("TIER_ID") != "marker":
            continue

        for alignable in tier.findall(".//ALIGNABLE_ANNOTATION"):
            annotation_id = alignable.attrib.get("ANNOTATION_ID")
            ref1 = alignable.attrib.get("TIME_SLOT_REF1")
            ref2 = alignable.attrib.get("TIME_SLOT_REF2")
            if ref1 is None or ref2 is None:
                continue
            if ref1 not in time_slots or ref2 not in time_slots:
                continue

            value_elem = alignable.find("ANNOTATION_VALUE")
            if value_elem is None or value_elem.text is None:
                continue
            if value_elem.text.strip() != "GESTURE":
                continue

            start_ms = time_slots[ref1]
            end_ms = time_slots[ref2]
            if annotation_id is not None:
                annotation_ids[(start_ms, end_ms)] = annotation_id

    return annotation_ids


def _lookup_gesture_annotation_id(
    annotation: dict[str, Any],
    *,
    eaf_cache: dict[str, dict[tuple[int, int], str]],
) -> str | None:
    """Resolve the EAF ``ANNOTATION_ID`` for one gesture segment."""
    annotation_path = annotation.get("annotation_path")
    start_ms = annotation.get("start_ms")
    end_ms = annotation.get("end_ms")
    if not annotation_path or start_ms is None or end_ms is None:
        return None

    eaf_path = Path(str(annotation_path))
    if not eaf_path.is_file():
        return None

    cache_key = str(eaf_path.resolve())
    if cache_key not in eaf_cache:
        try:
            eaf_cache[cache_key] = _parse_marker_gesture_annotation_ids(eaf_path)
        except ET.ParseError:
            eaf_cache[cache_key] = {}

    return eaf_cache[cache_key].get((int(start_ms), int(end_ms)))


def _annotation_field_order(annotation: dict[str, Any]) -> tuple[str, ...]:
    if annotation.get("marker") == "negative" or "negative_video_index" in annotation:
        return NEGATIVE_ANNOTATION_FIELD_ORDER
    return POSITIVE_ANNOTATION_FIELD_ORDER


def write_annotation_metadata(annotation: dict[str, Any], output_dir: Path) -> None:
    """Write gesture or negative segment metadata to ``annotation.txt``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_tier_values = annotation.get("raw_tier_values", {})
    if not isinstance(raw_tier_values, dict):
        raw_tier_values = {}

    ordered_fields = _annotation_field_order(annotation)
    lines: list[str] = []
    for field in ordered_fields:
        if field == "raw_tier_values":
            if "raw_tier_values" not in annotation:
                continue
            lines.append(
                f"raw_tier_values: {json.dumps(raw_tier_values, ensure_ascii=True)}"
            )
            continue
        if field not in annotation:
            continue
        lines.append(f"{field}: {_annotation_line_value(annotation[field])}")

    written_fields = set(ordered_fields)
    for field in sorted(annotation):
        if field in written_fields:
            continue
        lines.append(f"{field}: {_annotation_line_value(annotation[field])}")

    (output_dir / "annotation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_segment_annotation(
    *,
    source_segment_dir: Path,
    dest_segment_dir: Path,
    overwrite: bool,
    verbose: bool,
    eaf_cache: dict[str, dict[tuple[int, int], str]],
) -> str:
    """
    Copy and enrich one segment's ``annotation.txt`` with EAF ``annotation_id``.

    Returns:
        One of: "processed", "skipped", "failed_no_annotation"
    """
    source_annotation_path = source_segment_dir / "annotation.txt"
    dest_annotation_path = dest_segment_dir / "annotation.txt"

    if not source_annotation_path.is_file():
        print(
            f"Warning: missing annotation.txt in source segment: {source_segment_dir}"
        )
        return "failed_no_annotation"

    if dest_annotation_path.is_file() and not overwrite:
        existing = parse_segment_annotation_file(dest_annotation_path)
        if existing.get("annotation_id"):
            if verbose:
                print(f"Skipping existing annotation: {dest_annotation_path}")
            return "skipped"

    annotation = parse_segment_annotation_file(source_annotation_path)
    annotation["annotation_id"] = _lookup_gesture_annotation_id(
        annotation,
        eaf_cache=eaf_cache,
    )

    dest_segment_dir.mkdir(parents=True, exist_ok=True)
    write_annotation_metadata(annotation, dest_segment_dir)

    if verbose:
        print(
            f"Saved annotation: {dest_annotation_path} "
            f"(annotation_id={annotation.get('annotation_id')!r})"
        )

    return "processed"


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
    model_complexity: int,
    smooth_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    overwrite: bool,
    verbose: bool,
    eaf_cache: dict[str, dict[tuple[int, int], str]],
) -> str:
    """
    Process one gesture segment.

    Returns:
        One of: "processed", "skipped", "failed_no_frames", "failed_no_annotation"
    """
    if ANNOTATION_ONLY_MODE:
        return process_segment_annotation(
            source_segment_dir=source_segment_dir,
            dest_segment_dir=dest_segment_dir,
            overwrite=overwrite,
            verbose=verbose,
            eaf_cache=eaf_cache,
        )

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
    landmarks = extract_pose_landmarks(
        frames,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    assert isinstance(landmarks, np.ndarray)

    payload = build_landmarks_payload(
        video_filename=video_filename,
        segment_name=segment_name,
        source_segment_dir=source_segment_dir,
        frame_paths=frame_paths,
        landmarks=landmarks,
        num_joints=NUM_POSE_LANDMARKS,
    )

    dest_segment_dir.mkdir(parents=True, exist_ok=True)
    with dest_landmarks_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(payload), output_file, indent=2)
        output_file.write("\n")

    process_segment_annotation(
        source_segment_dir=source_segment_dir,
        dest_segment_dir=dest_segment_dir,
        overwrite=True,
        verbose=verbose,
        eaf_cache=eaf_cache,
    )

    if verbose:
        print(f"Saved landmarks: {dest_landmarks_path}")

    return "processed"


def process_direct_clip(
    *,
    source_clip_path: Path,
    dest_segment_dir: Path,
    video_filename: str,
    segment_name: str,
    model_complexity: int,
    smooth_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    overwrite: bool,
    verbose: bool,
) -> str:
    """
    Process one pre-cut gesture clip MP4.

    Returns:
        One of: "processed", "skipped", "failed_no_frames"
    """
    if ANNOTATION_ONLY_MODE:
        print(
            "Warning: annotation-only mode is not supported for direct clip parsing."
        )
        return "failed_no_frames"

    dest_landmarks_path = dest_segment_dir / "landmarks.json"

    if dest_landmarks_path.exists() and not overwrite:
        if verbose:
            print(f"Skipping existing landmarks: {dest_landmarks_path}")
        return "skipped"

    try:
        frames, frame_paths = load_direct_clip_frames(source_clip_path)
    except (RuntimeError, ValueError) as exc:
        print(f"Warning: {exc}")
        return "failed_no_frames"

    landmarks = extract_pose_landmarks(
        frames,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    assert isinstance(landmarks, np.ndarray)

    payload = build_landmarks_payload(
        video_filename=video_filename,
        segment_name=segment_name,
        source_segment_dir=source_clip_path,
        frame_paths=frame_paths,
        landmarks=landmarks,
        num_joints=NUM_POSE_LANDMARKS,
    )

    dest_segment_dir.mkdir(parents=True, exist_ok=True)
    with dest_landmarks_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_safe(payload), output_file, indent=2)
        output_file.write("\n")

    if verbose:
        print(f"Saved landmarks: {dest_landmarks_path}")

    return "processed"


def run_batch(
    *,
    source_root: Path,
    dest_root: Path,
    model_complexity: int,
    smooth_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    overwrite: bool,
    max_segments: int | None,
    verbose: bool,
    direct_clip_parse: bool = False,
) -> None:
    source_root = source_root.expanduser().resolve()
    dest_root = dest_root.expanduser().resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)

    if direct_clip_parse:
        segment_entries = discover_direct_clips(source_root)
    else:
        segment_entries = discover_segment_dirs(source_root)
    if max_segments is not None:
        segment_entries = segment_entries[:max_segments]

    video_dirs = sorted(
        {video_filename for _, video_filename, _ in segment_entries},
        key=str.lower,
    )

    print(f"Source root: {source_root}")
    print(f"Destination root: {dest_root}")
    if direct_clip_parse:
        print("Mode: direct gesture clip MP4 parsing")
    if ANNOTATION_ONLY_MODE:
        print("Mode: annotation-only (landmark extraction temporarily disabled)")
    else:
        print(f"MediaPipe model complexity: {model_complexity}")
        print(f"Landmarks per frame: {NUM_POSE_LANDMARKS} body (MediaPipe Pose)")
    print(f"Video directories found: {len(video_dirs)}")
    if direct_clip_parse:
        print(f"Gesture clip files found: {len(segment_entries)}")
    else:
        print(f"Segment directories found: {len(segment_entries)}")
    print()

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    eaf_cache: dict[str, dict[tuple[int, int], str]] = {}

    progress_desc = (
        "Writing segment annotations"
        if ANNOTATION_ONLY_MODE
        else "Storing MediaPipe landmarks"
    )
    iterator = segment_entries
    if tqdm is not None:
        iterator = tqdm(segment_entries, desc=progress_desc, unit="segment")

    for index, (source_path, video_filename, segment_name) in enumerate(
        iterator, start=1
    ):
        dest_segment_dir = dest_root / video_filename / segment_name
        if verbose or tqdm is None:
            print(
                f"Processing segment {index}/{len(segment_entries)}: "
                f"{source_path}"
            )

        try:
            if direct_clip_parse:
                status = process_direct_clip(
                    source_clip_path=source_path,
                    dest_segment_dir=dest_segment_dir,
                    video_filename=video_filename,
                    segment_name=segment_name,
                    model_complexity=model_complexity,
                    smooth_landmarks=smooth_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    overwrite=overwrite,
                    verbose=verbose,
                )
            else:
                status = process_segment(
                    source_segment_dir=source_path,
                    dest_segment_dir=dest_segment_dir,
                    video_filename=video_filename,
                    segment_name=segment_name,
                    model_complexity=model_complexity,
                    smooth_landmarks=smooth_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    overwrite=overwrite,
                    verbose=verbose,
                    eaf_cache=eaf_cache,
                )
        except Exception as exc:
            failed_count += 1
            print(f"Failed segment: {source_path}")
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

    source_root = args.source_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    if args.direct_clip_parse:
        if source_root == Path(DEFAULT_NEG_SOURCE_ROOT).expanduser().resolve():
            source_root = Path(DEFAULT_SOURCE_ROOT).expanduser().resolve()
        if dest_root == Path(DEFAULT_NEG_DEST_ROOT).expanduser().resolve():
            dest_root = Path(DEFAULT_DEST_ROOT).expanduser().resolve()

    run_batch(
        source_root=source_root,
        dest_root=dest_root,
        model_complexity=args.model_complexity,
        smooth_landmarks=args.smooth_landmarks,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        overwrite=args.overwrite,
        max_segments=args.max_segments,
        verbose=args.verbose,
        direct_clip_parse=args.direct_clip_parse,
    )


if __name__ == "__main__":
    main()
