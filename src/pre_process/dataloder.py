from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import random
import re
import warnings
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from read_transcript import get_transcript_lines
except ModuleNotFoundError:  # pragma: no cover - import path fallback
    import sys

    _DATASET_PROCESSING_DIR = Path(__file__).resolve().parent.parent / "dataset_processing"
    if str(_DATASET_PROCESSING_DIR) not in sys.path:
        sys.path.insert(0, str(_DATASET_PROCESSING_DIR))
    from read_transcript import get_transcript_lines


GESTURE_ATTRIBUTE_TIERS = (
    "task",
    "type",
    "function",
    "LEF@PAR",
    "HSL@PAR",
    "RIG@PAR",
    "HSR@PAR",
)


@dataclass(frozen=True)
class VideoKey:
    """
    Example:
        65-1.mp4 -> subject_id="65", day="1"
        68-2.mp4 -> subject_id="68", day="2"
    """
    subject_id: str
    day: str


@dataclass(frozen=True)
class AnnotationKey:
    """
    Example:
        rc65_day1_AB.eaf                  -> subject_id="65", day="1"
        rc65.day1.KU.eaf                  -> subject_id="65", day="1"
        nbd65.day1.gesturecoded.KU.eaf    -> subject_id="65", day="1"
        rc65-1.gesturecoded.eaf           -> subject_id="65", day="1"
        54-1.gesturecoded.eaf             -> subject_id="54", day="1"
    """
    subject_id: str
    day: str


@dataclass(frozen=True)
class VideoAnnotationPair:
    video_path: Path
    annotation_path: Path
    video_key: VideoKey
    annotation_key: AnnotationKey


@dataclass(frozen=True)
class GestureAnnotation:
    video_path: Path
    annotation_path: Path
    start_ms: int
    end_ms: int
    duration_ms: int
    marker: str | None
    task: str | None
    gesture_type: str | None
    function: str | None
    left_hand_label: str | None
    left_hand_shape: str | None
    right_hand_label: str | None
    right_hand_shape: str | None
    raw_tier_values: dict[str, str | None] = field(default_factory=dict)


CHA_MEDIA_HEADER_RE = re.compile(
    r"^@Media:\s*(?P<media>[^,\r\n]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GestureSegmentSample:
    segment_dir: Path
    landmarks_path: Path
    video_filename: str
    gesture_index: int
    label: int
    annotation: dict
    transcript_path: Path | None = None

    @property
    def label_name(self) -> str:
        """Human-readable class name for POS/NEG segments."""
        return LABEL_NAME_BY_VALUE[self.label]

    @property
    def gesture_type(self) -> str:
        """Training label alias kept for backward compatibility."""
        return self.label_name


GESTURE_SEGMENT_DIR_RE = re.compile(r"^gest_(\d+)$")
NEGATIVE_SEGMENT_DIR_RE = re.compile(r"^neg_(\d+)$")

POSITIVE_SEGMENT_LABEL = 1
NEGATIVE_SEGMENT_LABEL = 0
POSITIVE_SEGMENT_LABEL_NAME = "POS"
NEGATIVE_SEGMENT_LABEL_NAME = "NEG"
LABEL_NAME_BY_VALUE = {
    POSITIVE_SEGMENT_LABEL: POSITIVE_SEGMENT_LABEL_NAME,
    NEGATIVE_SEGMENT_LABEL: NEGATIVE_SEGMENT_LABEL_NAME,
}

MEDIAPIPE_POSE_LANDMARK_COUNT = 33
OPENPOSE_BODY25_LANDMARK_COUNT = 25
OPENPOSE_HAND21_LANDMARK_COUNT = 21

# MediaPipe Pose face landmarks (indices 0-10 in the 33-joint pose model).
MEDIAPIPE_FACE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
)

# Shared model landmark order for the upper-body subset:
#   0 left_shoulder,  1 right_shoulder,
#   2 left_elbow,     3 right_elbow,
#   4 left_wrist,     5 right_wrist,
#   6 left_pinky,     7 right_pinky,
#   8 left_index,     9 right_index,
#   10 left_thumb,    11 right_thumb,
#   12 left_hip,      13 right_hip
UPPER_BODY_LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
)

MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_NAMES = (
    MEDIAPIPE_FACE_LANDMARK_NAMES + UPPER_BODY_LANDMARK_NAMES
)

# MediaPipe Pose face + upper-body subset: pose indices 0-24.
MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_INDICES = tuple(
    range(len(MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_NAMES))
)

# MediaPipe Pose upper-body-only subset (legacy 25-joint layout).
MEDIAPIPE_UPPER_BODY_LANDMARK_INDICES = (
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    17,  # left_pinky
    18,  # right_pinky
    19,  # left_index
    20,  # right_index
    21,  # left_thumb
    22,  # right_thumb
    23,  # left_hip
    24,  # right_hip
)

# OpenPose BODY_25 + optional HAND_21 subset in the same shared order.
# Each entry is (source, joint_index).
OPENPOSE_UPPER_BODY_JOINT_SOURCES: tuple[tuple[str, int], ...] = (
    ("body", 5),   # LShoulder
    ("body", 2),   # RShoulder
    ("body", 6),   # LElbow
    ("body", 3),   # RElbow
    ("body", 7),   # LWrist
    ("body", 4),   # RWrist
    ("hand_left", 20),   # LPinky tip
    ("hand_right", 20),  # RPinky tip
    ("hand_left", 8),    # LIndex tip
    ("hand_right", 8),   # RIndex tip
    ("hand_left", 4),    # LThumb tip
    ("hand_right", 4),   # RThumb tip
    ("body", 12),  # LHip
    ("body", 9),   # RHip
)

NUM_UPPER_BODY_LANDMARKS = len(OPENPOSE_UPPER_BODY_JOINT_SOURCES)
NUM_MEDIAPIPE_FACE_UPPER_BODY_LANDMARKS = len(
    MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_INDICES
)

# Shoulder indices within the OpenPose upper-body subset.
UPPER_BODY_LEFT_SHOULDER_INDEX = 0
UPPER_BODY_RIGHT_SHOULDER_INDEX = 1

# Shoulder indices within the MediaPipe face+upper-body subset.
MEDIAPIPE_FACE_UPPER_BODY_LEFT_SHOULDER_INDEX = 11
MEDIAPIPE_FACE_UPPER_BODY_RIGHT_SHOULDER_INDEX = 12

OPENPOSE_LEFT_WRIST_OUTPUT_INDEX = 4
OPENPOSE_RIGHT_WRIST_OUTPUT_INDEX = 5
OPENPOSE_WRIST_FALLBACKS: tuple[tuple[str, int], ...] = (
    ("hand_left", 0),
    ("hand_right", 0),
)

_INT_ANNOTATION_FIELDS = frozenset(
    {
        "start_ms",
        "end_ms",
        "duration_ms",
        "gesture_video_index",
        "negative_video_index",
    }
)


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _pair_key(subject_id: str, day: str) -> tuple[str, str]:
    return subject_id.lower(), day.lower()


def _parse_video_filename_key(video_filename: str) -> VideoKey | None:
    """Parse gesture segment video directory names such as ``65-1.mp4``."""
    return AphasiaDataset._parse_video_key(Path(video_filename))


def _cha_logical_stem(cha_path: Path) -> str:
    """Return the filename stem used for CHA subject/day parsing."""
    name = cha_path.name
    lower_name = name.lower()
    if lower_name.endswith(".elan.cha"):
        return name[: -len(".elan.cha")]
    if lower_name.endswith(".cha"):
        return cha_path.stem
    return cha_path.stem


def _parse_cha_media_reference(cha_path: Path) -> str | None:
    """Read ``@Media:`` from a CHAT transcript header."""
    for line in _read_text_with_fallback(cha_path).splitlines():
        match = CHA_MEDIA_HEADER_RE.match(line.strip())
        if match is not None:
            media_ref = match.group("media").strip()
            return media_ref or None
    return None


def _parse_cha_filename_key(cha_path: Path) -> VideoKey | None:
    """Parse subject/day from CHAT filenames such as ``rc65_day1_AB.elan.cha``."""
    annotation_key = AphasiaDataset._parse_annotation_key_from_filename(
        Path(_cha_logical_stem(cha_path) + ".eaf")
    )
    if annotation_key is None:
        return None
    return VideoKey(
        subject_id=annotation_key.subject_id,
        day=annotation_key.day,
    )


def _scan_cha_transcript_files(transcript_dir: Path) -> list[Path]:
    """Recursively collect ``.cha`` files, including compound ``*.elan.cha`` names."""
    return sorted(
        path
        for path in transcript_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith(".cha")
    )


def _register_transcript_candidate(
    candidates: dict[str, list[Path]],
    key: str,
    cha_path: Path,
) -> None:
    normalized_key = key.strip().lower()
    if not normalized_key:
        return
    bucket = candidates.setdefault(normalized_key, [])
    if cha_path not in bucket:
        bucket.append(cha_path)


def _choose_transcript_candidate(
    candidates: list[Path],
    *,
    video_filename: str,
    match_kind: str,
    strict_transcripts: bool,
    warned_keys: set[str],
) -> Path | None:
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    sorted_candidates = sorted(candidates, key=lambda path: str(path).lower())
    warning_key = f"{match_kind}:{video_filename.lower()}"
    candidate_text = "\n  ".join(str(path) for path in sorted_candidates)
    message = (
        f"Multiple CHAT transcripts matched video {video_filename!r} via "
        f"{match_kind}:\n  {candidate_text}"
    )

    if strict_transcripts:
        raise ValueError(message)

    if warning_key not in warned_keys:
        warnings.warn(message, stacklevel=3)
        warned_keys.add(warning_key)

    return sorted_candidates[0]


def _build_transcript_lookup(
    cha_paths: Sequence[Path],
    *,
    strict_transcripts: bool,
    warned_keys: set[str],
) -> tuple[dict[str, list[Path]], dict[tuple[str, str], list[Path]]]:
    """
    Build transcript lookup tables keyed by ``@Media`` reference and by
    normalized ``(subject_id, day)``.
    """
    by_media_ref: dict[str, list[Path]] = {}
    by_pair_key: dict[tuple[str, str], list[Path]] = {}

    for cha_path in cha_paths:
        media_ref = _parse_cha_media_reference(cha_path)
        if media_ref is not None:
            _register_transcript_candidate(by_media_ref, media_ref, cha_path)

        filename_key = _parse_cha_filename_key(cha_path)
        if filename_key is not None:
            pair = _pair_key(filename_key.subject_id, filename_key.day)
            bucket = by_pair_key.setdefault(pair, [])
            if cha_path not in bucket:
                bucket.append(cha_path)

    for bucket in by_media_ref.values():
        bucket.sort(key=lambda path: str(path).lower())
    for bucket in by_pair_key.values():
        bucket.sort(key=lambda path: str(path).lower())

    duplicate_media: list[tuple[str, list[Path]]] = []
    duplicate_pair: list[tuple[tuple[str, str], list[Path]]] = []

    for media_ref, candidates in by_media_ref.items():
        if len(candidates) > 1:
            duplicate_media.append((media_ref, candidates))

    for pair, candidates in by_pair_key.items():
        if len(candidates) > 1:
            duplicate_pair.append((pair, candidates))

    if strict_transcripts and (duplicate_media or duplicate_pair):
        details: list[str] = []
        for media_ref, candidates in duplicate_media:
            candidate_text = "\n  ".join(str(path) for path in candidates)
            details.append(
                f"@Media reference {media_ref!r}:\n  {candidate_text}"
            )
        for pair, candidates in duplicate_pair:
            candidate_text = "\n  ".join(str(path) for path in candidates)
            details.append(
                f"subject/day {pair!r}:\n  {candidate_text}"
            )
        raise ValueError(
            "Ambiguous CHAT transcript indexing detected during dataset "
            "initialization:\n" + "\n\n".join(details)
        )

    for media_ref, candidates in duplicate_media:
        warning_key = f"index:media:{media_ref}"
        if warning_key not in warned_keys:
            candidate_text = "\n  ".join(str(path) for path in candidates)
            warnings.warn(
                "Multiple CHAT transcripts share the same @Media reference "
                f"{media_ref!r}; deterministic resolution will use the first "
                f"sorted path unless strict_transcripts=True:\n  {candidate_text}",
                stacklevel=2,
            )
            warned_keys.add(warning_key)

    for pair, candidates in duplicate_pair:
        warning_key = f"index:pair:{pair[0]}:{pair[1]}"
        if warning_key not in warned_keys:
            candidate_text = "\n  ".join(str(path) for path in candidates)
            warnings.warn(
                "Multiple CHAT transcripts share the same subject/day key "
                f"{pair!r}; deterministic resolution will use the first sorted "
                f"path unless strict_transcripts=True:\n  {candidate_text}",
                stacklevel=2,
            )
            warned_keys.add(warning_key)

    return by_media_ref, by_pair_key


def _normalize_transcript_kwargs(
    transcript_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    return dict(transcript_kwargs) if transcript_kwargs is not None else {}


def _transcript_cache_key(
    transcript_path: Path,
    start_ms: int,
    end_ms: int,
    transcript_kwargs: dict[str, Any] | None,
    *,
    transcript_offset_seconds: float,
    num_transcript_chunks: int,
) -> tuple[str, int, int, tuple[tuple[str, Any], ...], float, int]:
    normalized_kwargs = _normalize_transcript_kwargs(transcript_kwargs)
    frozen_kwargs = tuple(sorted(normalized_kwargs.items()))
    return (
        str(transcript_path),
        start_ms,
        end_ms,
        frozen_kwargs,
        float(transcript_offset_seconds),
        int(num_transcript_chunks),
    )


def _format_transcript_lines(transcript_lines: Sequence[dict[str, Any]]) -> str:
    formatted_lines = [
        f"{line['speaker']}: {line['text']}"
        for line in transcript_lines
        if str(line.get("text", "")).strip()
    ]
    return "\n".join(formatted_lines)


def _utterance_overlaps_interval(
    utterance: dict[str, Any],
    interval_start_ms: float,
    interval_end_ms: float,
) -> bool:
    start_ts = int(utterance["start_ts"])
    end_ts = int(utterance["end_ts"])
    return start_ts < interval_end_ms and end_ts > interval_start_ms


def _build_transcript_chunks(
    transcript_lines: Sequence[dict[str, Any]],
    *,
    window_start_ms: int,
    window_end_ms: int,
    num_chunks: int,
) -> tuple[list[str], torch.Tensor]:
    """Split ``[window_start_ms, window_end_ms]`` into ``num_chunks`` equal intervals."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")

    window_start_ms = max(0, int(window_start_ms))
    window_end_ms = int(window_end_ms)
    window_duration_ms = window_end_ms - window_start_ms
    if window_duration_ms <= 0:
        empty_chunks = [""] * num_chunks
        return empty_chunks, torch.zeros(num_chunks, dtype=torch.bool)

    chunk_duration_ms = window_duration_ms / num_chunks
    chunks: list[str] = []
    mask_values: list[bool] = []

    for chunk_index in range(num_chunks):
        chunk_start_ms = window_start_ms + chunk_index * chunk_duration_ms
        chunk_end_ms = window_start_ms + (chunk_index + 1) * chunk_duration_ms
        overlapping_lines = [
            line
            for line in transcript_lines
            if _utterance_overlaps_interval(line, chunk_start_ms, chunk_end_ms)
        ]
        chunk_text = _format_transcript_lines(overlapping_lines)
        chunks.append(chunk_text)
        mask_values.append(bool(chunk_text.strip()))

    return chunks, torch.tensor(mask_values, dtype=torch.bool)


CLAN_MEDIA_TIMESTAMP_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<milliseconds>\d+)$"
)
UTTERANCE_ALIGNMENT_MARKER_RE = re.compile(r"<+\+?utt_[se]>", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

GESTURE_TEXT_ALIGNMENT_FIELDS = (
    "aligned_utt_start_ts",
    "aligned_utt_end_ts",
    "aligned_utterance",
    "full_transcript",
)

DEBUG_GETITEM_VIDEO_SUBJECT_IDS = frozenset(
    {
        "65",
        "81",
        "92",
        "101",
        "104",
        "105",
        "122",
        "123",
        "128",
        "133",
        "171",
        "202",
        "208",
    }
)


def _parse_media_timestamp_to_ms(timestamp: str) -> int:
    """Parse ``HH:MM:SS.mmm`` media timestamps used in alignment annotations."""
    match = CLAN_MEDIA_TIMESTAMP_RE.match(str(timestamp).strip())
    if match is None:
        raise ValueError(f"Invalid media timestamp: {timestamp!r}")

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + milliseconds


def _empty_gesture_text_alignment() -> dict[str, None]:
    return {field: None for field in GESTURE_TEXT_ALIGNMENT_FIELDS}


def _clean_alignment_transcript_markers(text: str) -> str:
    """Remove CLAN utterance boundary markers from alignment transcript text."""
    return UTTERANCE_ALIGNMENT_MARKER_RE.sub("", str(text))


def _normalize_alignment_match_text(text: str) -> str:
    """Normalize transcript text for substring-based alignment matching."""
    cleaned = _clean_alignment_transcript_markers(text)
    normalized_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ": " in line:
            _, _, content = line.partition(": ")
            normalized_lines.append(content.strip())
        else:
            normalized_lines.append(line)
    joined = " ".join(normalized_lines)
    return WHITESPACE_RE.sub(" ", joined.strip().lower())


def _normalize_transcript_chunk_for_match(chunk: str) -> str:
    return _normalize_alignment_match_text(str(chunk))


def _find_best_aligned_transcript_chunk_index(
    transcript_chunks: Sequence[str],
    *,
    full_transcript: str,
    aligned_utterance: str,
) -> int | None:
    """
    Find the longest transcript chunk whose normalized text is contained in
    ``full_transcript`` and also contains ``aligned_utterance``.
    """
    normalized_full = _normalize_alignment_match_text(full_transcript)
    normalized_utterance = _normalize_alignment_match_text(aligned_utterance)
    if not normalized_full or not normalized_utterance:
        return None

    best_index: int | None = None
    best_match_length = -1

    for chunk_index, chunk in enumerate(transcript_chunks):
        normalized_chunk = _normalize_transcript_chunk_for_match(chunk)
        if not normalized_chunk:
            continue
        if normalized_chunk not in normalized_full:
            continue
        if normalized_utterance not in normalized_chunk:
            continue
        if len(normalized_chunk) > best_match_length:
            best_match_length = len(normalized_chunk)
            best_index = chunk_index

    return best_index


def _build_transcript_alignment_mask(
    transcript_chunks: Sequence[str],
    *,
    full_transcript: str | None,
    aligned_utterance: str | None,
) -> torch.Tensor:
    """Return a one-hot chunk mask marking the best alignment match, else zeros."""
    num_chunks = len(transcript_chunks)
    alignment_mask = torch.zeros(num_chunks, dtype=torch.float32)
    if not full_transcript or not aligned_utterance:
        return alignment_mask

    match_index = _find_best_aligned_transcript_chunk_index(
        transcript_chunks,
        full_transcript=full_transcript,
        aligned_utterance=aligned_utterance,
    )
    if match_index is not None:
        alignment_mask[match_index] = 1.0
    return alignment_mask


def _extract_gesture_text_alignment_fields(
    alignment_entry: dict[str, Any],
) -> dict[str, str]:
    fields = {
        field: str(alignment_entry[field])
        for field in GESTURE_TEXT_ALIGNMENT_FIELDS
    }
    if fields["full_transcript"] is not None:
        fields["full_transcript"] = _clean_alignment_transcript_markers(
            fields["full_transcript"]
        )
    return fields


def _find_gesture_text_alignment(
    alignment_entries: Sequence[dict[str, Any]],
    *,
    annotation_id: str | None,
) -> dict[str, str] | None:
    """Return alignment text fields when ``annotation_id`` matches an entry."""
    if annotation_id is None:
        return None

    normalized_id = str(annotation_id).strip()
    if not normalized_id:
        return None

    matching_entries = [
        entry
        for entry in alignment_entries
        if str(entry.get("annotation_id", "")).strip() == normalized_id
    ]
    if not matching_entries:
        return None

    if len(matching_entries) > 1:
        warnings.warn(
            f"Multiple alignment entries share annotation_id {normalized_id!r}; "
            "using the first match.",
            stacklevel=2,
        )

    return _extract_gesture_text_alignment_fields(matching_entries[0])


def _resolve_alignment_anns_path(
    alignment_anns_dir: Path,
    video_filename: str,
) -> Path | None:
    video_stem = Path(video_filename).stem
    candidate = alignment_anns_dir / f"{video_stem}_anns.json"
    if candidate.is_file():
        return candidate
    return None


def _scan_alignment_anns_dir(alignment_anns_dir: Path) -> dict[str, Path]:
    """Map ``video_filename`` values to alignment JSON paths."""
    if not alignment_anns_dir.is_dir():
        warnings.warn(
            f"Alignment annotation directory does not exist: {alignment_anns_dir}. "
            "Gesture-text alignment fields will be empty.",
            stacklevel=2,
        )
        return {}

    index: dict[str, Path] = {}
    for path in sorted(alignment_anns_dir.glob("*_anns.json")):
        if not path.is_file():
            continue
        video_stem = path.name[: -len("_anns.json")]
        video_filename = f"{video_stem}.mp4"
        index[video_filename] = path
    return index


def _load_alignment_anns_payload(
    alignment_path: Path,
    *,
    expected_video_filename: str,
) -> list[dict[str, Any]]:
    with alignment_path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    payload_video_filename = str(payload.get("video_filename", "")).strip()
    if payload_video_filename != expected_video_filename:
        raise ValueError(
            "Alignment annotation video_filename mismatch for "
            f"{alignment_path.name}: expected {expected_video_filename!r}, "
            f"got {payload_video_filename!r}"
        )

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(
            f"Alignment annotation file {alignment_path} is missing an "
            "'annotations' list."
        )
    return annotations


def parse_segment_annotation_file(annotation_path: Path) -> dict:
    """Parse a gesture-segment ``annotation.txt`` file written by extraction."""
    metadata: dict = {}

    for line in _read_text_with_fallback(annotation_path).splitlines():
        line = line.strip()
        if not line or ": " not in line:
            continue

        key, value = line.split(": ", 1)
        if key == "raw_tier_values":
            metadata[key] = json.loads(value) if value else {}
        elif key in _INT_ANNOTATION_FIELDS:
            metadata[key] = int(value) if value else None
        else:
            metadata[key] = value

    return metadata


def list_segment_frame_paths(segment_dir: Path) -> list[Path]:
    """Return sorted PNG frame paths named with numeric indices (e.g. 0000.png)."""
    return sorted(
        (
            path
            for path in segment_dir.glob("*.png")
            if path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )


def _select_temporal_indices(
    num_source_frames: int,
    num_frames: int | None,
) -> list[int]:
    """Select evenly spaced frame indices, matching segment frame subsampling."""
    if num_frames is None:
        return list(range(num_source_frames))

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    if num_source_frames == 0:
        raise ValueError("Cannot subsample landmarks from an empty sequence.")

    if num_frames == 1:
        return [0]

    last_index = num_source_frames - 1
    return [
        int(round(frame_idx * last_index / (num_frames - 1)))
        for frame_idx in range(num_frames)
    ]


@dataclass(frozen=True)
class PoseLandmarkSequence:
    """Pose landmarks loaded from a segment ``landmarks.json`` file."""

    body: np.ndarray
    extractor: str
    hand_left: np.ndarray | None = None
    hand_right: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.body.shape[0])

    @property
    def is_openpose(self) -> bool:
        if self.extractor.startswith("openpose"):
            return True
        return self.body.shape[1] == OPENPOSE_BODY25_LANDMARK_COUNT


def _parse_frame_joint_array(
    frame_entry: dict,
    key: str,
    num_joints: int,
) -> np.ndarray:
    """Parse one frame's landmark list into ``[num_joints, 4]`` with NaN gaps."""
    joints = np.full((num_joints, 4), np.nan, dtype=np.float32)
    frame_landmarks = frame_entry.get(key)
    if not frame_landmarks:
        return joints

    for joint_idx, joint_values in enumerate(frame_landmarks[:num_joints]):
        for channel_idx, value in enumerate(joint_values[:4]):
            if value is not None:
                joints[joint_idx, channel_idx] = float(value)

    return joints


def _infer_landmark_format(payload: dict, body_joint_count: int) -> str:
    extractor = str(payload.get("extractor", "")).lower()
    if extractor.startswith("openpose"):
        return "openpose"
    if body_joint_count == OPENPOSE_BODY25_LANDMARK_COUNT:
        return "openpose"
    if body_joint_count >= MEDIAPIPE_POSE_LANDMARK_COUNT:
        return "mediapipe"
    return "unknown"


def load_landmarks_from_json(landmarks_path: Path) -> PoseLandmarkSequence:
    """
    Load pose landmarks from a ``landmarks.json`` file.

    Supports MediaPipe (33 body joints) and OpenPose BODY_25 exports. When
    present, OpenPose ``hand_left`` / ``hand_right`` sequences are loaded too.

    Returns:
        ``PoseLandmarkSequence`` with body shape ``[T, J, 4]`` and optional
        hand arrays shaped ``[T, 21, 4]``.
    """
    with landmarks_path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    frames = payload.get("frames")
    if not frames:
        raise ValueError(f"No frame landmarks found in: {landmarks_path}")

    first_body = frames[0].get("landmarks") or []
    body_joint_count = len(first_body)
    if body_joint_count == 0:
        landmark_shape = payload.get("landmark_shape")
        if isinstance(landmark_shape, list) and len(landmark_shape) >= 2:
            body_joint_count = int(landmark_shape[1])
        else:
            body_joint_count = MEDIAPIPE_POSE_LANDMARK_COUNT

    extractor = str(payload.get("extractor", ""))
    landmark_format = _infer_landmark_format(payload, body_joint_count)
    if landmark_format == "openpose":
        body_joint_count = OPENPOSE_BODY25_LANDMARK_COUNT
    elif body_joint_count < MEDIAPIPE_POSE_LANDMARK_COUNT:
        body_joint_count = MEDIAPIPE_POSE_LANDMARK_COUNT

    has_hand_left = any("hand_left" in frame for frame in frames)
    has_hand_right = any("hand_right" in frame for frame in frames)

    body = np.full((len(frames), body_joint_count, 4), np.nan, dtype=np.float32)
    hand_left: np.ndarray | None = None
    hand_right: np.ndarray | None = None

    if has_hand_left:
        hand_left = np.full(
            (len(frames), OPENPOSE_HAND21_LANDMARK_COUNT, 4),
            np.nan,
            dtype=np.float32,
        )
    if has_hand_right:
        hand_right = np.full(
            (len(frames), OPENPOSE_HAND21_LANDMARK_COUNT, 4),
            np.nan,
            dtype=np.float32,
        )

    for frame_idx, frame_entry in enumerate(frames):
        body[frame_idx] = _parse_frame_joint_array(
            frame_entry,
            "landmarks",
            body_joint_count,
        )
        if hand_left is not None:
            hand_left[frame_idx] = _parse_frame_joint_array(
                frame_entry,
                "hand_left",
                OPENPOSE_HAND21_LANDMARK_COUNT,
            )
        if hand_right is not None:
            hand_right[frame_idx] = _parse_frame_joint_array(
                frame_entry,
                "hand_right",
                OPENPOSE_HAND21_LANDMARK_COUNT,
            )

    return PoseLandmarkSequence(
        body=body,
        extractor=extractor,
        hand_left=hand_left,
        hand_right=hand_right,
    )


def _joint_from_sources(
    sources: dict[str, np.ndarray],
    source_key: str,
    joint_idx: int,
    frame_idx: int,
) -> np.ndarray:
    source = sources.get(source_key)
    if source is None:
        return np.full(4, np.nan, dtype=np.float32)

    if joint_idx < 0 or joint_idx >= source.shape[1]:
        return np.full(4, np.nan, dtype=np.float32)

    return source[frame_idx, joint_idx]


def _is_valid_joint(joint: np.ndarray) -> bool:
    return not np.isnan(joint[3]) or (
        not np.isnan(joint[0]) and not np.isnan(joint[1])
    )


def select_openpose_upper_body_landmarks(
    sequence: PoseLandmarkSequence,
) -> np.ndarray:
    """
    Build the upper-body tensor from OpenPose BODY_25 plus optional HAND_21 data.

    Includes shoulders, elbows, wrists, pinky/index/thumb tips, and hips in the
    same 14-joint order used for MediaPipe. Missing body wrists fall back to the
    corresponding hand wrist (index 0) when hand data is present.
    """
    num_frames = sequence.num_frames
    upper_body = np.full(
        (num_frames, NUM_UPPER_BODY_LANDMARKS, 4),
        np.nan,
        dtype=np.float32,
    )
    sources = {
        "body": sequence.body,
        "hand_left": sequence.hand_left,
        "hand_right": sequence.hand_right,
    }

    for frame_idx in range(num_frames):
        for out_joint_idx, (source_key, joint_idx) in enumerate(
            OPENPOSE_UPPER_BODY_JOINT_SOURCES
        ):
            joint = _joint_from_sources(sources, source_key, joint_idx, frame_idx)

            if (
                out_joint_idx
                in (OPENPOSE_LEFT_WRIST_OUTPUT_INDEX, OPENPOSE_RIGHT_WRIST_OUTPUT_INDEX)
                and source_key == "body"
                and not _is_valid_joint(joint)
            ):
                fallback_key, fallback_idx = OPENPOSE_WRIST_FALLBACKS[
                    out_joint_idx - OPENPOSE_LEFT_WRIST_OUTPUT_INDEX
                ]
                fallback_joint = _joint_from_sources(
                    sources,
                    fallback_key,
                    fallback_idx,
                    frame_idx,
                )
                if _is_valid_joint(fallback_joint):
                    joint = fallback_joint

            upper_body[frame_idx, out_joint_idx] = joint

    return upper_body


def select_mediapipe_face_upper_body_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Select MediaPipe Pose face and upper-body landmarks.

    Returns pose indices 0-24: nose, eyes, ears, mouth, shoulders, arms,
    hands, and hips.
    """
    if landmarks.ndim != 3:
        raise ValueError(
            "Expected landmarks with shape [T, J, C], "
            f"got ndim={landmarks.ndim} with shape {tuple(landmarks.shape)}"
        )

    max_index = max(MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_INDICES)
    if landmarks.shape[1] <= max_index:
        raise ValueError(
            "Expected MediaPipe pose landmarks through right_hip index 24, "
            f"got J={landmarks.shape[1]}"
        )

    return landmarks[:, MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_INDICES, :]


def read_segment_landmarks(
    landmarks_path: Path,
    *,
    num_frames: int | None,
    return_tensor: bool = True,
) -> torch.Tensor | np.ndarray:
    """
    Load, temporally subsample, and select upper-body joints from one segment.

    Matches the landmark preprocessing used by ``AphasiaGestureSegmentDataset``.
    """
    sequence = load_landmarks_from_json(landmarks_path)
    selected_indices = _select_temporal_indices(sequence.num_frames, num_frames)
    sequence = PoseLandmarkSequence(
        body=sequence.body[selected_indices],
        extractor=sequence.extractor,
        hand_left=(
            sequence.hand_left[selected_indices]
            if sequence.hand_left is not None
            else None
        ),
        hand_right=(
            sequence.hand_right[selected_indices]
            if sequence.hand_right is not None
            else None
        ),
    )
    landmarks = select_upper_body_landmarks(sequence)

    if return_tensor:
        return torch.from_numpy(landmarks).float()

    return landmarks.astype(np.float32, copy=False)


def select_upper_body_landmarks(
    landmarks: np.ndarray | PoseLandmarkSequence,
    *,
    hand_left: np.ndarray | None = None,
    hand_right: np.ndarray | None = None,
) -> np.ndarray:
    """
    Keep pose landmarks for model input.

    For OpenPose BODY_25 JSON, pass a ``PoseLandmarkSequence`` or a body array.
    Optional HAND_21 data is used for pinky/index/thumb tips and as a body-wrist
    fallback. For MediaPipe pose arrays, selects face landmarks (indices 0-10)
    plus shoulders, elbows, wrists, pinkies, indexes, thumbs, and hips
    (indices 11-24).

    Returns:
        Array with shape [T, NUM_UPPER_BODY_LANDMARKS, C] for OpenPose or
        [T, NUM_MEDIAPIPE_FACE_UPPER_BODY_LANDMARKS, C] for MediaPipe.
    """
    if isinstance(landmarks, PoseLandmarkSequence):
        if landmarks.is_openpose:
            return select_openpose_upper_body_landmarks(landmarks)
        landmarks = landmarks.body

    if not isinstance(landmarks, np.ndarray):
        raise TypeError(
            "landmarks must be a numpy.ndarray or PoseLandmarkSequence, "
            f"got {type(landmarks)!r}"
        )

    if landmarks.ndim != 3:
        raise ValueError(
            "Expected landmarks with shape [T, J, C], "
            f"got ndim={landmarks.ndim} with shape {tuple(landmarks.shape)}"
        )

    if landmarks.shape[1] == OPENPOSE_BODY25_LANDMARK_COUNT:
        sequence = PoseLandmarkSequence(
            body=landmarks,
            extractor="openpose_body25",
            hand_left=hand_left,
            hand_right=hand_right,
        )
        return select_openpose_upper_body_landmarks(sequence)

    if landmarks.shape[1] < max(MEDIAPIPE_FACE_UPPER_BODY_LANDMARK_INDICES) + 1:
        raise ValueError(
            "Expected MediaPipe pose landmarks through right_hip index 24 or "
            f"{OPENPOSE_BODY25_LANDMARK_COUNT} for OpenPose selection, "
            f"got J={landmarks.shape[1]}"
        )

    return select_mediapipe_face_upper_body_landmarks(landmarks)
def _time_overlap_ms(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> int:
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0, overlap_end - overlap_start)


def _get_best_tier_value_for_interval(
    annotations: list[tuple[int, int, str]],
    start_ms: int,
    end_ms: int,
) -> str | None:
    for ann_start, ann_end, value in annotations:
        if ann_start == start_ms and ann_end == end_ms:
            return value

    best_value: str | None = None
    best_overlap = 0

    for ann_start, ann_end, value in annotations:
        overlap = _time_overlap_ms(start_ms, end_ms, ann_start, ann_end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_value = value

    return best_value if best_overlap > 0 else None


class AphasiaDataset(Dataset):
    """
    AphasiaBank videos + EAF gesture annotations.

    Matches video files to annotation files, then parses each matched EAF into
    per-gesture ``GestureAnnotation`` entries stored in ``self.samples``.

    Each dataset index corresponds to one gesture segment (a marker==GESTURE
    interval with aligned task/type/function/handshape tiers), not one full video.
    A single video may therefore contribute many ``__getitem__`` samples.

    Expected video names:
        65-1.mp4
        65-2.mp4
        68-1.mp4
        68-2.mp4

    Expected annotation names:
        rc65_day1_AB.eaf
        rc65.day1.KU.eaf
        nbd65.day1.gesturecoded.KU.eaf
        rc65-1.gesturecoded.eaf
        54-1.gesturecoded.eaf

    ``annotation_dir`` may be a single directory or a list/tuple of directories
    (e.g. separate ``aphasia`` and ``NBD`` annotation folders).

    Matching rule:
        video subject_id and day == annotation subject_id and day
        (optional rc/nbd prefix on annotation filenames is ignored)

    Example:
        65-1.mp4 -> rc65_day1_AB.eaf
        65-1.mp4 -> nbd65.day1.KU.eaf
        65-1.mp4 -> 65-1.gesturecoded.eaf
    """

    DEFAULT_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    DEFAULT_ANNOTATION_EXTS = {".eaf"}

    VIDEO_NAME_RE = re.compile(
        r"^(?P<subject_id>\d+)-(?P<day>\d+)$",
        re.IGNORECASE,
    )

    ANNOTATION_DAY_WORD_RE = re.compile(
        r"^(?P<prefix>rc|nbd)?(?P<subject_id>\d+)[._-]*day(?P<day>\d+)(?:[._-].*)?$",
        re.IGNORECASE,
    )
    ANNOTATION_DASH_DAY_RE = re.compile(
        r"^(?P<prefix>rc|nbd)?(?P<subject_id>\d+)-(?P<day>\d+)(?:[._-].*)?$",
        re.IGNORECASE,
    )
    ANNOTATION_PREFIX_DOT_DAY_WORD_RE = re.compile(
        r"^(?P<prefix>rc|nbd)\.(?P<subject_id>\d+)[._-]*day(?P<day>\d+)(?:[._-].*)?$",
        re.IGNORECASE,
    )
    ANNOTATION_SUBJECT_UNDERSCORE_SUFFIX_RE = re.compile(
        r"^(?P<prefix>rc|nbd)(?P<subject_id>\d+)_(?P<suffix>.+)$",
        re.IGNORECASE,
    )

    SUPPORTED_ANNOTATION_PATTERNS_MSG = (
        "Supported annotation patterns:\n"
        "   rc65_day1_AB.eaf\n"
        "   rc65.day1.KU.eaf\n"
        "   rc65.day1.gesturecoded.KU.eaf\n"
        "   rc.199.day1.coded.BB.eaf\n"
        "   nbd65_day1_AB.eaf\n"
        "   nbd65.day1.KU.eaf\n"
        "   nbd65.day1.gesturecoded.KU.eaf\n"
        "   rc65-1.gesturecoded.eaf\n"
        "   nbd65-1.gesturecoded.eaf\n"
        "   54-1.gesturecoded.eaf\n"
        "   rc110_gesturesoutofview.eaf (day from EAF media reference)"
    )

    def __init__(
        self,
        video_dir: str | Path,
        annotation_dir: str | Path | list[str | Path] | tuple[str | Path, ...],
        *,
        video_exts: Iterable[str] | None = None,
        annotation_exts: Iterable[str] | None = None,
        recursive: bool = False,
        strict: bool = False,
        allow_multiple_annotations_per_video: bool = False,
        num_frames: int | None = None,
        frame_size: tuple[int, int] | None = None,
        rgb: bool = True,
        return_tensor: bool = True,
    ) -> None:
        self.video_dir = Path(video_dir).expanduser().resolve()
        if isinstance(annotation_dir, (str, Path)):
            self.annotation_dirs = [
                Path(annotation_dir).expanduser().resolve()
            ]
        else:
            self.annotation_dirs = [
                Path(path).expanduser().resolve() for path in annotation_dir
            ]
        if not self.annotation_dirs:
            raise ValueError("annotation_dir must provide at least one directory.")

        self.recursive = recursive
        self.strict = strict
        self.allow_multiple_annotations_per_video = allow_multiple_annotations_per_video
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.rgb = rgb
        self.return_tensor = return_tensor

        self.video_exts = self._normalize_exts(video_exts or self.DEFAULT_VIDEO_EXTS)
        self.annotation_exts = self._normalize_exts(
            annotation_exts or self.DEFAULT_ANNOTATION_EXTS
        )

        self.video_paths = self._scan_files(self.video_dir, self.video_exts)
        self.annotation_paths: list[Path] = []
        for ann_dir in self.annotation_dirs:
            self.annotation_paths.extend(
                self._scan_files(ann_dir, self.annotation_exts)
            )
        self.annotation_paths = sorted(self.annotation_paths)

        self.video_index = self._build_video_index(self.video_paths)
        self.annotation_index = self._build_annotation_index(self.annotation_paths)

        self.pairs = self._match_video_annotation_pairs()

        self.samples: list[GestureAnnotation] = []
        for pair in self.pairs:
            self.samples.extend(
                self.parse_eaf_gesture_annotations(
                    pair.annotation_path,
                    pair.video_path,
                )
            )

    def __len__(self) -> int:
        # One item per gesture segment, not per video.
        return len(self.samples)

    @staticmethod
    def _annotation_dict_from_sample(sample: GestureAnnotation) -> dict:
        return {
            "video_path": str(sample.video_path),
            "annotation_path": str(sample.annotation_path),
            "video_filename": sample.video_path.name,
            "annotation_filename": sample.annotation_path.name,
            "start_ms": sample.start_ms,
            "end_ms": sample.end_ms,
            "duration_ms": sample.duration_ms,
            "marker": sample.marker,
            "task": sample.task,
            "gesture_type": sample.gesture_type,
            "function": sample.function,
            "left_hand_label": sample.left_hand_label,
            "left_hand_shape": sample.left_hand_shape,
            "right_hand_label": sample.right_hand_label,
            "right_hand_shape": sample.right_hand_shape,
            "raw_tier_values": sample.raw_tier_values,
        }

    def __getitem__(self, index: int) -> dict:
        """
        Load frames for one gesture segment.

        ``index`` selects ``self.samples[index]``, i.e. a single GESTURE interval
        from a matched video/EAF pair—not the entire video file.
        """
        sample = self.samples[index]
        frames = self._read_video_segment_frames(
            sample.video_path,
            sample.start_ms,
            sample.end_ms,
            self.num_frames,
            self.frame_size,
            self.rgb,
            self.return_tensor,
        )
        return {
            "frames": frames,
            "annotation": self._annotation_dict_from_sample(sample),
        }

    def get_annotation(self, index: int) -> GestureAnnotation:
        """Return the gesture annotation for ``index`` without loading video frames."""
        return self.samples[index]

    def get_annotations_for_video(self, video_filename: str) -> list[GestureAnnotation]:
        """Return all gesture segments whose ``video_path.name`` matches ``video_filename``."""
        return [
            sample
            for sample in self.samples
            if sample.video_path.name == video_filename
        ]

    @staticmethod
    def _normalize_exts(exts: Iterable[str]) -> set[str]:
        normalized: set[str] = set()

        for ext in exts:
            ext = ext.lower().strip()
            if not ext.startswith("."):
                ext = f".{ext}"
            normalized.add(ext)

        return normalized

    def _scan_files(self, directory: Path, allowed_exts: set[str]) -> list[Path]:
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected directory, got: {directory}")

        pattern = "**/*" if self.recursive else "*"

        return sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in allowed_exts
        )

    @classmethod
    def _parse_video_key(cls, video_path: Path) -> VideoKey | None:
        """
        Parse names like:
            65-1.mp4
            68-2.mp4
        """
        match = cls.VIDEO_NAME_RE.match(video_path.stem)

        if match is None:
            return None

        return VideoKey(
            subject_id=match.group("subject_id"),
            day=match.group("day"),
        )

    @classmethod
    def _parse_annotation_key_from_filename(
        cls,
        annotation_path: Path,
    ) -> AnnotationKey | None:
        """
        Parse annotation filenames into (subject_id, day).

        Tries day-word patterns first (e.g. rc65_day1_AB, nbd65.day1.KU), then
        dash-day patterns (e.g. rc65-1.gesturecoded, 54-1.coded), then
        dotted-prefix day-word patterns (e.g. rc.199.day1.coded.BB). Optional
        rc/nbd prefixes are ignored for matching.

        Subject-only underscore stems (e.g. rc110_gesturesoutofview) fall back
        to the EAF MEDIA_DESCRIPTOR reference.
        """
        stem = annotation_path.stem

        for pattern in (
            cls.ANNOTATION_DAY_WORD_RE,
            cls.ANNOTATION_DASH_DAY_RE,
            cls.ANNOTATION_PREFIX_DOT_DAY_WORD_RE,
        ):
            match = pattern.match(stem)
            if match is not None:
                return AnnotationKey(
                    subject_id=match.group("subject_id"),
                    day=match.group("day"),
                )

        if cls.ANNOTATION_SUBJECT_UNDERSCORE_SUFFIX_RE.match(stem):
            return cls._parse_annotation_key_from_eaf_media(annotation_path)

        return None

    @staticmethod
    def _pair_key(subject_id: str, day: str) -> tuple[str, str]:
        return subject_id.lower(), day.lower()

    def _build_video_index(
        self,
        video_paths: list[Path],
    ) -> dict[tuple[str, str], tuple[VideoKey, Path]]:
        index: dict[tuple[str, str], tuple[VideoKey, Path]] = {}
        skipped: list[Path] = []

        for video_path in video_paths:
            video_key = self._parse_video_key(video_path)

            if video_key is None:
                skipped.append(video_path)
                continue

            key = self._pair_key(video_key.subject_id, video_key.day)

            if key in index:
                existing_key, existing_path = index[key]
                raise ValueError(
                    "Duplicate videos resolve to the same subject/day key:\n"
                    f"Key: {existing_key}\n"
                    f"Existing: {existing_path}\n"
                    f"New: {video_path}"
                )

            index[key] = (video_key, video_path)

        if self.strict and skipped:
            skipped_text = "\n".join(str(path) for path in skipped)
            raise ValueError(
                "Some video filenames did not match the expected pattern "
                "'<subject_id>-<day>.mp4':\n"
                f"{skipped_text}"
            )

        return index

    def _build_annotation_index(
        self,
        annotation_paths: list[Path],
    ) -> dict[tuple[str, str], list[tuple[AnnotationKey, Path]]]:
        index: dict[tuple[str, str], list[tuple[AnnotationKey, Path]]] = {}
        skipped: list[Path] = []

        for annotation_path in annotation_paths:
            annotation_key = self._parse_annotation_key_from_filename(annotation_path)

            if annotation_key is None:
                annotation_key = self._parse_annotation_key_from_eaf_media(
                    annotation_path
                )

            if annotation_key is None:
                skipped.append(annotation_path)
                continue

            key = self._pair_key(annotation_key.subject_id, annotation_key.day)
            index.setdefault(key, []).append((annotation_key, annotation_path))

        if self.strict and skipped:
            skipped_text = "\n".join(str(path) for path in skipped)
            raise ValueError(
                "Some annotation filenames did not match any supported pattern.\n"
                f"{self.SUPPORTED_ANNOTATION_PATTERNS_MSG}\n"
                f"{skipped_text}"
            )

        for key in index:
            index[key] = sorted(index[key], key=lambda item: item[1].name.lower())

        return index

    def _match_video_annotation_pairs(self) -> list[VideoAnnotationPair]:
        pairs: list[VideoAnnotationPair] = []
        ambiguous_videos: list[tuple[Path, list[Path]]] = []

        for key, (video_key, video_path) in sorted(
            self.video_index.items(),
            key=lambda item: item[1][1].name.lower(),
        ):
            annotation_candidates = self.annotation_index.get(key, [])

            if not annotation_candidates:
                continue

            if (
                len(annotation_candidates) > 1
                and not self.allow_multiple_annotations_per_video
            ):
                ambiguous_videos.append(
                    (
                        video_path,
                        [annotation_path for _, annotation_path in annotation_candidates],
                    )
                )
                continue

            for annotation_key, annotation_path in annotation_candidates:
                pairs.append(
                    VideoAnnotationPair(
                        video_path=video_path,
                        annotation_path=annotation_path,
                        video_key=video_key,
                        annotation_key=annotation_key,
                    )
                )

        if self.strict and ambiguous_videos:
            details = []

            for video_path, annotation_paths in ambiguous_videos:
                annotation_text = "\n    ".join(str(path) for path in annotation_paths)
                details.append(f"{video_path}\n    {annotation_text}")

            raise ValueError(
                "Multiple annotation files matched the same video. "
                "Set allow_multiple_annotations_per_video=True if this is expected.\n\n"
                + "\n\n".join(details)
            )

        return pairs

    @classmethod
    def _parse_annotation_key_from_eaf_media(
        cls,
        annotation_path: Path,
    ) -> AnnotationKey | None:
        """
        Optional fallback.

        If filename parsing fails, try reading the EAF media reference.

        Example inside EAF:
            RELATIVE_MEDIA_URL="./100-1.mp4"

        Then:
            100-1.mp4 -> subject_id=100, day=1
        """
        media_filenames = cls._extract_media_filenames_from_eaf(annotation_path)

        for media_filename in media_filenames:
            video_like_path = Path(media_filename)
            video_key = cls._parse_video_key(video_like_path)

            if video_key is not None:
                return AnnotationKey(
                    subject_id=video_key.subject_id,
                    day=video_key.day,
                )

        return None

    @staticmethod
    def parse_eaf_gesture_annotations(
        annotation_path: Path,
        video_path: Path,
    ) -> list[GestureAnnotation]:
        text = _read_text_with_fallback(annotation_path)
        root = ET.fromstring(text)

        time_slots: dict[str, int] = {}
        for time_slot in root.findall(".//TIME_SLOT"):
            slot_id = time_slot.attrib.get("TIME_SLOT_ID")
            time_value = time_slot.attrib.get("TIME_VALUE")
            if slot_id is None or time_value is None:
                continue
            time_slots[slot_id] = int(time_value)

        tier_annotations: dict[str, list[tuple[int, int, str]]] = {}

        for tier in root.findall(".//TIER"):
            tier_id = tier.attrib.get("TIER_ID")
            if not tier_id:
                continue

            for alignable in tier.findall(".//ALIGNABLE_ANNOTATION"):
                ref1 = alignable.attrib.get("TIME_SLOT_REF1")
                ref2 = alignable.attrib.get("TIME_SLOT_REF2")
                if ref1 is None or ref2 is None:
                    continue
                if ref1 not in time_slots or ref2 not in time_slots:
                    continue

                start_ms = time_slots[ref1]
                end_ms = time_slots[ref2]

                value_elem = alignable.find("ANNOTATION_VALUE")
                if value_elem is None or value_elem.text is None:
                    continue

                value = value_elem.text.strip()
                tier_annotations.setdefault(tier_id, []).append(
                    (start_ms, end_ms, value)
                )

        marker_segments = [
            (start_ms, end_ms, value)
            for start_ms, end_ms, value in tier_annotations.get("marker", [])
            if value == "GESTURE"
        ]

        samples: list[GestureAnnotation] = []

        for start_ms, end_ms, marker_value in marker_segments:
            raw_tier_values: dict[str, str | None] = {}

            task = _get_best_tier_value_for_interval(
                tier_annotations.get("task", []), start_ms, end_ms
            )
            gesture_type = _get_best_tier_value_for_interval(
                tier_annotations.get("type", []), start_ms, end_ms
            )
            function = _get_best_tier_value_for_interval(
                tier_annotations.get("function", []), start_ms, end_ms
            )
            left_hand_label = _get_best_tier_value_for_interval(
                tier_annotations.get("LEF@PAR", []), start_ms, end_ms
            )
            left_hand_shape = _get_best_tier_value_for_interval(
                tier_annotations.get("HSL@PAR", []), start_ms, end_ms
            )
            right_hand_label = _get_best_tier_value_for_interval(
                tier_annotations.get("RIG@PAR", []), start_ms, end_ms
            )
            right_hand_shape = _get_best_tier_value_for_interval(
                tier_annotations.get("HSR@PAR", []), start_ms, end_ms
            )

            for tier_name in GESTURE_ATTRIBUTE_TIERS:
                raw_tier_values[tier_name] = _get_best_tier_value_for_interval(
                    tier_annotations.get(tier_name, []), start_ms, end_ms
                )

            samples.append(
                GestureAnnotation(
                    video_path=video_path,
                    annotation_path=annotation_path,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    marker=marker_value,
                    task=task,
                    gesture_type=gesture_type,
                    function=function,
                    left_hand_label=left_hand_label,
                    left_hand_shape=left_hand_shape,
                    right_hand_label=right_hand_label,
                    right_hand_shape=right_hand_shape,
                    raw_tier_values=raw_tier_values,
                )
            )

        return samples

    @staticmethod
    def _read_video_segment_frames(
        video_path: Path,
        start_ms: int,
        end_ms: int,
        num_frames: int | None,
        frame_size: tuple[int, int] | None,
        rgb: bool,
        return_tensor: bool,
    ) -> torch.Tensor | np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or math.isnan(fps):
                raise RuntimeError(
                    f"Invalid FPS ({fps}) for video: {video_path}"
                )

            total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frame_count <= 0:
                raise RuntimeError(
                    f"Invalid frame count ({total_frame_count}) for video: {video_path}"
                )

            start_frame = math.floor(start_ms / 1000.0 * fps)
            end_frame = math.ceil(end_ms / 1000.0 * fps)

            start_frame = max(0, min(start_frame, total_frame_count - 1))
            end_frame = max(0, min(end_frame, total_frame_count))

            if end_frame <= start_frame:
                if start_frame < total_frame_count - 1:
                    end_frame = start_frame + 1
                else:
                    raise RuntimeError(
                        f"Segment timestamps produced no valid frame range for "
                        f"video {video_path}: start_ms={start_ms}, end_ms={end_ms}, "
                        f"fps={fps}, start_frame={start_frame}, end_frame={end_frame}"
                    )

            if num_frames is None:
                frame_indices = list(range(start_frame, end_frame))
            else:
                if num_frames <= 0:
                    raise ValueError(f"num_frames must be positive, got {num_frames}")
                last_frame = end_frame - 1
                if num_frames == 1:
                    frame_indices = [start_frame]
                else:
                    frame_indices = [
                        int(round(start_frame + i * (last_frame - start_frame) / (num_frames - 1)))
                        for i in range(num_frames)
                    ]

            frames: list[np.ndarray] = []
            for frame_idx in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                if frame_size is not None:
                    height, width = frame_size
                    frame = cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )

                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)

            if not frames:
                raise RuntimeError(
                    f"No frames could be read from video {video_path} for segment "
                    f"[{start_ms}, {end_ms}] ms "
                    f"(requested frame indices: {frame_indices})"
                )

            stacked = np.stack(frames, axis=0)

            if return_tensor:
                tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
                return tensor.float().div(255.0)

            return stacked.astype(np.uint8, copy=False)
        finally:
            capture.release()

    @staticmethod
    def _extract_media_filenames_from_eaf(annotation_path: Path) -> list[str]:
        try:
            text = _read_text_with_fallback(annotation_path)
        except OSError:
            return []

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

        media_filenames: list[str] = []

        for media_descriptor in root.findall(".//MEDIA_DESCRIPTOR"):
            for attr_name in ("RELATIVE_MEDIA_URL", "MEDIA_URL"):
                media_ref = media_descriptor.attrib.get(attr_name)

                if not media_ref:
                    continue

                filename = AphasiaDataset._filename_from_media_ref(media_ref)

                if filename:
                    media_filenames.append(filename)

        return media_filenames

    @staticmethod
    def _filename_from_media_ref(media_ref: str) -> str | None:
        media_ref = media_ref.strip()

        if not media_ref:
            return None

        if media_ref.startswith("file://"):
            media_ref = media_ref.replace("file://", "", 1)

        return Path(media_ref).name or None

    def get_pair(self, index: int) -> VideoAnnotationPair:
        """
        Helper for debugging matched pairs.
        """
        return self.pairs[index]

    def print_matched_pairs(self, max_rows: int | None = None) -> None:
        """
        Optional debugging helper.
        """
        rows = self.pairs if max_rows is None else self.pairs[:max_rows]

        for pair in rows:
            print(f"{pair.video_path.name}  -->  {pair.annotation_path.name}")

    def debug_parse_annotation_filenames(self, max_rows: int = 50) -> None:
        """
        Print annotation filename parsing results for debugging.
        """
        rows = self.annotation_paths if max_rows is None else self.annotation_paths[:max_rows]

        print("Annotation filename parsing:")
        for annotation_path in rows:
            annotation_key = self._parse_annotation_key_from_filename(annotation_path)
            if annotation_key is None:
                print(
                    f"{annotation_path.name}\t"
                    f"subject_id=\t"
                    f"day=\t"
                    f"parsed=False"
                )
            else:
                print(
                    f"{annotation_path.name}\t"
                    f"subject_id={annotation_key.subject_id}\t"
                    f"day={annotation_key.day}\t"
                    f"parsed=True"
                )

    def print_gesture_samples(self, max_rows: int | None = None) -> None:
        """
        Print parsed gesture segments for debugging.
        """
        rows = self.samples if max_rows is None else self.samples[:max_rows]

        for sample in rows:
            print(
                f"{sample.video_path.name}\t"
                f"{sample.annotation_path.name}\t"
                f"{sample.start_ms}\t"
                f"{sample.end_ms}\t"
                f"{sample.task}\t"
                f"{sample.gesture_type}\t"
                f"{sample.function}\t"
                f"{sample.left_hand_label}"
            )


class AphasiaGestureSegmentDataset(Dataset):
    """
    Pre-extracted gesture segments with stored pose landmarks and transcripts.

    Expected layout::

        {landmarks_root}/{video_filename}/gest_{idx}/landmarks.json
        {landmarks_root}/{video_filename}/gest_{idx}/annotation.txt
        {negative_landmarks_root}/{video_filename}/neg_{idx}/landmarks.json
        {negative_landmarks_root}/{video_filename}/neg_{idx}/annotation.txt

    Each dataset index is one segment. Positive ``gest_*`` segments are labeled
    ``1`` (``POS``); negative ``neg_*`` segments are labeled ``0`` (``NEG``).
    Original ``gesture_type`` values in ``annotation.txt`` are ignored.

    ``__getitem__`` returns:

    - ``landmarks``: Tensor or NumPy array
    - ``transcript_chunks``: list of ``num_transcript_chunks`` transcript strings
    - ``transcript_chunk_mask``: bool tensor ``[num_transcript_chunks]``
    - ``transcript``: joined non-empty chunk text (legacy/debug convenience)
    - ``aligned_utt_start_ts``: aligned utterance start timestamp when available
    - ``aligned_utt_end_ts``: aligned utterance end timestamp when available
    - ``aligned_utterance``: aligned utterance text when available
    - ``full_transcript``: full aligned transcript snippet when available
    - ``transcript_alignment_mask``: float tensor ``[num_transcript_chunks]`` with
      ``1.0`` on the chunk best matching the aligned utterance and ``0.0`` elsewhere
    - ``label``: ``0`` or ``1``
    - ``gesture_type``: ``NEG`` or ``POS``
    - ``annotation``: segment metadata

    MediaPipe segments use face + upper body (25 joints), shape
    ``[T, NUM_MEDIAPIPE_FACE_UPPER_BODY_LANDMARKS, 4]``. OpenPose BODY_25 plus
    optional HAND_21 joints are mapped into the legacy 14-joint upper-body
    layout, shape ``[T, NUM_UPPER_BODY_LANDMARKS, 4]``.
    """

    DEFAULT_LANDMARKS_ROOT = (
        "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments_landmarks"
    )
    DEFAULT_NEGATIVE_LANDMARKS_ROOT = (
        "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments_landmarks"
    )
    DEFAULT_ALIGNMENT_ANNS_DIR = (
        "/data/quantization/zaima/aphasiabank/neural2/control/ts_videos/alignment_anns"
    )

    def __init__(
        self,
        landmarks_dir: str | Path | None = None,
        segments_dir: str | Path | None = None,
        negative_landmarks_dir: str | Path | None = None,
        *,
        transcript_dir: str | Path | None = None,
        strict: bool = False,
        strict_transcripts: bool | None = None,
        transcript_kwargs: dict | None = None,
        transcript_offset_seconds: float = 4.0,
        num_transcript_chunks: int = 5,
        num_frames: int | None = None,
        frame_size: tuple[int, int] | None = None,
        rgb: bool = True,
        return_tensor: bool = True,
        per_video_samples: int | None = None,
        sample_indices: Sequence[int] | None = None,
    ) -> None:
        if landmarks_dir is None:
            landmarks_dir = segments_dir or self.DEFAULT_LANDMARKS_ROOT

        self.landmarks_root = Path(landmarks_dir).expanduser().resolve()
        self.strict = strict
        self.strict_transcripts = self.strict if strict_transcripts is None else bool(
            strict_transcripts
        )
        self.transcript_kwargs = (
            dict(transcript_kwargs) if transcript_kwargs is not None else None
        )
        if num_transcript_chunks <= 0:
            raise ValueError(
                f"num_transcript_chunks must be positive, got {num_transcript_chunks}"
            )
        if transcript_offset_seconds < 0:
            raise ValueError(
                "transcript_offset_seconds must be non-negative, got "
                f"{transcript_offset_seconds}"
            )
        self.transcript_offset_seconds = float(transcript_offset_seconds)
        self.num_transcript_chunks = int(num_transcript_chunks)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.rgb = rgb
        self.return_tensor = return_tensor
        self.per_video_samples = (
            None if per_video_samples is None or per_video_samples <= 0 else int(per_video_samples)
        )

        if not self.landmarks_root.is_dir():
            raise FileNotFoundError(
                f"Gesture landmarks directory does not exist: {self.landmarks_root}"
            )

        if negative_landmarks_dir is None:
            negative_landmarks_dir = self.DEFAULT_NEGATIVE_LANDMARKS_ROOT

        self.negative_landmarks_root = (
            Path(negative_landmarks_dir).expanduser().resolve()
        )
        if not self.negative_landmarks_root.is_dir():
            raise FileNotFoundError(
                "Negative gesture landmarks directory does not exist: "
                f"{self.negative_landmarks_root}"
            )

        self.transcript_dir = (
            Path(transcript_dir).expanduser().resolve()
            if transcript_dir is not None
            else None
        )
        self._transcript_by_media_ref: dict[str, list[Path]] = {}
        self._transcript_by_pair_key: dict[tuple[str, str], list[Path]] = {}
        self._transcript_lookup_warned: set[str] = set()
        self._missing_transcript_warned: set[str] = set()
        self._transcript_cache: dict[
            tuple[str, int, int, tuple[tuple[str, Any], ...], float, int],
            tuple[list[str], torch.Tensor],
        ] = {}

        self.alignment_anns_dir = Path(
            self.DEFAULT_ALIGNMENT_ANNS_DIR
        ).expanduser().resolve()
        self._alignment_anns_by_video = _scan_alignment_anns_dir(self.alignment_anns_dir)
        self._alignment_anns_payload_cache: dict[str, list[dict[str, Any]]] = {}
        self._missing_alignment_warned: set[str] = set()
        self._alignment_mismatch_warned: set[str] = set()

        if self.transcript_dir is not None:
            if not self.transcript_dir.is_dir():
                raise FileNotFoundError(
                    f"Transcript directory does not exist: {self.transcript_dir}"
                )
            cha_paths = _scan_cha_transcript_files(self.transcript_dir)
            (
                self._transcript_by_media_ref,
                self._transcript_by_pair_key,
            ) = _build_transcript_lookup(
                cha_paths,
                strict_transcripts=self.strict_transcripts,
                warned_keys=self._transcript_lookup_warned,
            )

        self.samples = self._assign_transcript_paths(self._scan_gesture_segments())

        if self.strict and not self.samples:
            raise ValueError(
                f"No gesture segments found under: {self.landmarks_root}"
            )

        if sample_indices is None:
            self.sample_indices = list(range(len(self.samples)))
        else:
            self.sample_indices = [int(index) for index in sample_indices]

        self.video_to_indices = self._build_video_to_indices()
        self.epoch = 0
        self.active_indices = list(self.sample_indices)
        if self.per_video_samples is not None:
            self.resample_epoch(self.epoch)

    @classmethod
    def from_existing(
        cls,
        dataset: AphasiaGestureSegmentDataset,
        *,
        sample_indices: Sequence[int],
        per_video_samples: int | None = None,
    ) -> AphasiaGestureSegmentDataset:
        """Reuse scanned samples from ``dataset`` for a train/val split view."""
        view = cls.__new__(cls)
        view.landmarks_root = dataset.landmarks_root
        view.negative_landmarks_root = dataset.negative_landmarks_root
        view.strict = dataset.strict
        view.strict_transcripts = dataset.strict_transcripts
        view.transcript_kwargs = dataset.transcript_kwargs
        view.transcript_offset_seconds = dataset.transcript_offset_seconds
        view.num_transcript_chunks = dataset.num_transcript_chunks
        view.transcript_dir = dataset.transcript_dir
        view._transcript_by_media_ref = dataset._transcript_by_media_ref
        view._transcript_by_pair_key = dataset._transcript_by_pair_key
        view._transcript_lookup_warned = dataset._transcript_lookup_warned
        view._missing_transcript_warned = dataset._missing_transcript_warned
        view._transcript_cache = dataset._transcript_cache
        view.alignment_anns_dir = dataset.alignment_anns_dir
        view._alignment_anns_by_video = dataset._alignment_anns_by_video
        view._alignment_anns_payload_cache = dataset._alignment_anns_payload_cache
        view._missing_alignment_warned = dataset._missing_alignment_warned
        view._alignment_mismatch_warned = dataset._alignment_mismatch_warned
        view.num_frames = dataset.num_frames
        view.frame_size = dataset.frame_size
        view.rgb = dataset.rgb
        view.return_tensor = dataset.return_tensor
        view.per_video_samples = (
            None
            if per_video_samples is None or per_video_samples <= 0
            else int(per_video_samples)
        )
        view.samples = dataset.samples
        view.sample_indices = [int(index) for index in sample_indices]
        view.video_to_indices = view._build_video_to_indices()
        view.epoch = 0
        view.active_indices = list(view.sample_indices)
        if view.per_video_samples is not None:
            view.resample_epoch(view.epoch)
        return view

    @property
    def num_videos(self) -> int:
        return len(self.video_to_indices)

    def _build_video_to_indices(self) -> dict[str, list[int]]:
        video_to_indices: dict[str, list[int]] = defaultdict(list)
        for index in self.sample_indices:
            video_name = self.samples[index].video_filename
            video_to_indices[video_name].append(index)
        return dict(video_to_indices)

    def set_epoch(self, epoch: int) -> None:
        """Resample the active training indices for a new epoch."""
        self.resample_epoch(epoch)

    def resample_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.epoch = int(epoch)

        if self.per_video_samples is None:
            self.active_indices = list(self.sample_indices)
            return

        rng = random.Random(self.epoch)
        active_indices: list[int] = []
        for video_name in sorted(self.video_to_indices):
            indices = self.video_to_indices[video_name]
            sample_count = self.per_video_samples
            if len(indices) >= sample_count:
                active_indices.extend(rng.sample(indices, sample_count))
            else:
                active_indices.extend(rng.choices(indices, k=sample_count))
        self.active_indices = active_indices

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, index: int) -> dict:
        real_index = self.active_indices[index]
        sample = self.samples[real_index]
        # video_key = _parse_video_filename_key(sample.video_filename)
        # if (
        #     video_key is not None
        #     and video_key.subject_id in DEBUG_GETITEM_VIDEO_SUBJECT_IDS
        # ):
            # print(sample.video_filename)

        landmarks = self._read_segment_landmarks(sample.landmarks_path)
        transcript_chunks, transcript_chunk_mask = self._extract_segment_transcript_chunks(
            sample
        )
        transcript = "\n".join(
            chunk for chunk in transcript_chunks if str(chunk).strip()
        )
        gesture_text_alignment = self._extract_gesture_text_alignment(sample)
        transcript_alignment_mask = _build_transcript_alignment_mask(
            transcript_chunks,
            full_transcript=gesture_text_alignment.get("full_transcript"),
            aligned_utterance=gesture_text_alignment.get("aligned_utterance"),
        )

        return {
            "landmarks": landmarks,
            "transcript_chunks": transcript_chunks,
            "transcript_chunk_mask": transcript_chunk_mask,
            "transcript": transcript,
            "transcript_alignment_mask": transcript_alignment_mask,
            **gesture_text_alignment,
            "label": sample.label,
            "gesture_type": sample.label_name,
            "annotation": sample.annotation,
        }

    def _get_alignment_entries_for_video(
        self,
        video_filename: str,
    ) -> list[dict[str, Any]] | None:
        alignment_path = self._alignment_anns_by_video.get(video_filename)
        if alignment_path is None:
            alignment_path = _resolve_alignment_anns_path(
                self.alignment_anns_dir,
                video_filename,
            )
            if alignment_path is None:
                return None
            self._alignment_anns_by_video[video_filename] = alignment_path

        cached = self._alignment_anns_payload_cache.get(video_filename)
        if cached is not None:
            return cached

        try:
            annotations = _load_alignment_anns_payload(
                alignment_path,
                expected_video_filename=video_filename,
            )
        except ValueError as exc:
            warning_key = f"{video_filename}:{alignment_path.name}"
            if warning_key not in self._alignment_mismatch_warned:
                warnings.warn(str(exc), stacklevel=2)
                self._alignment_mismatch_warned.add(warning_key)
            return None

        self._alignment_anns_payload_cache[video_filename] = annotations
        return annotations

    def _extract_gesture_text_alignment(
        self,
        sample: GestureSegmentSample,
    ) -> dict[str, str | None]:
        alignment_entries = self._get_alignment_entries_for_video(sample.video_filename)
        if alignment_entries is None:
            return _empty_gesture_text_alignment()

        annotation_id = sample.annotation.get("annotation_id")
        matched = _find_gesture_text_alignment(
            alignment_entries,
            annotation_id=annotation_id,
        )
        if matched is None:
            return _empty_gesture_text_alignment()
        return matched

    def _resolve_transcript_path(self, video_filename: str) -> Path | None:
        if self.transcript_dir is None:
            return None

        media_ref = Path(video_filename).stem.lower()
        media_candidates = self._transcript_by_media_ref.get(media_ref, [])
        selected = _choose_transcript_candidate(
            media_candidates,
            video_filename=video_filename,
            match_kind="@Media",
            strict_transcripts=self.strict_transcripts,
            warned_keys=self._transcript_lookup_warned,
        )
        if selected is not None:
            return selected

        video_key = _parse_video_filename_key(video_filename)
        if video_key is None:
            return None

        pair = _pair_key(video_key.subject_id, video_key.day)
        pair_candidates = self._transcript_by_pair_key.get(pair, [])
        return _choose_transcript_candidate(
            pair_candidates,
            video_filename=video_filename,
            match_kind=f"subject/day {pair!r}",
            strict_transcripts=self.strict_transcripts,
            warned_keys=self._transcript_lookup_warned,
        )

    def _assign_transcript_paths(
        self,
        samples: list[GestureSegmentSample],
    ) -> list[GestureSegmentSample]:
        if self.transcript_dir is None:
            return list(samples)

        assigned: list[GestureSegmentSample] = []
        for sample in samples:
            transcript_path = self._resolve_transcript_path(sample.video_filename)
            assigned.append(replace(sample, transcript_path=transcript_path))
        return assigned

    def _validate_segment_timestamps(self, sample: GestureSegmentSample) -> tuple[int, int]:
        start_ms = sample.annotation.get("start_ms")
        end_ms = sample.annotation.get("end_ms")

        if not isinstance(start_ms, int) or isinstance(start_ms, bool):
            raise TypeError(
                f"Segment {sample.segment_dir} has invalid start_ms: {start_ms!r}"
            )
        if not isinstance(end_ms, int) or isinstance(end_ms, bool):
            raise TypeError(
                f"Segment {sample.segment_dir} has invalid end_ms: {end_ms!r}"
            )
        if start_ms >= end_ms:
            raise ValueError(
                f"Segment {sample.segment_dir} has invalid timestamp window: "
                f"start_ms={start_ms}, end_ms={end_ms}"
            )

        return start_ms, end_ms

    def _extract_segment_transcript_chunks(
        self,
        sample: GestureSegmentSample,
    ) -> tuple[list[str], torch.Tensor]:
        empty_chunks = [""] * self.num_transcript_chunks
        empty_mask = torch.zeros(self.num_transcript_chunks, dtype=torch.bool)

        if sample.transcript_path is None:
            if self.transcript_dir is None:
                return empty_chunks, empty_mask

            # warning_key = sample.video_filename.lower()
            # message = (
            #     f"No CHAT transcript found for video {sample.video_filename!r} "
            #     f"(segment {sample.segment_dir})"
            # )
            # if self.strict_transcripts:
            #     raise ValueError(message)

            # if warning_key not in self._missing_transcript_warned:
            #     warnings.warn(message, stacklevel=2)
            #     self._missing_transcript_warned.add(warning_key)
            return empty_chunks, empty_mask

        try:
            start_ms, end_ms = self._validate_segment_timestamps(sample)
        except (TypeError, ValueError) as exc:
            message = (
                f"Invalid transcript timestamps for segment {sample.segment_dir}: "
                f"{exc}"
            )
            if self.strict_transcripts:
                raise ValueError(message) from exc

            warning_key = f"timestamps:{sample.segment_dir}"
            if warning_key not in self._missing_transcript_warned:
                warnings.warn(message, stacklevel=2)
                self._missing_transcript_warned.add(warning_key)
            return empty_chunks, empty_mask

        cache_key = _transcript_cache_key(
            sample.transcript_path,
            start_ms,
            end_ms,
            self.transcript_kwargs,
            transcript_offset_seconds=self.transcript_offset_seconds,
            num_transcript_chunks=self.num_transcript_chunks,
        )
        cached = self._transcript_cache.get(cache_key)
        if cached is not None:
            return cached

        offset_ms = int(round(self.transcript_offset_seconds * 1000.0))
        window_start_ms = max(0, start_ms - offset_ms)
        window_end_ms = end_ms + offset_ms

        transcript_lines = get_transcript_lines(
            sample.transcript_path,
            window_start_ms,
            window_end_ms,
            context_before_ms=0,
            context_after_ms=0,
            min_utterances_before=0,
            min_utterances_after=0,
        )
        transcript_chunks, transcript_chunk_mask = _build_transcript_chunks(
            transcript_lines,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            num_chunks=self.num_transcript_chunks,
        )
        assert len(transcript_chunks) == self.num_transcript_chunks
        assert transcript_chunk_mask.shape == (self.num_transcript_chunks,)

        self._transcript_cache[cache_key] = (transcript_chunks, transcript_chunk_mask)
        return transcript_chunks, transcript_chunk_mask

    def _extract_segment_transcript(self, sample: GestureSegmentSample) -> str:
        transcript_chunks, _ = self._extract_segment_transcript_chunks(sample)
        return "\n".join(chunk for chunk in transcript_chunks if str(chunk).strip())

    def get_annotation(self, index: int) -> GestureSegmentSample:
        """Return segment metadata for ``index`` without loading frames."""
        return self.samples[self.active_indices[index]]

    def _scan_labeled_segments(
        self,
        landmarks_root: Path,
        segment_dir_re: re.Pattern[str],
        label: int,
    ) -> tuple[list[GestureSegmentSample], list[str]]:
        samples: list[GestureSegmentSample] = []
        skipped: list[str] = []

        for video_dir in sorted(landmarks_root.iterdir()):
            if not video_dir.is_dir():
                continue

            video_filename = video_dir.name

            for segment_dir in sorted(video_dir.iterdir()):
                if not segment_dir.is_dir():
                    continue

                dir_match = segment_dir_re.match(segment_dir.name)
                if dir_match is None:
                    continue

                segment_index = int(dir_match.group(1))
                landmarks_path = segment_dir / "landmarks.json"
                if not landmarks_path.is_file():
                    skipped.append(f"{segment_dir}: missing landmarks.json")
                    continue

                annotation_path = segment_dir / "annotation.txt"
                if not annotation_path.is_file():
                    skipped.append(f"{segment_dir}: missing annotation.txt")
                    continue

                annotation = parse_segment_annotation_file(annotation_path)
                annotation.setdefault("video_filename", video_filename)
                annotation.setdefault("segment_dir", str(segment_dir))
                annotation.setdefault("segment_index", segment_index)
                annotation.setdefault("landmarks_path", str(landmarks_path))
                annotation["label"] = label
                annotation["label_name"] = LABEL_NAME_BY_VALUE[label]

                samples.append(
                    GestureSegmentSample(
                        segment_dir=segment_dir,
                        landmarks_path=landmarks_path,
                        video_filename=video_filename,
                        gesture_index=segment_index,
                        label=label,
                        annotation=annotation,
                    )
                )

        return samples, skipped

    def _scan_gesture_segments(self) -> list[GestureSegmentSample]:
        samples, skipped = self._scan_labeled_segments(
            self.landmarks_root,
            GESTURE_SEGMENT_DIR_RE,
            POSITIVE_SEGMENT_LABEL,
        )

        negative_samples, negative_skipped = self._scan_labeled_segments(
            self.negative_landmarks_root,
            NEGATIVE_SEGMENT_DIR_RE,
            NEGATIVE_SEGMENT_LABEL,
        )
        samples.extend(negative_samples)
        skipped.extend(negative_skipped)

        if self.strict and skipped:
            skipped_text = "\n".join(skipped)
            raise ValueError(
                "Some gesture segment directories were invalid:\n"
                f"{skipped_text}"
            )

        samples.sort(
            key=lambda sample: (
                sample.video_filename.lower(),
                sample.label,
                sample.gesture_index,
            )
        )
        return samples

    def _read_segment_landmarks(
        self,
        landmarks_path: Path,
    ) -> torch.Tensor | np.ndarray:
        return read_segment_landmarks(
            landmarks_path,
            num_frames=self.num_frames,
            return_tensor=self.return_tensor,
        )

    def _read_segment_frames(
        self,
        segment_dir: Path,
    ) -> torch.Tensor | np.ndarray:
        frame_paths = list_segment_frame_paths(segment_dir)
        if not frame_paths:
            raise RuntimeError(f"No frame PNG files found in: {segment_dir}")

        if self.num_frames is None:
            selected_paths = frame_paths
        else:
            if self.num_frames <= 0:
                raise ValueError(f"num_frames must be positive, got {self.num_frames}")

            if self.num_frames == 1:
                selected_paths = [frame_paths[0]]
            else:
                last_index = len(frame_paths) - 1
                selected_paths = [
                    frame_paths[
                        int(round(i * last_index / (self.num_frames - 1)))
                    ]
                    for i in range(self.num_frames)
                ]

        frames: list[np.ndarray] = []
        for frame_path in selected_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read frame image: {frame_path}")

            if self.frame_size is not None:
                height, width = self.frame_size
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        stacked = np.stack(frames, axis=0)

        if self.return_tensor:
            tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
            return tensor.float().div(255.0)

        return stacked.astype(np.uint8, copy=False)

    def print_gesture_samples(self, max_rows: int | None = None) -> None:
        rows = self.samples if max_rows is None else self.samples[:max_rows]

        for sample in rows:
            print(
                f"{sample.video_filename}\t"
                f"{sample.segment_dir.name}\t"
                f"{sample.annotation.get('start_ms')}\t"
                f"{sample.annotation.get('end_ms')}\t"
                f"{sample.label}\t"
                f"{sample.label_name}"
            )


AphasiaGestureEAFDataset = AphasiaDataset