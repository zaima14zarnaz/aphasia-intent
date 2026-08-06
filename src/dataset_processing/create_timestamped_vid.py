#!/usr/bin/env python3
"""
Export padded gesture segment videos with timestamps and transcript overlays.

When ``--src_gest_ann`` is provided, each annotated gesture produces a separate
short clip from the source video, padded by ``FRAME_PADDING`` frames before and
after the gesture interval. Clips are written to ``DEST_VIDEO_DIR`` as:

    {video_stem}_{gesture_index}_{gesture_type}.mp4

A companion metadata JSON file is written per source video with details for
each exported gesture segment.

When gesture annotations are not provided, the script falls back to writing a
single timestamped full-length video per source file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pre_process.dataloder import AphasiaDataset, GestureAnnotation

from dataset_processing.read_transcript import get_transcript_lines

SRC_VIDEO_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/videos"
)
DEST_VIDEO_DIR = Path(
    "/data/quantization/zaima/aphasiabank/neural2/control/ts_videos"
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
FRAME_PADDING = 100

TIMESTAMP_FONT = cv2.FONT_HERSHEY_SIMPLEX
TIMESTAMP_FONT_SCALE = 0.7
TIMESTAMP_FONT_THICKNESS = 2
TIMESTAMP_MARGIN_PX = 10
TIMESTAMP_LINE_GAP_PX = 4
TIMESTAMP_TEXT_COLOR = (255, 255, 255)
TIMESTAMP_BG_COLOR = (0, 0, 0)
NOT_AVAILABLE = "N/A"


@dataclass(frozen=True)
class GestAnn:
    gesture_index: int
    annotation_id: str | None
    start_ms: int
    end_ms: int
    duration_ms: int
    gesture_type: str
    marker: str | None = None
    task: str | None = None
    function: str | None = None
    left_hand_label: str | None = None
    left_hand_shape: str | None = None
    right_hand_label: str | None = None
    right_hand_shape: str | None = None
    raw_tier_values: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoExportRecord:
    src_path: Path
    eaf_path: Path
    cha_path: Path | None
    fps: float
    width: int
    height: int
    total_frames: int
    gestures: tuple[GestAnn, ...]
    jobs: tuple[GestureSegmentJob, ...]


@dataclass(frozen=True)
class GestureSegmentJob:
    src_path: Path
    dest_path: Path
    gesture: GestAnn
    segment_start_ms: int
    segment_end_ms: int
    start_frame: int
    end_frame: int
    transcript_lines: tuple[str, ...] = ()
    render_transcript: bool = True


@dataclass(frozen=True)
class VideoJob:
    src_path: Path
    dest_path: Path


def format_timestamp(frame_idx: int, fps: float) -> str:
    """Format a frame index as ``HH:MM:SS.mmm`` using integer milliseconds."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    total_ms = int(round(frame_idx * 1000.0 / fps))
    return format_timestamp_ms(total_ms)


def format_timestamp_ms(timestamp_ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS.mmm``."""
    hours, rem = divmod(timestamp_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _frame_padding_ms(fps: float, frame_padding: int = FRAME_PADDING) -> int:
    return int(round(frame_padding * 1000.0 / fps))


def _segment_time_bounds(
    gesture: GestAnn,
    fps: float,
    *,
    frame_padding: int = FRAME_PADDING,
) -> tuple[int, int]:
    padding_ms = _frame_padding_ms(fps, frame_padding)
    segment_start_ms = max(0, gesture.start_ms - padding_ms)
    segment_end_ms = gesture.end_ms + padding_ms
    if segment_end_ms <= segment_start_ms:
        segment_end_ms = segment_start_ms + 1
    return segment_start_ms, segment_end_ms


def _segment_frame_bounds(
    segment_start_ms: int,
    segment_end_ms: int,
    fps: float,
    total_frame_count: int,
) -> tuple[int, int]:
    start_frame = max(0, math.floor(segment_start_ms / 1000.0 * fps))
    end_frame = min(
        total_frame_count,
        max(start_frame + 1, math.ceil(segment_end_ms / 1000.0 * fps)),
    )
    return start_frame, end_frame


def _sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip())
    return cleaned or NOT_AVAILABLE


def _gesture_segment_filename(
    video_path: Path,
    *,
    gesture_index: int,
    gesture_type: str,
) -> str:
    sanitized_type = _sanitize_filename_component(gesture_type)
    return (
        f"{video_path.stem}_{gesture_index}_{sanitized_type}"
        f"{video_path.suffix.lower()}"
    )


def _video_metadata_filename(video_path: Path) -> str:
    return f"{video_path.stem}_metadata.json"


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


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


def _gesture_from_annotation(
    annotation: GestureAnnotation,
    *,
    gesture_index: int,
    annotation_id: str | None,
) -> GestAnn:
    return GestAnn(
        gesture_index=gesture_index,
        annotation_id=annotation_id,
        start_ms=annotation.start_ms,
        end_ms=annotation.end_ms,
        duration_ms=annotation.duration_ms,
        gesture_type=annotation.gesture_type or NOT_AVAILABLE,
        marker=annotation.marker,
        task=annotation.task,
        function=annotation.function,
        left_hand_label=annotation.left_hand_label,
        left_hand_shape=annotation.left_hand_shape,
        right_hand_label=annotation.right_hand_label,
        right_hand_shape=annotation.right_hand_shape,
        raw_tier_values=dict(annotation.raw_tier_values),
    )


def _resolve_video_paths(src_video_filename: str | None) -> list[Path]:
    """Return source video paths to process."""
    if src_video_filename:
        candidate = Path(src_video_filename)
        if candidate.is_file():
            return [candidate.resolve()]

        video_path = SRC_VIDEO_DIR / candidate.name
        if video_path.is_file():
            return [video_path.resolve()]

        raise FileNotFoundError(f"Video file not found: {candidate}")

    if not SRC_VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"Source video directory not found: {SRC_VIDEO_DIR}")

    video_paths = sorted(
        path.resolve()
        for path in SRC_VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError(
            f"No video files found in {SRC_VIDEO_DIR} "
            f"with extensions {sorted(VIDEO_EXTENSIONS)}"
        )
    return video_paths


def _scan_eaf_files(annotation_dir: Path) -> list[Path]:
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Gesture annotation directory not found: {annotation_dir}"
        )
    eaf_paths = sorted(
        path.resolve()
        for path in annotation_dir.rglob("*.eaf")
        if path.is_file()
    )
    if not eaf_paths:
        raise FileNotFoundError(f"No .eaf files found under {annotation_dir}")
    return eaf_paths


def _build_annotation_index(
    annotation_paths: list[Path],
) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for annotation_path in annotation_paths:
        annotation_key = AphasiaDataset._parse_annotation_key_from_filename(
            annotation_path
        )
        if annotation_key is None:
            annotation_key = AphasiaDataset._parse_annotation_key_from_eaf_media(
                annotation_path
            )
        if annotation_key is None:
            continue

        key = (
            annotation_key.subject_id.lower(),
            annotation_key.day.lower(),
        )
        index.setdefault(key, []).append(annotation_path)

    for key in index:
        index[key] = sorted(index[key], key=lambda path: path.name.lower())
    return index


def _find_eaf_for_video(annotation_dir: Path, video_path: Path) -> Path:
    video_key = AphasiaDataset._parse_video_key(video_path)
    if video_key is None:
        raise ValueError(
            f"Could not parse subject/day from video filename: {video_path.name}"
        )

    annotation_index = _build_annotation_index(_scan_eaf_files(annotation_dir))
    pair_key = (
        video_key.subject_id.lower(),
        video_key.day.lower(),
    )
    candidates = annotation_index.get(pair_key, [])
    if not candidates:
        raise FileNotFoundError(
            f"No EAF annotation matched video {video_path.name} "
            f"(subject={video_key.subject_id}, day={video_key.day}) "
            f"in {annotation_dir}"
        )
    if len(candidates) > 1:
        preferred = [
            path
            for path in candidates
            if "gesturecoded" in path.stem.lower()
        ]
        if len(preferred) == 1:
            return preferred[0]
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple EAF files matched video {video_path.name}: {candidate_names}"
        )
    return candidates[0]


def load_gest_anns(
    annotation_dir: Path,
    video_path: Path,
) -> tuple[Path, list[GestAnn]]:
    """Parse gesture annotations and per-video indices from the EAF for ``video_path``."""
    eaf_path = _find_eaf_for_video(annotation_dir, video_path)
    parsed: list[GestureAnnotation] = AphasiaDataset.parse_eaf_gesture_annotations(
        eaf_path,
        video_path,
    )
    annotation_ids = _parse_marker_gesture_annotation_ids(eaf_path)
    gestures: list[GestAnn] = []
    for gesture_index, annotation in enumerate(
        sorted(parsed, key=lambda item: (item.start_ms, item.end_ms)),
        start=1,
    ):
        gestures.append(
            _gesture_from_annotation(
                annotation,
                gesture_index=gesture_index,
                annotation_id=annotation_ids.get(
                    (annotation.start_ms, annotation.end_ms)
                ),
            )
        )
    return eaf_path, gestures


def _scan_cha_files(cha_dir: Path) -> list[Path]:
    if not cha_dir.is_dir():
        raise FileNotFoundError(f"CHA transcript directory not found: {cha_dir}")
    cha_paths = sorted(
        path.resolve()
        for path in cha_dir.rglob("*.cha")
        if path.is_file()
    )
    if not cha_paths:
        raise FileNotFoundError(f"No .cha files found under {cha_dir}")
    return cha_paths


def _find_cha_for_video(cha_dir: Path, video_path: Path) -> Path:
    video_key = AphasiaDataset._parse_video_key(video_path)
    if video_key is None:
        raise ValueError(
            f"Could not parse subject/day from video filename: {video_path.name}"
        )

    annotation_index = _build_annotation_index(_scan_cha_files(cha_dir))
    pair_key = (
        video_key.subject_id.lower(),
        video_key.day.lower(),
    )
    candidates = annotation_index.get(pair_key, [])
    if not candidates:
        raise FileNotFoundError(
            f"No CHA transcript matched video {video_path.name} "
            f"(subject={video_key.subject_id}, day={video_key.day}) "
            f"in {cha_dir}"
        )
    if len(candidates) > 1:
        preferred = [
            path
            for path in candidates
            if "elan" in path.stem.lower() or "gesturecoded" in path.stem.lower()
        ]
        if len(preferred) == 1:
            return preferred[0]
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple CHA files matched video {video_path.name}: {candidate_names}"
        )
    return candidates[0]


def _choose_writer_fourcc(src_path: Path) -> int:
    """Pick a reasonable fourcc based on the source file extension."""
    suffix = src_path.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    if suffix == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def _ffmpeg_runs(ffmpeg_path: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _resolve_ffmpeg() -> str | None:
    """Return a working ffmpeg executable path, or ``None`` if none is found."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        resolved = str(Path(candidate).expanduser())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    add(os.environ.get("FFMPEG_BINARY"))
    add(shutil.which("ffmpeg"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        add(str(Path(conda_prefix) / "bin" / "ffmpeg"))

    for candidate in candidates:
        if Path(candidate).is_file() and _ffmpeg_runs(candidate):
            return candidate
    return None


def _require_ffmpeg() -> str:
    ffmpeg_path = _resolve_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to mux source audio into gesture segment videos, "
            "but no working ffmpeg executable was found. Install ffmpeg (for example "
            "`conda install -c conda-forge ffmpeg`) or set FFMPEG_BINARY to a valid "
            "ffmpeg path."
        )
    return ffmpeg_path


def _measure_text_block(lines: list[str]) -> tuple[int, int, int]:
    max_width = 0
    total_height = 0
    line_heights: list[int] = []
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            TIMESTAMP_FONT,
            TIMESTAMP_FONT_SCALE,
            TIMESTAMP_FONT_THICKNESS,
        )
        line_heights.append(text_height + baseline)
        max_width = max(max_width, text_width)
        total_height += text_height + baseline

    if len(lines) > 1:
        total_height += TIMESTAMP_LINE_GAP_PX * (len(lines) - 1)
    return max_width, total_height, line_heights[0] if line_heights else 0


def _draw_text_block(
    frame,
    lines: list[str],
    *,
    anchor_x: int,
    anchor_y: int,
    align: str,
) -> None:
    if not lines:
        return

    max_width, total_height, _ = _measure_text_block(lines)
    if align == "right":
        x1 = max(0, anchor_x - max_width - 2 * TIMESTAMP_MARGIN_PX)
        x2 = anchor_x
    elif align == "center":
        frame_width = frame.shape[1]
        x1 = max(0, (frame_width - max_width) // 2 - TIMESTAMP_MARGIN_PX)
        x2 = min(frame_width - 1, x1 + max_width + 2 * TIMESTAMP_MARGIN_PX)
    else:
        x1 = anchor_x
        x2 = min(frame.shape[1] - 1, anchor_x + max_width + 2 * TIMESTAMP_MARGIN_PX)

    y1 = anchor_y
    y2 = min(frame.shape[0] - 1, anchor_y + total_height + 2 * TIMESTAMP_MARGIN_PX)
    cv2.rectangle(frame, (x1, y1), (x2, y2), TIMESTAMP_BG_COLOR, thickness=-1)

    text_y = anchor_y + TIMESTAMP_MARGIN_PX
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            TIMESTAMP_FONT,
            TIMESTAMP_FONT_SCALE,
            TIMESTAMP_FONT_THICKNESS,
        )
        if align == "right":
            text_x = x2 - text_width - TIMESTAMP_MARGIN_PX
        elif align == "center":
            text_x = x1 + max(0, (x2 - x1 - text_width) // 2)
        else:
            text_x = x1 + TIMESTAMP_MARGIN_PX
        text_y += text_height
        cv2.putText(
            frame,
            line,
            (text_x, text_y),
            TIMESTAMP_FONT,
            TIMESTAMP_FONT_SCALE,
            TIMESTAMP_TEXT_COLOR,
            TIMESTAMP_FONT_THICKNESS,
            lineType=cv2.LINE_8,
        )
        text_y += baseline + TIMESTAMP_LINE_GAP_PX


def _wrap_display_line(text: str, max_width_px: int) -> list[str]:
    """Wrap one display line to fit within ``max_width_px``."""
    if not text:
        return []

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width, _, _ = _measure_text_block([candidate])
        if candidate_width <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _format_transcript_display_lines(
    utterances: list[dict],
    *,
    max_width_px: int,
) -> list[str]:
    """Format transcript utterances for bottom-center overlay rendering."""
    display_lines: list[str] = []
    for utterance in utterances:
        prefix = f'{utterance["speaker"]}: '
        wrapped = _wrap_display_line(prefix + utterance["text"], max_width_px)
        display_lines.extend(wrapped)
    return display_lines


def _load_segment_transcript_lines(
    cha_path: Path | None,
    *,
    segment_start_ms: int,
    segment_end_ms: int,
    max_width_px: int,
) -> tuple[str, ...]:
    if cha_path is None:
        return ()

    utterances = get_transcript_lines(
        cha_path,
        segment_start_ms,
        segment_end_ms,
        context_before_ms=0,
        context_after_ms=0,
        min_utterances_before=0,
        min_utterances_after=0,
    )
    return tuple(
        _format_transcript_display_lines(
            utterances,
            max_width_px=max_width_px,
        )
    )


class _SegmentFrameOverlay:
    """Render a timestamp and fixed segment transcript on each frame."""

    __slots__ = ("_fps", "_transcript_lines")

    def __init__(
        self,
        fps: float,
        transcript_lines: tuple[str, ...],
    ) -> None:
        self._fps = fps
        self._transcript_lines = transcript_lines

    def draw(self, frame, source_frame_idx: int) -> None:
        timestamp_lines = [format_timestamp(source_frame_idx, self._fps)]
        _draw_text_block(
            frame,
            timestamp_lines,
            anchor_x=frame.shape[1] - TIMESTAMP_MARGIN_PX,
            anchor_y=TIMESTAMP_MARGIN_PX,
            align="right",
        )

        if not self._transcript_lines:
            return

        transcript_lines = list(self._transcript_lines)
        _, total_height, _ = _measure_text_block(transcript_lines)
        anchor_y = max(
            TIMESTAMP_MARGIN_PX,
            frame.shape[0] - total_height - 2 * TIMESTAMP_MARGIN_PX,
        )
        _draw_text_block(
            frame,
            transcript_lines,
            anchor_x=frame.shape[1] // 2,
            anchor_y=anchor_y,
            align="center",
        )


def _open_video_capture(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid frame size ({width}x{height}) for video: {video_path}"
        )
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count ({total_frames}) for video: {video_path}")

    return capture, fps, width, height, total_frames


def _mux_audio_into_segment(
    src_path: Path,
    silent_video_path: Path,
    dest_path: Path,
    *,
    segment_start_ms: int,
    segment_end_ms: int,
    ffmpeg_path: str,
) -> None:
    start_s = segment_start_ms / 1000.0
    duration_s = max(0.001, (segment_end_ms - segment_start_ms) / 1000.0)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video_path),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(src_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(dest_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(
            f"ffmpeg audio mux failed for {dest_path.name}: {detail}"
        )

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = Path(conda_prefix) / "bin" / "ffprobe"
            if candidate.is_file():
                ffprobe_path = str(candidate)

    if ffprobe_path is not None:
        probe = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(dest_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0 or "audio" not in probe.stdout:
            raise RuntimeError(
                f"ffmpeg produced {dest_path.name} without an audio track. "
                f"Source video: {src_path.name}"
            )


def create_gesture_segment_video(job: GestureSegmentJob, *, ffmpeg_path: str) -> None:
    """Write one padded gesture segment with overlays and source audio."""
    capture, fps, width, height, total_frames = _open_video_capture(job.src_path)
    if job.end_frame > total_frames:
        raise RuntimeError(
            f"Segment end frame {job.end_frame} exceeds video frame count "
            f"{total_frames} for {job.src_path.name}"
        )

    job.dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".silent.mp4",
        delete=False,
        dir=job.dest_path.parent,
    ) as temp_handle:
        silent_video_path = Path(temp_handle.name)

    writer = cv2.VideoWriter(
        str(silent_video_path),
        _choose_writer_fourcc(job.src_path),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        silent_video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not open segment video writer: {silent_video_path}"
        )

    overlay = _SegmentFrameOverlay(
        fps,
        job.transcript_lines if job.render_transcript else (),
    )
    frame_count = job.end_frame - job.start_frame
    progress = tqdm(
        total=frame_count,
        desc=job.dest_path.name,
        unit="frame",
        leave=False,
    )

    frames_written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, job.start_frame)
        for source_frame_idx in range(job.start_frame, job.end_frame):
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            overlay.draw(frame, source_frame_idx)
            writer.write(frame)
            frames_written += 1
            progress.update(1)
    finally:
        progress.close()
        writer.release()
        capture.release()

    if frames_written == 0:
        silent_video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No frames written for gesture segment {job.dest_path.name}"
        )

    try:
        _mux_audio_into_segment(
            job.src_path,
            silent_video_path,
            job.dest_path,
            segment_start_ms=job.segment_start_ms,
            segment_end_ms=job.segment_end_ms,
            ffmpeg_path=ffmpeg_path,
        )
    finally:
        silent_video_path.unlink(missing_ok=True)


def create_timestamped_full_video(src_path: Path, dest_path: Path) -> None:
    """Fallback path: write one timestamped full-length video without audio mux."""
    capture, fps, width, height, total_frames = _open_video_capture(src_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dest_path),
        _choose_writer_fourcc(src_path),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {dest_path}")

    overlay = _SegmentFrameOverlay(fps, ())
    frame_idx = 0
    progress = tqdm(
        total=total_frames,
        desc=src_path.name,
        unit="frame",
        mininterval=0.2,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            overlay.draw(frame, frame_idx)
            writer.write(frame)
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        writer.release()
        capture.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames read from video: {src_path}")


def _build_gesture_segment_jobs(
    src_path: Path,
    *,
    eaf_path: Path,
    gest_anns: list[GestAnn],
    cha_path: Path | None,
    dest_dir: Path,
    render_transcript: bool = True,
) -> VideoExportRecord:
    capture, fps, width, height, total_frames = _open_video_capture(src_path)
    capture.release()

    max_width_px = max(1, width - 2 * TIMESTAMP_MARGIN_PX)
    jobs: list[GestureSegmentJob] = []
    for gesture in gest_anns:
        segment_start_ms, segment_end_ms = _segment_time_bounds(gesture, fps)
        start_frame, end_frame = _segment_frame_bounds(
            segment_start_ms,
            segment_end_ms,
            fps,
            total_frames,
        )
        transcript_lines = _load_segment_transcript_lines(
            cha_path,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
            max_width_px=max_width_px,
        )
        jobs.append(
            GestureSegmentJob(
                src_path=src_path,
                dest_path=dest_dir
                / _gesture_segment_filename(
                    src_path,
                    gesture_index=gesture.gesture_index,
                    gesture_type=gesture.gesture_type,
                ),
                gesture=gesture,
                segment_start_ms=segment_start_ms,
                segment_end_ms=segment_end_ms,
                start_frame=start_frame,
                end_frame=end_frame,
                transcript_lines=transcript_lines,
                render_transcript=render_transcript,
            )
        )

    return VideoExportRecord(
        src_path=src_path,
        eaf_path=eaf_path,
        cha_path=cha_path,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        gestures=tuple(gest_anns),
        jobs=tuple(jobs),
    )


def _build_gesture_metadata_entry(
    gesture: GestAnn,
    job: GestureSegmentJob,
    *,
    total_gesture_count: int,
    fps: float,
) -> dict:
    frame_count = job.end_frame - job.start_frame
    return {
        "gesture_index": gesture.gesture_index,
        "annotation_id": gesture.annotation_id,
        "dest_filename": job.dest_path.name,
        "dest_path": str(job.dest_path),
        "start_ms": gesture.start_ms,
        "end_ms": gesture.end_ms,
        "start_timestamp": format_timestamp_ms(gesture.start_ms),
        "end_timestamp": format_timestamp_ms(gesture.end_ms),
        "duration_ms": gesture.duration_ms,
        "gesture_type": gesture.gesture_type,
        "total_gesture_count": total_gesture_count,
        "segment_start_ms": job.segment_start_ms,
        "segment_end_ms": job.segment_end_ms,
        "segment_start_timestamp": format_timestamp_ms(job.segment_start_ms),
        "segment_end_timestamp": format_timestamp_ms(job.segment_end_ms),
        "frame_padding_frames": FRAME_PADDING,
        "frame_padding_ms": _frame_padding_ms(fps),
        "start_frame": job.start_frame,
        "end_frame": job.end_frame,
        "segment_frame_count": frame_count,
        "marker": gesture.marker,
        "task": gesture.task,
        "function": gesture.function,
        "left_hand_label": gesture.left_hand_label,
        "left_hand_shape": gesture.left_hand_shape,
        "right_hand_label": gesture.right_hand_label,
        "right_hand_shape": gesture.right_hand_shape,
        "raw_tier_values": gesture.raw_tier_values,
        "transcript_line_count": len(job.transcript_lines),
        "transcript_lines": list(job.transcript_lines),
        "render_transcript": job.render_transcript,
    }


def _build_video_gesture_metadata(
    export_record: VideoExportRecord,
    *,
    eaf_path: Path,
    render_transcript: bool,
) -> dict:
    total_gesture_count = len(export_record.gestures)
    job_by_index = {
        job.gesture.gesture_index: job for job in export_record.jobs
    }
    return {
        "video_filename": export_record.src_path.name,
        "video_path": str(export_record.src_path),
        "video_stem": export_record.src_path.stem,
        "annotation_filename": eaf_path.name,
        "annotation_path": str(eaf_path),
        "cha_filename": (
            export_record.cha_path.name if export_record.cha_path is not None else None
        ),
        "cha_path": (
            str(export_record.cha_path)
            if export_record.cha_path is not None
            else None
        ),
        "metadata_filename": _video_metadata_filename(export_record.src_path),
        "dest_dir": str(export_record.jobs[0].dest_path.parent)
        if export_record.jobs
        else str(DEST_VIDEO_DIR),
        "total_gesture_count": total_gesture_count,
        "render_transcript": render_transcript,
        "frame_padding_frames": FRAME_PADDING,
        "frame_padding_ms": _frame_padding_ms(export_record.fps),
        "fps": export_record.fps,
        "width": export_record.width,
        "height": export_record.height,
        "total_frames": export_record.total_frames,
        "gestures": [
            _build_gesture_metadata_entry(
                gesture,
                job_by_index[gesture.gesture_index],
                total_gesture_count=total_gesture_count,
                fps=export_record.fps,
            )
            for gesture in export_record.gestures
        ],
    }


def write_video_gesture_metadata(
    export_record: VideoExportRecord,
    *,
    eaf_path: Path,
    dest_dir: Path,
    render_transcript: bool,
) -> Path:
    metadata_path = dest_dir / _video_metadata_filename(export_record.src_path)
    metadata = _build_video_gesture_metadata(
        export_record,
        eaf_path=eaf_path,
        render_transcript=render_transcript,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _process_gesture_segment_job(
    job: GestureSegmentJob,
    *,
    ffmpeg_path: str,
) -> tuple[Path, str | None]:
    try:
        create_gesture_segment_video(job, ffmpeg_path=ffmpeg_path)
        return job.dest_path, None
    except Exception as exc:
        return job.dest_path, str(exc)


def _process_full_video_job(job: VideoJob) -> tuple[Path, str | None]:
    try:
        create_timestamped_full_video(job.src_path, job.dest_path)
        return job.dest_path, None
    except Exception as exc:
        return job.dest_path, str(exc)


def process_videos(
    src_video_filename: str | None = None,
    *,
    src_gest_ann: Path | None = None,
    src_cha: Path | None = None,
    render_transcript: bool = True,
    workers: int = 1,
) -> None:
    """Process source videos into gesture segment clips or full timestamped videos."""
    if render_transcript and src_cha is None and src_gest_ann is not None:
        print(
            "Warning: --render-transcript is enabled but --src_cha was not provided; "
            "no transcript lines will be rendered.",
            file=sys.stderr,
        )

    video_paths = _resolve_video_paths(src_video_filename)

    segment_jobs: list[GestureSegmentJob] = []
    full_video_jobs: list[VideoJob] = []
    video_export_records: list[VideoExportRecord] = []

    for src_path in video_paths:
        if src_gest_ann is not None:
            eaf_path, gest_anns = load_gest_anns(src_gest_ann, src_path)
            cha_path = (
                _find_cha_for_video(src_cha, src_path) if src_cha is not None else None
            )
            tqdm.write(
                f"Prepared {len(gest_anns)} gesture segment(s) for {src_path.name}"
            )
            if cha_path is not None:
                tqdm.write(f"Using CHA transcript {cha_path.name} for {src_path.name}")
            export_record = _build_gesture_segment_jobs(
                src_path,
                eaf_path=eaf_path,
                gest_anns=gest_anns,
                cha_path=cha_path,
                dest_dir=DEST_VIDEO_DIR,
                render_transcript=render_transcript,
            )
            video_export_records.append(export_record)
            segment_jobs.extend(export_record.jobs)
        else:
            full_video_jobs.append(
                VideoJob(
                    src_path=src_path,
                    dest_path=DEST_VIDEO_DIR / src_path.name,
                )
            )

    ffmpeg_path: str | None = None
    if segment_jobs:
        ffmpeg_path = _require_ffmpeg()
        print(
            f"Processing {len(segment_jobs)} gesture segment(s) via "
            "opencv + ffmpeg audio mux"
        )
        print(f"  ffmpeg:     {ffmpeg_path}")
    if full_video_jobs:
        print(f"Processing {len(full_video_jobs)} full video(s)")
    print(f"  source dir: {SRC_VIDEO_DIR}")
    print(f"  dest dir:   {DEST_VIDEO_DIR}")
    if src_gest_ann is not None:
        print(f"  gest ann:   {src_gest_ann}")
    if src_cha is not None:
        print(f"  cha dir:    {src_cha}")
    if src_gest_ann is not None:
        print(f"  transcript: {'rendered on frames' if render_transcript else 'not rendered'}")
    if workers > 1:
        print(f"  workers:    {workers}")

    failures: list[tuple[Path, str]] = []

    if workers > 1 and len(segment_jobs) + len(full_video_jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for job in segment_jobs:
                futures[
                    executor.submit(
                        _process_gesture_segment_job,
                        job,
                        ffmpeg_path=ffmpeg_path,
                    )
                ] = job
            for job in full_video_jobs:
                futures[executor.submit(_process_full_video_job, job)] = job

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Clips",
                unit="job",
            ):
                output_path, error = future.result()
                if error is not None:
                    failures.append((output_path, error))
                    tqdm.write(f"FAILED {output_path.name}: {error}")
    else:
        iterable = segment_jobs + full_video_jobs
        for job in iterable:
            if isinstance(job, GestureSegmentJob):
                output_path, error = _process_gesture_segment_job(
                    job,
                    ffmpeg_path=ffmpeg_path,
                )
            else:
                output_path, error = _process_full_video_job(job)
            if error is not None:
                failures.append((output_path, error))
                tqdm.write(f"FAILED {output_path.name}: {error}")

    if failures:
        print(
            f"\nCompleted with {len(failures)} failure(s).",
            file=sys.stderr,
        )
        for output_path, message in failures:
            print(f"  {output_path}: {message}", file=sys.stderr)
        raise SystemExit(1)

    metadata_paths: list[Path] = []
    for export_record in video_export_records:
        metadata_path = write_video_gesture_metadata(
            export_record,
            eaf_path=export_record.eaf_path,
            dest_dir=DEST_VIDEO_DIR,
            render_transcript=render_transcript,
        )
        metadata_paths.append(metadata_path)

    total_outputs = len(segment_jobs) + len(full_video_jobs)
    print(f"\nSuccessfully wrote {total_outputs} video(s) to {DEST_VIDEO_DIR}")
    if metadata_paths:
        print(f"Successfully wrote {len(metadata_paths)} metadata file(s):")
        for metadata_path in metadata_paths:
            print(f"  {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export padded gesture segment videos with timestamps and transcript "
            "overlays to DEST_VIDEO_DIR."
        )
    )
    parser.add_argument(
        "--src_video_filename",
        type=str,
        default=None,
        help=(
            "Process only this video (filename in SRC_VIDEO_DIR or a full path). "
            "If omitted, all videos in SRC_VIDEO_DIR are processed."
        ),
    )
    parser.add_argument(
        "--src_gest_ann",
        type=str,
        default=None,
        help=(
            "Directory containing EAF gesture annotation files. When provided, "
            "one padded clip is written per annotated gesture."
        ),
    )
    parser.add_argument(
        "--src_cha",
        type=str,
        default=None,
        help=(
            "Directory containing CHAT ``.cha`` transcript files. Transcript lines "
            "for the padded segment window are loaded from these files when "
            "available."
        ),
    )
    parser.add_argument(
        "--render-transcript",
        dest="render_transcript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Burn transcript lines into each exported frame at the bottom center. "
            "Requires --src_cha. Use --no-render-transcript to export clips without "
            "transcript overlays while still recording transcript text in metadata."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers when processing multiple clips.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    src_gest_ann = (
        Path(args.src_gest_ann).expanduser().resolve()
        if args.src_gest_ann
        else None
    )
    src_cha = (
        Path(args.src_cha).expanduser().resolve()
        if args.src_cha
        else None
    )
    process_videos(
        src_video_filename=args.src_video_filename,
        src_gest_ann=src_gest_ann,
        src_cha=src_cha,
        render_transcript=args.render_transcript,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
