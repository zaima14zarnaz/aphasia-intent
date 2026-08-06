"""
Extract negative (non-gesture) video segments from gaps between annotated gestures.

Samples intervals from parts of each video that do not overlap any valid
positive gesture annotation, using positive segment frame statistics to choose
negative segment lengths.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict
from dataclasses import dataclass
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

POSITIVE_SEGMENTS_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments"
)
OUTPUT_ROOT = (
    "/data/quantization/zaima/aphasiabank/neural2/control/neg_gest_segments"
)

MAX_ATTEMPTS_PER_VIDEO = 500


@dataclass(frozen=True)
class TimeIntervalMs:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class PositiveInterval:
    video_path: Path
    annotation_path: Path | None
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class NegativeSegmentPlan:
    video_path: Path
    annotation_path: Path | None
    video_filename: str
    annotation_filename: str | None
    start_ms: int
    end_ms: int
    duration_ms: int
    negative_video_index: int
    length_frames: int
    fps: float
    gap_start_ms: int
    gap_end_ms: int


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


def _annotation_line_value(value) -> str:
    """Format one annotation.txt field value like extract_gest_segments.py."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def write_negative_annotation_metadata(
    annotation: dict,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> None:
    """Write negative-segment metadata for one segment to annotation.txt."""
    output_path = Path(output_dir)

    lines = [
        f"video_path: {_annotation_line_value(annotation.get('video_path'))}",
        f"annotation_path: {_annotation_line_value(annotation.get('annotation_path'))}",
        f"video_filename: {_annotation_line_value(annotation.get('video_filename'))}",
        f"annotation_filename: {_annotation_line_value(annotation.get('annotation_filename'))}",
        f"start_ms: {_annotation_line_value(annotation.get('start_ms'))}",
        f"end_ms: {_annotation_line_value(annotation.get('end_ms'))}",
        f"duration_ms: {_annotation_line_value(annotation.get('duration_ms'))}",
        f"negative_video_index: {_annotation_line_value(annotation.get('negative_video_index'))}",
        f"marker: {_annotation_line_value(annotation.get('marker'))}",
        f"task: {_annotation_line_value(annotation.get('task'))}",
        f"gesture_type: {_annotation_line_value(annotation.get('gesture_type'))}",
        f"function: {_annotation_line_value(annotation.get('function'))}",
        f"left_hand_label: {_annotation_line_value(annotation.get('left_hand_label'))}",
        f"left_hand_shape: {_annotation_line_value(annotation.get('left_hand_shape'))}",
        f"right_hand_label: {_annotation_line_value(annotation.get('right_hand_label'))}",
        f"right_hand_shape: {_annotation_line_value(annotation.get('right_hand_shape'))}",
        f"source: {_annotation_line_value(annotation.get('source'))}",
        f"positive_overlap: {_annotation_line_value(annotation.get('positive_overlap'))}",
        f"length_frames: {_annotation_line_value(annotation.get('length_frames'))}",
        f"fps: {_annotation_line_value(annotation.get('fps'))}",
        f"gap_start_ms: {_annotation_line_value(annotation.get('gap_start_ms'))}",
        f"gap_end_ms: {_annotation_line_value(annotation.get('gap_end_ms'))}",
    ]
    annotation_text = "\n".join(lines) + "\n"

    if dry_run:
        print(f"[dry-run] {output_path / 'annotation.txt'}")
        print(annotation_text, end="")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "annotation.txt").write_text(annotation_text, encoding="utf-8")


def build_positive_intervals_by_video(
    dataset,
) -> dict[str, list[PositiveInterval]]:
    """Group positive gesture intervals by video filename."""
    intervals_by_video: dict[str, list[PositiveInterval]] = defaultdict(list)

    for sample in dataset.samples:
        intervals_by_video[sample.video_path.name].append(
            PositiveInterval(
                video_path=sample.video_path,
                annotation_path=sample.annotation_path,
                start_ms=sample.start_ms,
                end_ms=sample.end_ms,
            )
        )

    for video_filename in intervals_by_video:
        intervals_by_video[video_filename].sort(key=lambda item: item.start_ms)

    return dict(intervals_by_video)


def merge_positive_intervals(
    intervals: list[PositiveInterval],
) -> list[TimeIntervalMs]:
    """Merge overlapping or touching positive intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: item.start_ms)
    merged: list[TimeIntervalMs] = [
        TimeIntervalMs(
            start_ms=sorted_intervals[0].start_ms,
            end_ms=sorted_intervals[0].end_ms,
        )
    ]

    for interval in sorted_intervals[1:]:
        current = merged[-1]
        if interval.start_ms <= current.end_ms:
            merged[-1] = TimeIntervalMs(
                start_ms=current.start_ms,
                end_ms=max(current.end_ms, interval.end_ms),
            )
        else:
            merged.append(
                TimeIntervalMs(
                    start_ms=interval.start_ms,
                    end_ms=interval.end_ms,
                )
            )

    return merged


def merge_time_intervals(
    intervals: list[TimeIntervalMs],
) -> list[TimeIntervalMs]:
    """Merge overlapping or touching time intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: item.start_ms)
    merged: list[TimeIntervalMs] = [sorted_intervals[0]]

    for interval in sorted_intervals[1:]:
        current = merged[-1]
        if interval.start_ms <= current.end_ms:
            merged[-1] = TimeIntervalMs(
                start_ms=current.start_ms,
                end_ms=max(current.end_ms, interval.end_ms),
            )
        else:
            merged.append(interval)

    return merged


def apply_gap_margin_to_merged_intervals(
    merged_positive_intervals: list[TimeIntervalMs],
    video_duration_ms: float,
    gap_margin_ms: float,
) -> list[TimeIntervalMs]:
    """Expand positive intervals by a margin to keep negatives away from gestures."""
    if gap_margin_ms <= 0:
        return merged_positive_intervals

    margin_ms = int(round(gap_margin_ms))
    expanded = [
        TimeIntervalMs(
            start_ms=max(0, interval.start_ms - margin_ms),
            end_ms=min(int(round(video_duration_ms)), interval.end_ms + margin_ms),
        )
        for interval in merged_positive_intervals
    ]
    return merge_time_intervals(expanded)


def get_video_duration_ms(video_path: Path) -> float:
    """Return video duration in milliseconds."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    return float(total_frames) / float(fps) * 1000.0


def get_video_fps(video_path: Path) -> float:
    """Return video frames per second."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    return float(fps)


def collect_positive_segment_frame_counts(
    positive_segments_root: str | Path,
) -> list[int]:
    """Count numeric PNG frames in each positive gest_* segment directory."""
    positive_root = Path(positive_segments_root)
    frame_counts: list[int] = []

    if not positive_root.is_dir():
        return frame_counts

    for video_dir in sorted(positive_root.iterdir()):
        if not video_dir.is_dir():
            continue

        for segment_dir in sorted(video_dir.glob("gest_*")):
            if not segment_dir.is_dir():
                continue

            frame_counts.append(len(_list_segment_frame_paths(segment_dir)))

    return frame_counts


def compute_negative_length_range(
    frame_counts: list[int],
) -> tuple[float, float, int, int]:
    """Derive negative segment length bounds from positive frame count statistics."""
    if not frame_counts:
        raise RuntimeError(
            "No positive gesture segment frame counts found. "
            "Run extract_gest_segments.py first or check --positive-segments-root."
        )

    mean_frames = float(np.mean(frame_counts))
    std_frames = float(np.std(frame_counts))
    min_frames = int(round(mean_frames - 3 * std_frames))
    max_frames = int(round(mean_frames + 3 * std_frames))
    min_frames = max(1, min_frames)
    max_frames = max(min_frames, max_frames)
    return mean_frames, std_frames, min_frames, max_frames


def find_non_gesture_gaps(
    merged_positive_intervals: list[TimeIntervalMs],
    video_duration_ms: float,
    min_gap_ms: float,
) -> list[TimeIntervalMs]:
    """Return non-gesture gaps long enough for at least one negative segment."""
    gaps: list[TimeIntervalMs] = []

    if not merged_positive_intervals:
        if video_duration_ms >= min_gap_ms:
            gaps.append(TimeIntervalMs(start_ms=0, end_ms=int(round(video_duration_ms))))
        return gaps

    first_start = merged_positive_intervals[0].start_ms
    if first_start >= min_gap_ms:
        gaps.append(TimeIntervalMs(start_ms=0, end_ms=first_start))

    for index in range(len(merged_positive_intervals) - 1):
        gap_start = merged_positive_intervals[index].end_ms
        gap_end = merged_positive_intervals[index + 1].start_ms
        gap_duration = gap_end - gap_start
        if gap_duration >= min_gap_ms:
            gaps.append(TimeIntervalMs(start_ms=gap_start, end_ms=gap_end))

    last_end = merged_positive_intervals[-1].end_ms
    trailing_duration = video_duration_ms - last_end
    if trailing_duration >= min_gap_ms:
        gaps.append(
            TimeIntervalMs(
                start_ms=last_end,
                end_ms=int(round(video_duration_ms)),
            )
        )

    return gaps


def intervals_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    """Return True when two half-open intervals [start, end) overlap."""
    return start_a < end_b and start_b < end_a


def overlaps_any_positive_interval(
    start_ms: int,
    end_ms: int,
    merged_positive_intervals: list[TimeIntervalMs],
) -> bool:
    return any(
        intervals_overlap(start_ms, end_ms, interval.start_ms, interval.end_ms)
        for interval in merged_positive_intervals
    )


def sample_length_frames(
    rng: np.random.Generator,
    min_frames: int,
    max_frames: int,
) -> int:
    """Sample a random integer frame count for one negative segment."""
    return int(rng.integers(min_frames, max_frames + 1))


def plan_negative_segments_for_video(
    *,
    video_filename: str,
    positive_intervals: list[PositiveInterval],
    merged_positive_intervals: list[TimeIntervalMs],
    video_duration_ms: float,
    fps: float,
    min_frames: int,
    max_frames: int,
    negative_per_positive: float,
    gap_margin_ms: float,
    rng: np.random.Generator,
    log,
) -> tuple[list[NegativeSegmentPlan], int]:
    """Plan negative segments for one video."""
    num_positive = len(positive_intervals)
    target_negative = round(num_positive * negative_per_positive)

    if target_negative <= 0:
        return [], 0

    min_duration_ms = min_frames / fps * 1000.0
    excluded_intervals = apply_gap_margin_to_merged_intervals(
        merged_positive_intervals,
        video_duration_ms,
        gap_margin_ms,
    )
    gaps = find_non_gesture_gaps(
        excluded_intervals,
        video_duration_ms,
        min_duration_ms,
    )

    if not gaps:
        return [], target_negative

    video_path = positive_intervals[0].video_path
    annotation_path = positive_intervals[0].annotation_path
    annotation_filename = (
        annotation_path.name if annotation_path is not None else None
    )

    selected_negative_intervals: list[TimeIntervalMs] = []
    planned_segments: list[NegativeSegmentPlan] = []

    for attempt in range(MAX_ATTEMPTS_PER_VIDEO):
        if len(planned_segments) >= target_negative:
            break

        length_frames = sample_length_frames(rng, min_frames, max_frames)
        duration_ms = int(round(length_frames / fps * 1000.0))

        valid_gaps = [
            gap
            for gap in gaps
            if (gap.end_ms - gap.start_ms) >= duration_ms
        ]
        if not valid_gaps:
            continue

        gap = rng.choice(valid_gaps)
        latest_start = gap.end_ms - duration_ms
        if latest_start < gap.start_ms:
            continue

        start_ms = int(rng.integers(gap.start_ms, latest_start + 1))
        end_ms = start_ms + duration_ms

        if overlaps_any_positive_interval(
            start_ms,
            end_ms,
            excluded_intervals,
        ):
            continue

        if any(
            intervals_overlap(
                start_ms,
                end_ms,
                negative.start_ms,
                negative.end_ms,
            )
            for negative in selected_negative_intervals
        ):
            continue

        negative_index = len(planned_segments)
        selected_negative_intervals.append(
            TimeIntervalMs(start_ms=start_ms, end_ms=end_ms)
        )
        planned_segments.append(
            NegativeSegmentPlan(
                video_path=video_path,
                annotation_path=annotation_path,
                video_filename=video_filename,
                annotation_filename=annotation_filename,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                negative_video_index=negative_index,
                length_frames=length_frames,
                fps=fps,
                gap_start_ms=gap.start_ms,
                gap_end_ms=gap.end_ms,
            )
        )

    if len(planned_segments) < target_negative:
        log(
            f"Warning: only planned {len(planned_segments)}/{target_negative} "
            f"negative segments for {video_filename} after "
            f"{MAX_ATTEMPTS_PER_VIDEO} attempts."
        )

    return planned_segments, target_negative


def extract_video_clip_frames(
    video_path: Path,
    start_ms: int,
    length_frames: int,
) -> torch.Tensor:
    """
    Read a contiguous clip from a video file.

    Returns:
        RGB tensor with shape [T, C, H, W], float values in [0, 1].
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    start_frame = round(start_ms / 1000.0 * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    for _ in range(length_frames):
        success, frame_bgr = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    capture.release()

    if not frames:
        raise RuntimeError(
            f"No frames read from {video_path} at start_ms={start_ms}"
        )

    stacked = np.stack(frames, axis=0)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float().div(255.0)


def build_negative_output_dir(
    plan: NegativeSegmentPlan,
    output_root: str | Path,
) -> Path:
    return (
        Path(output_root)
        / plan.video_filename
        / f"neg_{plan.negative_video_index}"
    )


def _segment_has_existing_frames(output_dir: Path) -> bool:
    return output_dir.exists() and bool(_list_segment_frame_paths(output_dir))


def build_negative_annotation(plan: NegativeSegmentPlan) -> dict:
    return {
        "video_path": str(plan.video_path),
        "annotation_path": (
            str(plan.annotation_path) if plan.annotation_path is not None else None
        ),
        "video_filename": plan.video_filename,
        "annotation_filename": plan.annotation_filename,
        "start_ms": plan.start_ms,
        "end_ms": plan.end_ms,
        "duration_ms": plan.duration_ms,
        "negative_video_index": plan.negative_video_index,
        "marker": "negative",
        "task": "negative",
        "gesture_type": None,
        "function": None,
        "left_hand_label": None,
        "left_hand_shape": None,
        "right_hand_label": None,
        "right_hand_shape": None,
        "source": "non_gesture_gap",
        "positive_overlap": False,
        "length_frames": plan.length_frames,
        "fps": plan.fps,
        "gap_start_ms": plan.gap_start_ms,
        "gap_end_ms": plan.gap_end_ms,
    }


def print_negative_extraction_validation_report(output_root: str | Path) -> None:
    """Scan output root on disk and print a post-extraction validation summary."""
    output_path = Path(output_root)
    if not output_path.is_dir():
        print()
        print("Validation report:")
        print(f"  Output root does not exist: {output_path}")
        return

    video_dirs = sorted(
        path for path in output_path.iterdir() if path.is_dir()
    )

    segment_dirs: list[Path] = []
    segment_frame_counts: list[int] = []
    total_png_frames = 0
    annotation_file_count = 0
    segments_with_frames_missing_annotation: list[Path] = []
    example_output_path: Path | None = None

    for video_dir in video_dirs:
        for segment_dir in sorted(video_dir.glob("neg_*")):
            if not segment_dir.is_dir():
                continue

            segment_dirs.append(segment_dir)
            if example_output_path is None:
                example_output_path = segment_dir

            frame_count = len(_list_segment_frame_paths(segment_dir))
            has_annotation = (segment_dir / "annotation.txt").is_file()

            if has_annotation:
                annotation_file_count += 1

            if frame_count > 0:
                segment_frame_counts.append(frame_count)
                total_png_frames += frame_count
                if not has_annotation:
                    segments_with_frames_missing_annotation.append(segment_dir)

    print()
    print("Validation report:")
    print(f"  Video directories: {len(video_dirs)}")
    print(f"  neg_* segment directories: {len(segment_dirs)}")
    print(f"  Total PNG frames: {total_png_frames}")

    if segment_frame_counts:
        counts = np.array(segment_frame_counts, dtype=np.float64)
        print(
            "  Frames per negative segment mean/std/min/max: "
            f"{counts.mean():.2f}/{counts.std():.2f}/"
            f"{int(counts.min())}/{int(counts.max())}"
        )
    else:
        print("  Frames per negative segment mean/std/min/max: n/a (no frames found)")

    print(f"  annotation.txt files: {annotation_file_count}")

    if example_output_path is not None:
        print(f"  Example output path: {example_output_path}")
    else:
        print(f"  Example output path: (none under {output_path})")

    if segments_with_frames_missing_annotation:
        print(
            f"  Warning: {len(segments_with_frames_missing_annotation)} neg_* "
            "directories have frames but no annotation.txt:"
        )
        for missing_dir in segments_with_frames_missing_annotation[:10]:
            print(f"    {missing_dir}")
        remaining = len(segments_with_frames_missing_annotation) - 10
        if remaining > 0:
            print(f"    ... and {remaining} more")
    elif segment_frame_counts:
        print("  All neg_* directories with frames have annotation.txt.")


def print_dry_run_extraction_line(
    plan: NegativeSegmentPlan,
    output_dir: Path,
) -> None:
    """Print one planned negative segment without writing output files."""
    print(
        "[dry-run] "
        f"video={plan.video_filename} "
        f"neg={plan.negative_video_index} "
        f"start_ms={plan.start_ms} "
        f"end_ms={plan.end_ms} "
        f"length_frames={plan.length_frames} "
        f"output_dir={output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract negative (non-gesture) video segments from gaps between "
            "annotated gestures."
        )
    )
    parser.add_argument(
        "--video-dir",
        default=VIDEO_DIR,
        help=f"Directory containing source videos (default: {VIDEO_DIR})",
    )
    parser.add_argument(
        "--annotation-dir",
        action="append",
        dest="annotation_dirs",
        default=None,
        help=(
            "Annotation directory to scan. Repeat for multiple dirs. "
            f"Default: {ANNOTATION_DIRS}"
        ),
    )
    parser.add_argument(
        "--positive-segments-root",
        default=POSITIVE_SEGMENTS_ROOT,
        help=(
            "Root directory of extracted positive gest_* segments used for "
            f"length statistics (default: {POSITIVE_SEGMENTS_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help=f"Output root for negative neg_* segments (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--negative-per-positive",
        type=float,
        default=1.0,
        help="Target number of negative segments per positive gesture (default: 1.0)",
    )
    parser.add_argument(
        "--gap-margin-ms",
        type=float,
        default=250.0,
        help=(
            "Keep negative segments at least this many milliseconds away from "
            "positive gestures (default: 250.0)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing negative segment directories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative segment planning (default: 42)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit planning/extraction to the first N videos (default: all)",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=None,
        help="Limit total planned/extracted negative segments (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan segments and print what would be extracted without saving "
            "frames or annotation.txt"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dirs = (
        args.annotation_dirs
        if args.annotation_dirs is not None
        else list(ANNOTATION_DIRS)
    )

    output_root = Path(args.output_root)
    positive_root = Path(args.positive_segments_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    dataset = AphasiaGestureEAFDataset(
        video_dir=args.video_dir,
        annotation_dir=annotation_dirs,
        recursive=False,
        strict=False,
        allow_multiple_annotations_per_video=True,
        num_frames=None,
        frame_size=None,
        rgb=True,
        return_tensor=True,
    )

    frame_counts = collect_positive_segment_frame_counts(positive_root)
    mean_frames, std_frames, min_frames, max_frames = compute_negative_length_range(
        frame_counts
    )
    print(f"Positive segment directories counted: {len(frame_counts)}")
    print(f"Positive frame count mean: {mean_frames:.2f}")
    print(f"Positive frame count std: {std_frames:.2f}")
    print(f"Negative length min frames: {min_frames}")
    print(f"Negative length max frames: {max_frames}")
    print()

    positive_intervals_by_video = build_positive_intervals_by_video(dataset)
    rng = np.random.default_rng(args.seed)

    def _log(message: str) -> None:
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    all_plans: list[NegativeSegmentPlan] = []
    total_target_negatives = 0
    videos_with_no_valid_gaps = 0
    videos_processed = 0

    for video_filename, positive_intervals in sorted(
        positive_intervals_by_video.items()
    ):
        if args.max_videos is not None and videos_processed >= args.max_videos:
            break

        merged_positive_intervals = merge_positive_intervals(positive_intervals)

        try:
            video_duration_ms = get_video_duration_ms(positive_intervals[0].video_path)
            fps = get_video_fps(positive_intervals[0].video_path)
        except Exception as exc:
            _log(f"Failed to read video metadata for {video_filename}: {exc}")
            continue

        videos_processed += 1

        plans, target_negative = plan_negative_segments_for_video(
            video_filename=video_filename,
            positive_intervals=positive_intervals,
            merged_positive_intervals=merged_positive_intervals,
            video_duration_ms=video_duration_ms,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            negative_per_positive=args.negative_per_positive,
            gap_margin_ms=args.gap_margin_ms,
            rng=rng,
            log=_log,
        )

        total_target_negatives += target_negative
        if target_negative > 0 and not plans:
            videos_with_no_valid_gaps += 1

        all_plans.extend(plans)

        if args.max_negatives is not None and len(all_plans) >= args.max_negatives:
            all_plans = all_plans[: args.max_negatives]
            break

    if hasattr(dataset, "pairs"):
        print(f"Matched video/EAF pairs: {len(dataset.pairs)}")
    print(f"Positive gesture segments: {len(dataset)}")
    print(f"Videos with gestures: {len(positive_intervals_by_video)}")
    print(
        "Positive frame count mean/std/min/max used for negative length sampling: "
        f"{mean_frames:.2f}/{std_frames:.2f}/{min_frames}/{max_frames}"
    )
    print(f"Output root: {output_root}")
    print(f"Overwrite existing segments: {args.overwrite}")
    print(f"Dry run: {args.dry_run}")
    print(f"Gap margin ms: {args.gap_margin_ms}")
    print(f"Negative per positive: {args.negative_per_positive}")
    print(f"Total target negatives: {total_target_negatives}")
    print(f"Planned negatives: {len(all_plans)}")
    print()

    saved_segments = 0
    skipped_existing_segments = 0
    failed_segments = 0
    dry_run_segments = 0

    progress_bar = (
        tqdm(all_plans, desc="Extracting negative segments", unit="segment")
        if tqdm is not None and not args.dry_run
        else all_plans
    )

    for plan in progress_bar:
        output_dir = build_negative_output_dir(plan, output_root)

        try:
            if args.dry_run:
                print_dry_run_extraction_line(plan, output_dir)
                dry_run_segments += 1
                continue

            if _segment_has_existing_frames(output_dir) and not args.overwrite:
                skipped_existing_segments += 1
                continue

            frames = extract_video_clip_frames(
                plan.video_path,
                plan.start_ms,
                plan.length_frames,
            )
            save_frames_as_png(frames, output_dir, overwrite=args.overwrite)
            write_negative_annotation_metadata(
                build_negative_annotation(plan),
                output_dir,
                dry_run=False,
            )
            saved_segments += 1

        except Exception as exc:
            failed_segments += 1
            _log(
                f"Failed negative segment video={plan.video_filename} "
                f"neg_{plan.negative_video_index} "
                f"start_ms={plan.start_ms} end_ms={plan.end_ms}: {exc}"
            )

        if tqdm is not None and not args.dry_run:
            progress_bar.set_postfix(
                saved=saved_segments,
                skipped=skipped_existing_segments,
                failed=failed_segments,
                refresh=False,
            )

    print()
    print("Extraction summary:")
    print(f"  Total target negatives: {total_target_negatives}")
    print(f"  Planned negatives: {len(all_plans)}")
    if args.dry_run:
        print(f"  Dry-run segments: {dry_run_segments}")
    else:
        print(f"  Saved negatives: {saved_segments}")
        print(f"  Skipped existing negatives: {skipped_existing_segments}")
        print(f"  Failed negatives: {failed_segments}")
    print(f"  Videos with no valid negative gaps: {videos_with_no_valid_gaps}")
    print(f"  Output root: {output_root}")

    if not args.dry_run:
        print_negative_extraction_validation_report(output_root)


if __name__ == "__main__":
    main()
