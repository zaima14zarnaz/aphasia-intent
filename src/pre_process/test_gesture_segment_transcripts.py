"""Focused tests for transcript-aware AphasiaGestureSegmentDataset behavior."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_TEST_DIR = Path(__file__).resolve().parent
_DATASET_PROCESSING_DIR = _TEST_DIR.parent / "dataset_processing"
if str(_DATASET_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_PROCESSING_DIR))
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from dataloder import (  # noqa: E402
    AphasiaGestureSegmentDataset,
    NEGATIVE_SEGMENT_LABEL,
    POSITIVE_SEGMENT_LABEL,
    _build_transcript_chunks,
    _build_transcript_lookup,
    _parse_cha_media_reference,
    _scan_cha_transcript_files,
)
from collate import gesture_segment_collate_fn  # noqa: E402


def _write_min_landmarks(path: Path) -> None:
    joints = [[0.1, 0.2, 0.3, 1.0] for _ in range(33)]
    payload = {
        "extractor": "mediapipe",
        "frames": [{"landmarks": joints}, {"landmarks": joints}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_annotation(
    path: Path,
    *,
    start_ms: int = 113000,
    end_ms: int = 115000,
) -> None:
    lines = [
        "video_filename: 65-1.mp4",
        f"start_ms: {start_ms}",
        f"end_ms: {end_ms}",
        "duration_ms: 2000",
        "task: test",
        "gesture_type: beat",
        "function: test",
        "left_hand_label: none",
        "left_hand_shape: none",
        "right_hand_label: none",
        "right_hand_shape: none",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_segment(
    root: Path,
    *,
    video_filename: str,
    segment_name: str,
    label: int,
    start_ms: int = 113000,
    end_ms: int = 115000,
) -> None:
    segment_dir = root / video_filename / segment_name
    segment_dir.mkdir(parents=True, exist_ok=True)
    _write_min_landmarks(segment_dir / "landmarks.json")
    _write_annotation(
        segment_dir / "annotation.txt",
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _write_timed_cha(path: Path, *, media_ref: str | None = None) -> None:
    header = "@Begin\n"
    if media_ref is not None:
        header += f"@Media: {media_ref}, video\n"
    body = (
        "*INV: can you tell me about your trip ? \x154075_7935\x15\n"
        "*PAR: &-um I caught the flu from my mom . \x15113123_114473\x15\n"
        "*PAR: no overlap here . \x15200000_200100\x15\n"
        "@End\n"
    )
    path.write_text(header + body, encoding="utf-8")


def _make_fixture_root(root: Path) -> dict[str, Path]:
    landmarks_root = root / "gest_segments_landmarks"
    negative_root = root / "neg_gest_segments_landmarks"
    transcript_root = root / "transcripts"

    _write_segment(
        landmarks_root,
        video_filename="65-1.mp4",
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
    )
    _write_segment(
        negative_root,
        video_filename="65-1.mp4",
        segment_name="neg_0",
        label=NEGATIVE_SEGMENT_LABEL,
    )

    transcript_root.mkdir(parents=True, exist_ok=True)
    _write_timed_cha(transcript_root / "65-1.cha", media_ref="65-1")
    _write_timed_cha(
        transcript_root / "rc65_day1_AB.elan.cha",
        media_ref=None,
    )

    return {
        "landmarks_root": landmarks_root,
        "negative_root": negative_root,
        "transcript_root": transcript_root,
    }


def test_media_header_matching() -> None:
    cha_path = Path("example_media_header.cha")
    content = "@Begin\n@Media: 65-1, video\n@End\n"
    cha_path.write_text(content, encoding="utf-8")
    try:
        assert _parse_cha_media_reference(cha_path) == "65-1"
    finally:
        cha_path.unlink(missing_ok=True)


def test_fallback_filename_matching(fixture: dict[str, Path]) -> None:
    cha_paths = _scan_cha_transcript_files(fixture["transcript_root"])
    by_media, by_pair = _build_transcript_lookup(
        cha_paths,
        strict_transcripts=False,
        warned_keys=set(),
    )
    assert "65-1" in by_media
    assert ("65", "1") in by_pair
    assert any(
        path.name == "rc65_day1_AB.elan.cha"
        for path in by_pair[("65", "1")]
    )


def test_transcript_extraction_and_shapes(fixture: dict[str, Path]) -> None:
    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
        transcript_kwargs={
            "context_before_ms": 0,
            "context_after_ms": 0,
            "min_utterances_before": 0,
            "min_utterances_after": 0,
        },
        return_tensor=True,
    )

    pos_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == "65-1.mp4" and item.label == POSITIVE_SEGMENT_LABEL
    )
    dataset.sample_indices = [pos_index]
    dataset.active_indices = [pos_index]
    sample = dataset[0]
    assert sample["landmarks"].shape == (2, 25, 4)
    assert sample["label"] == POSITIVE_SEGMENT_LABEL
    assert sample["gesture_type"] == "POS"
    assert len(sample["transcript_chunks"]) == dataset.num_transcript_chunks
    assert sample["transcript_chunk_mask"].shape == (dataset.num_transcript_chunks,)
    assert "PAR:" in sample["transcript"]
    assert "INV:" not in sample["transcript"]
    assert sample["transcript_chunk_mask"].any()

    neg_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.label == NEGATIVE_SEGMENT_LABEL
    )
    dataset.sample_indices = [neg_index]
    dataset.active_indices = [neg_index]
    neg_sample = dataset[0]
    assert neg_sample["label"] == NEGATIVE_SEGMENT_LABEL
    assert neg_sample["gesture_type"] == "NEG"
    assert "PAR:" in neg_sample["transcript"]

    stored = dataset.samples[pos_index]
    assert stored.transcript_path is not None
    assert stored.transcript_path.name == "65-1.cha"


def test_no_overlap_returns_empty_transcript(fixture: dict[str, Path]) -> None:
    _write_segment(
        fixture["landmarks_root"],
        video_filename="68-2.mp4",
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
        start_ms=300000,
        end_ms=301000,
    )
    _write_timed_cha(fixture["transcript_root"] / "68-2.cha", media_ref="68-2")

    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
        transcript_offset_seconds=3.0,
    )

    gap_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == "68-2.mp4"
    )
    dataset.sample_indices = [gap_index]
    dataset.active_indices = [gap_index]
    assert dataset[0]["transcript"] == ""
    assert len(dataset[0]["transcript_chunks"]) == dataset.num_transcript_chunks
    assert not dataset[0]["transcript_chunk_mask"].any()


def test_expanded_window_includes_nearby_transcript(fixture: dict[str, Path]) -> None:
    _write_segment(
        fixture["landmarks_root"],
        video_filename="70-3.mp4",
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
        start_ms=10000,
        end_ms=11000,
    )
    _write_timed_cha(fixture["transcript_root"] / "70-3.cha", media_ref="70-3")

    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
        transcript_offset_seconds=3.0,
        num_transcript_chunks=3,
    )
    gap_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == "70-3.mp4"
    )
    dataset.sample_indices = [gap_index]
    dataset.active_indices = [gap_index]
    sample = dataset[0]
    assert "INV:" in sample["transcript"]
    assert len(sample["transcript_chunks"]) == 3


def test_non_strict_missing_transcript(fixture: dict[str, Path]) -> None:
    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
    )
    missing_video = "99-9.mp4"
    _write_segment(
        fixture["landmarks_root"],
        video_filename=missing_video,
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
    )
    dataset.samples = dataset._assign_transcript_paths(dataset._scan_gesture_segments())

    missing_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == missing_video
    )
    dataset.sample_indices = [missing_index]
    dataset.active_indices = [missing_index]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        item = dataset[0]
        item_again = dataset[0]

    assert item["transcript"] == ""
    assert item_again["transcript"] == ""
    missing_warnings = [
        warning
        for warning in caught
        if "No CHAT transcript found" in str(warning.message)
    ]
    assert len(missing_warnings) == 1


def test_strict_missing_transcript(root: Path) -> None:
    landmarks_root = root / "gest_only"
    negative_root = root / "neg_only"
    transcript_root = root / "transcripts_only"
    _write_segment(
        landmarks_root,
        video_filename="65-1.mp4",
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
    )
    _write_segment(
        negative_root,
        video_filename="65-1.mp4",
        segment_name="neg_0",
        label=NEGATIVE_SEGMENT_LABEL,
    )
    transcript_root.mkdir(parents=True, exist_ok=True)
    _write_timed_cha(transcript_root / "65-1.cha", media_ref="65-1")

    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=landmarks_root,
        negative_landmarks_dir=negative_root,
        transcript_dir=transcript_root,
        strict_transcripts=True,
    )
    missing_video = "99-9.mp4"
    _write_segment(
        landmarks_root,
        video_filename=missing_video,
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
    )
    dataset.samples = dataset._assign_transcript_paths(dataset._scan_gesture_segments())
    missing_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == missing_video
    )
    dataset.sample_indices = [missing_index]
    dataset.active_indices = [missing_index]

    try:
        _ = dataset[0]
    except ValueError as exc:
        assert "No CHAT transcript found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for strict missing transcript")


def test_from_existing_preserves_transcript_state(fixture: dict[str, Path]) -> None:
    base = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        transcript_kwargs={"context_before_ms": 0},
        strict_transcripts=False,
    )
    view = AphasiaGestureSegmentDataset.from_existing(
        base,
        sample_indices=[0],
    )

    assert view.transcript_dir == base.transcript_dir
    assert view.strict_transcripts == base.strict_transcripts
    assert view.transcript_kwargs == base.transcript_kwargs
    assert view._transcript_by_media_ref is base._transcript_by_media_ref
    assert view._transcript_by_pair_key is base._transcript_by_pair_key
    assert view._transcript_cache is base._transcript_cache
    assert view._missing_transcript_warned is base._missing_transcript_warned
    assert view[0]["transcript"] == base[0]["transcript"]
    assert view[0]["transcript_chunks"] == base[0]["transcript_chunks"]


def test_transcript_chunk_partitioning() -> None:
    lines = [
        {"speaker": "PAR", "text": "early", "start_ts": 1000, "end_ts": 2000},
        {"speaker": "PAR", "text": "middle", "start_ts": 2500, "end_ts": 3500},
        {"speaker": "PAR", "text": "late", "start_ts": 8000, "end_ts": 9000},
    ]
    chunks, mask = _build_transcript_chunks(
        lines,
        window_start_ms=0,
        window_end_ms=10000,
        num_chunks=5,
    )
    assert len(chunks) == 5
    assert mask.shape == (5,)
    assert "early" in chunks[0]
    assert "middle" in chunks[1] or "middle" in chunks[2]
    assert "late" in chunks[-1]


def test_variable_num_transcript_chunks(fixture: dict[str, Path]) -> None:
    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
        num_transcript_chunks=3,
        return_tensor=True,
    )
    sample = dataset[0]
    assert len(sample["transcript_chunks"]) == 3
    assert sample["transcript_chunk_mask"].shape == (3,)


def test_default_collate_accepts_variable_transcript_lengths(
    fixture: dict[str, Path],
) -> None:
    _write_segment(
        fixture["landmarks_root"],
        video_filename="68-2.mp4",
        segment_name="gest_0",
        label=POSITIVE_SEGMENT_LABEL,
        start_ms=300000,
        end_ms=301000,
    )
    _write_timed_cha(fixture["transcript_root"] / "68-2.cha", media_ref="68-2")

    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=fixture["landmarks_root"],
        negative_landmarks_dir=fixture["negative_root"],
        transcript_dir=fixture["transcript_root"],
        strict_transcripts=False,
        return_tensor=True,
    )

    pos_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == "65-1.mp4" and item.label == POSITIVE_SEGMENT_LABEL
    )
    gap_index = next(
        index
        for index, item in enumerate(dataset.samples)
        if item.video_filename == "68-2.mp4"
    )
    dataset.sample_indices = [pos_index, gap_index]
    dataset.active_indices = [pos_index, gap_index]

    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=gesture_segment_collate_fn,
            )
        )
    )
    assert isinstance(batch["transcripts"], list)
    assert len(batch["transcripts"]) == 2
    assert batch["transcripts"][0]
    assert batch["transcripts"][1] == ""
    assert len(batch["transcript_chunks"]) == 2
    assert batch["transcript_chunk_mask"].shape[0] == 2
    assert batch["landmarks"].shape[0] == 2


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture = _make_fixture_root(root)
        test_media_header_matching()
        test_fallback_filename_matching(fixture)
        test_transcript_extraction_and_shapes(fixture)
        test_no_overlap_returns_empty_transcript(fixture)
        test_expanded_window_includes_nearby_transcript(fixture)
        test_non_strict_missing_transcript(fixture)
        test_strict_missing_transcript(root)
        test_from_existing_preserves_transcript_state(fixture)
        test_transcript_chunk_partitioning()
        test_variable_num_transcript_chunks(fixture)
        test_default_collate_accepts_variable_transcript_lengths(fixture)

    print("gesture segment transcript tests: OK")


if __name__ == "__main__":
    main()
