"""
Compare gesture-classification inference on Selected-10 clips using two transcript sources:

1. CSV ``Aligned Utterance`` text (repeated across all transcript chunks).
2. CHAT ``.cha`` transcripts via ``AphasiaGestureSegmentDataset`` (same as train_multi).

Writes ``inference_outputs/results.csv`` with predictions from both sources.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

from model.custom_mamba import LandmarkGestureClassifier

_SRC_DIR = Path(__file__).resolve().parent
_PREPROCESS_DIR = _SRC_DIR / "pre_process"
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))

from collate import gesture_segment_collate_fn  # noqa: E402
from train_multi import (  # noqa: E402
    AphasiaGestureSegmentDataset,
    BEST_MODEL_PATH,
    IN_CHANNELS,
    LANDMARKS_DIR,
    NEGATIVE_LANDMARKS_DIR,
    NUM_FRAMES,
    NUM_JOINTS,
    TRANSCRIPT_DIR,
    get_batch_transcript_chunks,
    get_batch_transcripts,
    landmarks_to_tensor,
    print_model_config,
)

from dataloder import (  # noqa: E402
    PoseLandmarkSequence,
    _parse_media_timestamp_to_ms,
    _select_temporal_indices,
    load_landmarks_from_json,
    read_segment_landmarks,
    select_upper_body_landmarks,
)

TRANSCRIPT_FILE = (
    "/data/quantization/zaima/aphasiabank/neural2/control/unaligned_samples/"
    "anns/Selected-10-Table.csv"
)
INFERENCE_RESULT_PATH = _SRC_DIR / "inference_outputs" / "results.csv"
RESULT_CSV_FIELDNAMES = (
    "video_filename",
    "Gest-trans Temporal Window Gap",
    "Exceeds Average?",
    "CHA Transcript",
    "CHA transcript output labels",
    "Aligned utterance",
    "Aligned utterance output labels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LandmarkGestureClassifier predictions using CSV aligned "
            "utterances vs CHAT transcripts for clips listed in a transcript CSV."
        )
    )
    parser.add_argument(
        "--landmark-dir",
        type=Path,
        default=Path(LANDMARKS_DIR),
        help=(
            "Root directory containing training gesture landmarks at "
            "{video}.mp4/gest_{idx}/landmarks.json."
        ),
    )
    parser.add_argument(
        "--negative-landmark-dir",
        type=Path,
        default=Path(NEGATIVE_LANDMARKS_DIR),
        help="Negative landmark root required to initialize the gesture dataset.",
    )
    parser.add_argument(
        "--transcript-file",
        type=Path,
        default=Path(TRANSCRIPT_FILE),
        help=(
            "CSV with 'Actual Video Filename' and 'Aligned Utterance' columns "
            "(same format as ExplainableSaliency dataloader_v2)."
        ),
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        default=TRANSCRIPT_DIR,
        help="Directory containing CHAT .cha transcript files (train_multi default).",
    )
    parser.add_argument(
        "--transcript-offset-seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds of transcript context before/after each gesture when building "
            "CHAT transcript chunks (train_multi default)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INFERENCE_RESULT_PATH,
        help="CSV path for inference results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--gcn-dim", type=int, default=256)
    parser.add_argument("--gcn-layers", type=int, default=3)
    parser.add_argument("--graph-top-k", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name for transcript encoding.",
    )
    parser.add_argument(
        "--transcript-embed-dim",
        type=int,
        default=256,
        help="Projected BERT embedding dimension fused with frame tokens.",
    )
    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze BERT weights and only train the projection head.",
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Disable transcript encoding and fusion in the model.",
    )
    parser.add_argument(
        "--num-transcript-chunks",
        type=int,
        default=7,
        help="Number of equal-duration transcript chunks in the expanded window.",
    )
    parser.add_argument(
        "--max-transcript-chunks",
        type=int,
        default=16,
        help="Maximum supported transcript chunks for position embeddings.",
    )
    parser.add_argument(
        "--transcript-dropout-probability",
        type=float,
        default=0.0,
        help="Transcript modality dropout probability (0.0 for inference).",
    )
    parser.add_argument(
        "--skip-frame-padding",
        type=int,
        default=0,
        help=(
            "If > 0, drop this many landmark frames from the start and end of "
            "each clip before temporal subsampling."
        ),
    )
    args = parser.parse_args()
    if args.skip_frame_padding < 0:
        parser.error("--skip-frame-padding must be >= 0.")
    return args


def compute_gest_trans_temporal_window_gap(
    *,
    gesture_start_ts: str,
    gesture_end_ts: str,
    word_start_ts: str,
    word_end_ts: str,
) -> float:
    """
    Return temporal gap (seconds) between gesture and aligned-word windows.

    Gap = max(0, aligned-word start − gesture end, gesture start − aligned-word end)
    """
    gesture_start_ms = _parse_media_timestamp_to_ms(gesture_start_ts)
    gesture_end_ms = _parse_media_timestamp_to_ms(gesture_end_ts)
    word_start_ms = _parse_media_timestamp_to_ms(word_start_ts)
    word_end_ms = _parse_media_timestamp_to_ms(word_end_ts)

    gap_ms = max(
        0,
        word_start_ms - gesture_end_ms,
        gesture_start_ms - word_end_ms,
    )
    return gap_ms / 1000.0


def load_aligned_utterance_annotations(transcript_file: Path) -> dict[str, str]:
    """
    Load aligned utterances keyed by clip filename.

    Mirrors ``ExplainableSaliency.src.pre_process.dataloader_v2``.
    """
    csv_path = transcript_file.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Transcript CSV not found: {csv_path}")

    annotations: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_name = (row.get("Actual Video Filename") or "").strip()
            utterance = (row.get("Aligned Utterance") or "").strip()
            if not video_name or not utterance:
                continue

            annotations[video_name] = utterance
            annotations.setdefault(Path(video_name).stem, utterance)
    return annotations


def lookup_aligned_utterance(
    annotations: dict[str, str],
    video_filename: str,
) -> str | None:
    """Resolve the aligned utterance for one clip filename."""
    if not annotations:
        return None

    lookup_keys = [video_filename]
    if not video_filename.lower().endswith(".mp4"):
        lookup_keys.append(f"{video_filename}.mp4")
    lookup_keys.append(Path(video_filename).stem)

    for key in lookup_keys:
        utterance = annotations.get(key)
        if utterance:
            return utterance
    return None


def load_inference_clips_from_csv(
    transcript_file: Path,
) -> list[dict[str, Any]]:
    """Return ordered clip metadata dicts from the transcript CSV."""
    csv_path = transcript_file.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Transcript CSV not found: {csv_path}")

    annotations = load_aligned_utterance_annotations(csv_path)
    clips: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing_utterances: list[str] = []
    missing_timestamps: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_name = (row.get("Actual Video Filename") or "").strip()
            if not video_name or video_name in seen:
                continue
            seen.add(video_name)

            utterance = lookup_aligned_utterance(annotations, video_name)
            if utterance is None:
                missing_utterances.append(video_name)
                continue

            gesture_start_ts = (row.get("Gesture Start TS") or "").strip()
            gesture_end_ts = (row.get("Gesture End TS") or "").strip()
            word_start_ts = (row.get("Word Start TS") or "").strip()
            word_end_ts = (row.get("Word End TS") or "").strip()
            if not all(
                (gesture_start_ts, gesture_end_ts, word_start_ts, word_end_ts)
            ):
                missing_timestamps.append(video_name)
                continue

            try:
                gest_trans_gap = compute_gest_trans_temporal_window_gap(
                    gesture_start_ts=gesture_start_ts,
                    gesture_end_ts=gesture_end_ts,
                    word_start_ts=word_start_ts,
                    word_end_ts=word_end_ts,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid timestamp values for clip {video_name!r}: {exc}"
                ) from exc

            clips.append(
                {
                    "video_fname": video_name,
                    "aligned_utterance": utterance,
                    "gest_trans_temporal_window_gap": gest_trans_gap,
                    "exceeds_average": (row.get("Exceeds Average?") or "").strip(),
                }
            )

    if missing_utterances:
        missing_text = "\n".join(missing_utterances)
        raise KeyError(
            "Missing 'Aligned Utterance' for one or more clips in transcript CSV:\n"
            f"{missing_text}"
        )

    if missing_timestamps:
        missing_text = "\n".join(missing_timestamps)
        raise KeyError(
            "Missing gesture/word timestamp columns for one or more clips in "
            f"transcript CSV:\n{missing_text}"
        )

    if not clips:
        raise RuntimeError(
            f"No clips with aligned utterances found in transcript CSV: {csv_path}"
        )

    return clips


def build_transcript_sample_fields(
    aligned_utterance: str,
    *,
    num_transcript_chunks: int,
) -> dict[str, Any]:
    """
    Build transcript fields for one inference sample from a CSV aligned utterance.

    The utterance is split on whitespace into tokens. When there are fewer tokens
    than ``num_transcript_chunks``, the full utterance is repeated in every chunk
    (e.g. ``"one two"`` with 11 chunks -> 11 × ``"one two"``). Otherwise one
    token is placed in each chunk (using the first ``num_transcript_chunks`` tokens).
    """
    if num_transcript_chunks <= 0:
        raise ValueError(
            f"num_transcript_chunks must be positive, got {num_transcript_chunks}"
        )

    utterance = aligned_utterance.strip()
    empty_chunks = [""] * num_transcript_chunks
    empty_mask = torch.zeros(num_transcript_chunks, dtype=torch.bool)
    empty_alignment_mask = torch.zeros(num_transcript_chunks, dtype=torch.float32)

    if not utterance:
        return {
            "transcript": "",
            "transcript_chunks": empty_chunks,
            "transcript_chunk_mask": empty_mask,
            "aligned_utterance": None,
            "full_transcript": None,
            "aligned_utt_start_ts": None,
            "aligned_utt_end_ts": None,
            "transcript_alignment_mask": empty_alignment_mask,
        }

    tokens = utterance.split()
    if len(tokens) < num_transcript_chunks:
        transcript_chunks = [utterance] * num_transcript_chunks
    else:
        transcript_chunks = tokens[:num_transcript_chunks]

    transcript_chunk_mask = torch.tensor(
        [bool(chunk.strip()) for chunk in transcript_chunks],
        dtype=torch.bool,
    )
    transcript_alignment_mask = transcript_chunk_mask.to(dtype=torch.float32)

    return {
        "transcript": utterance,
        "transcript_chunks": transcript_chunks,
        "transcript_chunk_mask": transcript_chunk_mask,
        "aligned_utterance": utterance,
        "full_transcript": None,
        "aligned_utt_start_ts": None,
        "aligned_utt_end_ts": None,
        "transcript_alignment_mask": transcript_alignment_mask,
    }


def resolve_landmark_path(landmark_dir: Path, video_fname: str) -> Path:
    """
    Map a clip filename to its training landmark JSON path.

    Example:
        ``104-1_2_NUM.mp4`` -> ``landmark_dir/104-1.mp4/gest_1/landmarks.json``
    """
    stem = Path(video_fname).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot parse gesture index from clip filename {video_fname!r}. "
            "Expected format like '104-1_2_NUM.mp4'."
        )

    video_key = parts[0]
    if not parts[1].isdigit():
        raise ValueError(
            f"Cannot parse gesture index from clip filename {video_fname!r}. "
            f"Expected numeric gesture index, got {parts[1]!r}."
        )

    gesture_index = int(parts[1]) - 1
    if gesture_index < 0:
        raise ValueError(
            f"Invalid gesture index in clip filename {video_fname!r}: "
            f"{parts[1]!r}."
        )

    return (
        landmark_dir.expanduser().resolve()
        / f"{video_key}.mp4"
        / f"gest_{gesture_index}"
        / "landmarks.json"
    )


def find_dataset_index_for_landmarks_path(
    dataset: AphasiaGestureSegmentDataset,
    landmarks_path: Path,
) -> int:
    """Find the dataset sample index whose stored landmarks path matches."""
    target = landmarks_path.expanduser().resolve()
    for index, sample in enumerate(dataset.samples):
        if sample.landmarks_path.expanduser().resolve() == target:
            return index
    raise FileNotFoundError(
        f"No gesture dataset sample matched landmark file: {target}"
    )


def discover_inference_clips(
    transcript_file: Path,
    landmark_dir: Path,
    dataset: AphasiaGestureSegmentDataset,
) -> list[dict[str, Any]]:
    """Build inference entries from CSV clips, aligned utterances, and landmarks."""
    root = landmark_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Landmark directory not found: {root}")

    entries: list[dict[str, Any]] = []
    missing_landmarks: list[str] = []

    for clip in load_inference_clips_from_csv(transcript_file):
        video_fname = clip["video_fname"]
        aligned_utterance = clip["aligned_utterance"]
        landmarks_path = resolve_landmark_path(root, video_fname)
        if not landmarks_path.is_file():
            missing_landmarks.append(f"{video_fname} -> {landmarks_path}")
            continue

        dataset_index = find_dataset_index_for_landmarks_path(
            dataset,
            landmarks_path,
        )
        entries.append(
            {
                "video_fname": video_fname,
                "aligned_utterance": aligned_utterance,
                "gest_trans_temporal_window_gap": clip[
                    "gest_trans_temporal_window_gap"
                ],
                "exceeds_average": clip["exceeds_average"],
                "landmarks_path": landmarks_path,
                "dataset_index": dataset_index,
            }
        )

    if missing_landmarks:
        missing_text = "\n".join(missing_landmarks)
        raise FileNotFoundError(
            "Missing landmark files for one or more transcript clips:\n"
            f"{missing_text}"
        )

    if not entries:
        raise RuntimeError(
            f"No clips found in transcript CSV: {transcript_file}"
        )

    return entries


def trim_sequence_frame_padding(
    sequence: PoseLandmarkSequence,
    skip_frame_padding: int,
) -> PoseLandmarkSequence:
    """Remove ``skip_frame_padding`` frames from both ends of a landmark sequence."""
    if skip_frame_padding <= 0:
        return sequence

    num_frames = sequence.num_frames
    if num_frames <= 2 * skip_frame_padding:
        raise ValueError(
            "Cannot trim clip padding: sequence has "
            f"{num_frames} frames, but --skip-frame-padding={skip_frame_padding} "
            f"requires more than {2 * skip_frame_padding} frames."
        )

    frame_slice = slice(skip_frame_padding, num_frames - skip_frame_padding)
    return PoseLandmarkSequence(
        body=sequence.body[frame_slice],
        extractor=sequence.extractor,
        hand_left=(
            sequence.hand_left[frame_slice]
            if sequence.hand_left is not None
            else None
        ),
        hand_right=(
            sequence.hand_right[frame_slice]
            if sequence.hand_right is not None
            else None
        ),
    )


def load_landmark_tensor(
    landmarks_path: Path,
    *,
    num_frames: int,
    skip_frame_padding: int = 0,
) -> torch.Tensor:
    """Load and subsample one clip's landmarks to ``[T, J, C]``."""
    if skip_frame_padding <= 0:
        landmarks = read_segment_landmarks(
            landmarks_path,
            num_frames=num_frames,
            return_tensor=True,
        )
    else:
        sequence = load_landmarks_from_json(landmarks_path)
        sequence = trim_sequence_frame_padding(sequence, skip_frame_padding)
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
        landmarks = torch.from_numpy(
            select_upper_body_landmarks(sequence).astype("float32", copy=False)
        ).float()

    if not isinstance(landmarks, torch.Tensor):
        raise TypeError("Expected torch.Tensor landmarks from landmark loading.")
    return landmarks


def get_dataset_sample(
    dataset: AphasiaGestureSegmentDataset,
    samples_index: int,
) -> dict[str, Any]:
    """Return one collated-ready sample dict by index into ``dataset.samples``."""
    try:
        loader_index = dataset.active_indices.index(samples_index)
    except ValueError as exc:
        raise IndexError(
            f"Sample index {samples_index} is not in the dataset active indices."
        ) from exc
    return dict(dataset[loader_index])


def apply_skip_frame_padding_to_sample(
    sample: dict[str, Any],
    *,
    landmarks_path: Path,
    skip_frame_padding: int,
) -> dict[str, Any]:
    """Return a copy of ``sample`` with optionally trimmed landmark frames."""
    if skip_frame_padding <= 0:
        return sample

    updated = dict(sample)
    updated["landmarks"] = load_landmark_tensor(
        landmarks_path,
        num_frames=NUM_FRAMES,
        skip_frame_padding=skip_frame_padding,
    )
    return updated


def predict_sample_label(
    model: LandmarkGestureClassifier,
    sample: dict[str, Any],
    device: torch.device,
    idx_to_label: dict[int, str],
) -> str:
    """Run one forward pass and return the predicted class name."""
    batch = gesture_segment_collate_fn([sample])
    landmarks = landmarks_to_tensor(batch["landmarks"], device)
    transcripts = get_batch_transcripts(batch)
    transcript_chunks, transcript_chunk_mask = get_batch_transcript_chunks(
        batch,
        device,
    )

    logits = model(
        landmarks,
        transcript_chunks=transcript_chunks,
        transcript_chunk_mask=transcript_chunk_mask,
        transcripts=transcripts,
    )
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Expected model forward to return logits tensor.")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits.")
    return logits_to_label(logits, idx_to_label)


def build_model(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[LandmarkGestureClassifier, dict[int, str], int]:
    """Construct and load the classifier using the same config as ``train_multi.py``."""
    idx_to_label_raw = checkpoint.get("idx_to_label")
    if not idx_to_label_raw:
        raise KeyError("Checkpoint is missing 'idx_to_label'.")

    idx_to_label = {int(key): str(value) for key, value in idx_to_label_raw.items()}
    num_classes = len(idx_to_label)

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
        use_gcn=True,
        use_mamba_encoder=True,
        use_transcripts=not args.no_transcripts,
        bert_model_name=args.bert_model_name,
        transcript_embed_dim=args.transcript_embed_dim,
        freeze_bert=args.freeze_bert,
        num_transcript_chunks=args.num_transcript_chunks,
        max_transcript_chunks=args.max_transcript_chunks,
        transcript_dropout_probability=args.transcript_dropout_probability,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, idx_to_label, num_classes


def logits_to_label(
    logits: torch.Tensor,
    idx_to_label: dict[int, str],
) -> str:
    """Convert raw logits to a class name."""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(
            f"Expected logits with shape [1, num_classes], got {tuple(logits.shape)}"
        )

    pred_idx = int(logits.argmax(dim=-1).item())
    if pred_idx not in idx_to_label:
        raise KeyError(
            f"Predicted class index {pred_idx} missing from idx_to_label={idx_to_label}"
        )
    return idx_to_label[pred_idx]


@torch.inference_mode()
def run_inference(
    *,
    landmark_dir: Path,
    negative_landmark_dir: Path,
    transcript_file: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    """Run dual-transcript inference and write comparison ``results.csv``."""
    dataset = AphasiaGestureSegmentDataset(
        landmarks_dir=landmark_dir,
        negative_landmarks_dir=negative_landmark_dir,
        num_frames=NUM_FRAMES,
        return_tensor=True,
        transcript_dir=args.transcript_dir,
        strict_transcripts=False,
        transcript_offset_seconds=args.transcript_offset_seconds,
        num_transcript_chunks=args.num_transcript_chunks,
    )
    clip_entries = discover_inference_clips(transcript_file, landmark_dir, dataset)

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model, idx_to_label, num_classes = build_model(args, checkpoint, device)
    print_model_config(args, num_classes=num_classes)
    print(f"Label mapping: {idx_to_label}")
    print(f"Transcript CSV: {transcript_file}")
    print(f"Inference clips: {len(clip_entries)}")
    print()

    results: list[dict[str, str]] = []
    for entry in clip_entries:
        cha_sample = apply_skip_frame_padding_to_sample(
            get_dataset_sample(dataset, entry["dataset_index"]),
            landmarks_path=entry["landmarks_path"],
            skip_frame_padding=args.skip_frame_padding,
        )
        cha_transcript = str(cha_sample.get("transcript", ""))
        cha_label = predict_sample_label(model, cha_sample, device, idx_to_label)

        aligned_sample = apply_skip_frame_padding_to_sample(
            get_dataset_sample(dataset, entry["dataset_index"]),
            landmarks_path=entry["landmarks_path"],
            skip_frame_padding=args.skip_frame_padding,
        )
        aligned_utterance = entry["aligned_utterance"]
        aligned_sample.update(
            build_transcript_sample_fields(
                aligned_utterance,
                num_transcript_chunks=args.num_transcript_chunks,
            )
        )
        aligned_label = predict_sample_label(
            model,
            aligned_sample,
            device,
            idx_to_label,
        )

        row = {
            "video_filename": entry["video_fname"],
            "Gest-trans Temporal Window Gap": (
                f"{entry['gest_trans_temporal_window_gap']:.3f}"
            ),
            "Exceeds Average?": entry["exceeds_average"],
            "CHA Transcript": cha_transcript,
            "CHA transcript output labels": cha_label,
            "Aligned utterance": aligned_utterance,
            "Aligned utterance output labels": aligned_label,
        }
        results.append(row)
        print(
            f"{entry['video_fname']}: CHA={cha_label!r}, "
            f"Aligned={aligned_label!r}"
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} comparison results to: {output_path}")
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Landmark dir: {args.landmark_dir}")
    print(f"Negative landmark dir: {args.negative_landmark_dir}")
    print(f"Transcript CSV: {args.transcript_file}")
    print(f"CHAT transcript dir: {args.transcript_dir}")
    print(f"Transcript offset seconds: {args.transcript_offset_seconds}")
    print(f"Num transcript chunks: {args.num_transcript_chunks}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"Skip frame padding: {args.skip_frame_padding}")
    print()

    run_inference(
        landmark_dir=args.landmark_dir,
        negative_landmark_dir=args.negative_landmark_dir,
        transcript_file=args.transcript_file,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
