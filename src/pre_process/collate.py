"""
Collate utilities for batching gesture-segment samples from AphasiaDataset.

Example usage:

from torch.utils.data import DataLoader
from dataset_file import AphasiaGestureSegmentDataset
from collate import gesture_segment_collate_fn

dataset = AphasiaGestureEAFDataset(
    video_dir="/path/to/videos",
    annotation_dir="/path/to/annotations/aphasia",
    num_frames=None,
    frame_size=(224, 224),
    return_tensor=True,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=gesture_segment_collate_fn,
)

for batch in loader:
    landmarks = batch["landmarks"]
    mask = batch["landmark_mask"]
    annotations = batch["annotations"]
"""

from __future__ import annotations

import numpy as np
import torch


def _to_torch_video(frames: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Convert a single sample's frames to torch.Tensor [T, C, H, W].

    Accepts:
      - torch.Tensor [T, C, H, W]
      - numpy.ndarray [T, H, W, C] (uint8 or float)
    """
    if isinstance(frames, torch.Tensor):
        if frames.ndim != 4:
            raise ValueError(
                f"Expected torch frames with shape [T, C, H, W], got {tuple(frames.shape)}"
            )
        return frames

    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"frames must be torch.Tensor or numpy.ndarray, got {type(frames)!r}"
        )

    if frames.ndim != 4:
        raise ValueError(
            f"Expected numpy frames with shape [T, H, W, C], got {tuple(frames.shape)}"
        )

    video = torch.from_numpy(frames)

    if video.dtype == torch.uint8:
        video = video.float().div(255.0)
    else:
        video = video.float()

    return video.permute(0, 3, 1, 2).contiguous()


def _to_torch_landmarks(
    landmarks: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Convert a single sample's landmarks to torch.Tensor [T, J, C].
    """
    if isinstance(landmarks, torch.Tensor):
        if landmarks.ndim != 3:
            raise ValueError(
                "Expected torch landmarks with shape [T, J, C], "
                f"got {tuple(landmarks.shape)}"
            )
        return landmarks.float()

    if not isinstance(landmarks, np.ndarray):
        raise TypeError(
            "landmarks must be torch.Tensor or numpy.ndarray, "
            f"got {type(landmarks)!r}"
        )

    if landmarks.ndim != 3:
        raise ValueError(
            "Expected numpy landmarks with shape [T, J, C], "
            f"got {tuple(landmarks.shape)}"
        )

    return torch.from_numpy(landmarks).float()


def _pad_landmark_batch(
    landmarks_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Pad a list of [T, J, C] landmark sequences to [B, max_T, J, C].
    """
    if not landmarks_list:
        raise ValueError("Cannot collate an empty batch of landmarks.")

    lengths = torch.tensor(
        [landmarks.shape[0] for landmarks in landmarks_list],
        dtype=torch.long,
    )
    max_t = int(lengths.max().item())
    batch_size = len(landmarks_list)
    _, num_joints, num_channels = landmarks_list[0].shape

    padded_landmarks = landmarks_list[0].new_full(
        (batch_size, max_t, num_joints, num_channels),
        float("nan"),
    )
    landmark_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)

    for batch_idx, landmarks in enumerate(landmarks_list):
        num_frames = landmarks.shape[0]
        padded_landmarks[batch_idx, :num_frames] = landmarks
        landmark_mask[batch_idx, :num_frames] = True

    return padded_landmarks, landmark_mask, lengths


def _pad_video_batch(
    videos: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Pad a list of [T, C, H, W] videos to [B, max_T, C, H, W].

    Returns:
        padded_frames: [B, max_T, C, H, W]
        frame_mask:    [B, max_T], True for valid frames
        lengths:       [B], number of valid frames per sample
    """
    if not videos:
        raise ValueError("Cannot collate an empty batch of videos.")

    lengths = torch.tensor([video.shape[0] for video in videos], dtype=torch.long)
    max_t = int(lengths.max().item())
    batch_size = len(videos)
    _, channels, height, width = videos[0].shape

    padded_frames = videos[0].new_zeros((batch_size, max_t, channels, height, width))
    frame_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)

    for batch_idx, video in enumerate(videos):
        num_frames = video.shape[0]
        padded_frames[batch_idx, :num_frames] = video
        frame_mask[batch_idx, :num_frames] = True

    return padded_frames, frame_mask, lengths


def gesture_segment_collate_fn(batch: list[dict]) -> dict:
    """
    Collate gesture-segment samples with variable temporal length.

    Each sample is expected to have:
        {
            "landmarks": torch.Tensor [T, J, C] or numpy.ndarray [T, J, C],
            "transcript_chunks": list[str] (optional),
            "transcript_chunk_mask": torch.BoolTensor [N] (optional),
            "transcript": str (optional),
            "gesture_type": str,
            "annotation": dict (optional),
        }
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    landmarks_list: list[torch.Tensor] = []
    gesture_types: list[str | None] = []
    annotations: list[dict] = []

    for sample in batch:
        if "landmarks" not in sample:
            raise KeyError("Each sample must contain a 'landmarks' key.")

        landmarks = _to_torch_landmarks(sample["landmarks"])
        landmarks_list.append(landmarks)

        if "gesture_type" in sample:
            gesture_type = sample["gesture_type"]
        elif "annotation" in sample:
            gesture_type = sample["annotation"].get("label_name")
            if gesture_type is None:
                gesture_type = sample["annotation"].get("gesture_type")
        else:
            raise KeyError(
                "Each sample must contain 'gesture_type' or "
                "'annotation' with a label field."
            )

        gesture_types.append(
            None if gesture_type is None else str(gesture_type)
        )
        annotations.append(sample.get("annotation", {}))

    num_joints, num_channels = landmarks_list[0].shape[1:]
    for landmarks in landmarks_list[1:]:
        if landmarks.shape[1:] != (num_joints, num_channels):
            raise ValueError(
                "All landmarks in a batch must share the same joint/channel "
                f"dimensions. Found {(num_joints, num_channels)} and "
                f"{tuple(landmarks.shape[1:])}."
            )

    padded_landmarks, landmark_mask, lengths = _pad_landmark_batch(landmarks_list)
    transcripts = [str(sample.get("transcript", "")) for sample in batch]
    transcript_chunks = [list(sample.get("transcript_chunks", [])) for sample in batch]
    chunk_masks = [sample.get("transcript_chunk_mask") for sample in batch]
    aligned_utt_start_ts = [sample.get("aligned_utt_start_ts") for sample in batch]
    aligned_utt_end_ts = [sample.get("aligned_utt_end_ts") for sample in batch]
    aligned_utterances = [sample.get("aligned_utterance") for sample in batch]
    full_transcripts = [sample.get("full_transcript") for sample in batch]
    alignment_masks = [sample.get("transcript_alignment_mask") for sample in batch]

    if any(mask is not None for mask in chunk_masks):
        if not all(mask is not None for mask in chunk_masks):
            raise ValueError(
                "All samples in a batch must provide transcript_chunk_mask when "
                "transcript chunks are used."
            )
        num_chunks = chunk_masks[0].shape[0]
        for sample_chunks, sample_mask in zip(transcript_chunks, chunk_masks):
            if len(sample_chunks) != num_chunks:
                raise ValueError(
                    "All transcript_chunks in a batch must have the same length "
                    f"({num_chunks}), got {len(sample_chunks)}."
                )
            if int(sample_mask.shape[0]) != num_chunks:
                raise ValueError(
                    "transcript_chunk_mask length must match transcript_chunks."
                )
        transcript_chunk_mask = torch.stack(
            [mask.to(dtype=torch.bool) for mask in chunk_masks],
            dim=0,
        )
    else:
        transcript_chunk_mask = None

    if any(mask is not None for mask in alignment_masks):
        if not all(mask is not None for mask in alignment_masks):
            raise ValueError(
                "All samples in a batch must provide transcript_alignment_mask when "
                "alignment masks are used."
            )
        num_alignment_chunks = alignment_masks[0].shape[0]
        for sample_mask in alignment_masks:
            if int(sample_mask.shape[0]) != num_alignment_chunks:
                raise ValueError(
                    "transcript_alignment_mask length must be consistent within a "
                    "batch."
                )
        transcript_alignment_mask = torch.stack(
            [
                torch.as_tensor(
                    sample_mask,
                    dtype=torch.float32,
                )
                for sample_mask in alignment_masks
            ],
            dim=0,
        )
    else:
        transcript_alignment_mask = None

    return {
        "landmarks": padded_landmarks,
        "landmark_mask": landmark_mask,
        "labels": torch.tensor(
            [int(sample["label"]) for sample in batch],
            dtype=torch.long,
        ),
        "transcripts": transcripts,
        "transcript_chunks": transcript_chunks,
        "transcript_chunk_mask": transcript_chunk_mask,
        "aligned_utt_start_ts": aligned_utt_start_ts,
        "aligned_utt_end_ts": aligned_utt_end_ts,
        "aligned_utterance": aligned_utterances,
        "full_transcript": full_transcripts,
        "transcript_alignment_mask": transcript_alignment_mask,
        "gesture_types": gesture_types,
        "annotations": annotations,
        "lengths": lengths,
    }
