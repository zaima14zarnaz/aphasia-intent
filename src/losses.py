"""
Loss functions for gesture classification and pose-to-transcript alignment.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for gesture classification.

    Args:
        logits: Raw class logits with shape [B, num_classes].
        labels: Integer class indices with shape [B].
        class_weights: Optional per-class weights for cross-entropy.
        label_smoothing: Optional label smoothing factor in [0, 1).

    Returns:
        Scalar loss tensor.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2D with shape [B, num_classes], got {tuple(logits.shape)}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"labels must be 1D with shape [B], got {tuple(labels.shape)}"
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Batch size mismatch between logits and labels: "
            f"logits.shape[0]={logits.shape[0]}, labels.shape[0]={labels.shape[0]}"
        )

    if labels.dtype != torch.long:
        raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}")

    if logits.numel() > 0:
        if float(logits.min()) >= -1e-3 and float(logits.max()) <= 1.0 + 1e-3:
            row_sums = logits.detach().sum(dim=-1)
            if torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-3,
                rtol=0.0,
            ):
                raise ValueError(
                    "compute_classification_loss expected raw logits, but the "
                    "input looks like softmax probabilities. Disable "
                    "return_probabilities during training."
                )

    num_classes = logits.shape[1]
    if labels.numel() > 0:
        min_label = int(labels.min().item())
        max_label = int(labels.max().item())
        if min_label < 0 or max_label >= num_classes:
            raise ValueError(
                f"labels must be in [0, {num_classes - 1}], got "
                f"min={min_label}, max={max_label}"
            )

    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.shape[0] != num_classes:
            raise ValueError(
                "class_weights must be 1D with one weight per class, got "
                f"{tuple(class_weights.shape)} for num_classes={num_classes}"
            )
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)

    if class_weights is None:
        return F.cross_entropy(
            logits,
            labels,
            label_smoothing=label_smoothing,
        )

    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def get_batch_alignment_targets(
    batch: dict[str, Any],
    device: torch.device,
    expected_num_chunks: int,
) -> torch.Tensor:
    """Extract stacked partial alignment targets with shape ``[B, N]``."""
    if "transcript_alignment_mask" not in batch:
        raise KeyError(
            "Batch is missing 'transcript_alignment_mask'. Ensure "
            "gesture_segment_collate_fn is used and the dataset returns "
            "transcript_alignment_mask for each sample."
        )

    alignment_targets = torch.as_tensor(
        batch["transcript_alignment_mask"],
        dtype=torch.float32,
        device=device,
    )
    if alignment_targets.ndim != 2:
        raise ValueError(
            "transcript_alignment_mask must have shape [B, N], got "
            f"{tuple(alignment_targets.shape)}."
        )
    if alignment_targets.shape[1] != expected_num_chunks:
        raise ValueError(
            "transcript_alignment_mask chunk dimension mismatch: expected "
            f"{expected_num_chunks}, got {alignment_targets.shape[1]}."
        )
    if (alignment_targets < 0).any():
        raise ValueError("transcript_alignment_mask contains negative values.")
    if not torch.isfinite(alignment_targets).all():
        raise ValueError("transcript_alignment_mask contains non-finite values.")
    return alignment_targets


def compute_masked_alignment_loss(
    attention_weights: torch.Tensor,
    alignment_targets: torch.Tensor,
    transcript_chunk_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Soft-target cross-entropy between sample-level attention and alignment targets.

    Only samples with a non-zero alignment target contribute to the loss.
    """
    if attention_weights.ndim != 3:
        raise ValueError(
            "attention_weights must have shape [B, T, N], got "
            f"{tuple(attention_weights.shape)}."
        )
    batch_size, _, num_chunks = attention_weights.shape
    if alignment_targets.shape != (batch_size, num_chunks):
        raise ValueError(
            "alignment_targets must have shape "
            f"[{batch_size}, {num_chunks}], got {tuple(alignment_targets.shape)}."
        )
    if transcript_chunk_mask.shape != (batch_size, num_chunks):
        raise ValueError(
            "transcript_chunk_mask must have shape "
            f"[{batch_size}, {num_chunks}], got "
            f"{tuple(transcript_chunk_mask.shape)}."
        )

    mean_attention = attention_weights.mean(dim=1)
    annotated_mask = alignment_targets.sum(dim=-1) > 0
    num_annotated_samples = int(annotated_mask.sum().item())

    if num_annotated_samples == 0:
        return attention_weights.sum() * 0.0, 0, mean_attention

    annotated_predictions = mean_attention[annotated_mask].detach().cpu()
    annotated_targets_for_print = alignment_targets[annotated_mask].detach().cpu()
    # print(
    #     f"[alignment loss] num_annotated={num_annotated_samples}\n"
    #     f"  predictions: {annotated_predictions.tolist()}\n"
    #     f"  targets:     {annotated_targets_for_print.tolist()}"
    # )

    annotated_targets = alignment_targets[annotated_mask]
    annotated_chunk_mask = transcript_chunk_mask[annotated_mask]
    invalid_positive = (annotated_targets > 0) & (~annotated_chunk_mask)
    if bool(invalid_positive.any()):
        raise ValueError(
            "Alignment target assigns positive mass to invalid transcript chunks. "
            "Check transcript_alignment_mask against transcript_chunk_mask."
        )

    valid_positive = (annotated_targets > 0) & annotated_chunk_mask
    if not bool(valid_positive.any()):
        raise ValueError(
            "Annotated alignment samples must have at least one positive target "
            "on a valid transcript chunk."
        )

    normalized_target = alignment_targets / alignment_targets.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    if label_smoothing > 0.0:
        valid_uniform = transcript_chunk_mask.float() / transcript_chunk_mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        smoothed_target = (
            (1.0 - label_smoothing) * normalized_target
            + label_smoothing * valid_uniform
        )
    else:
        smoothed_target = normalized_target

    per_sample_loss = -(
        smoothed_target * torch.log(mean_attention.clamp_min(eps))
    ).sum(dim=-1)
    alignment_loss = per_sample_loss[annotated_mask].mean()
    return alignment_loss, num_annotated_samples, mean_attention


def compute_alignment_target_probability(
    mean_attention: torch.Tensor,
    alignment_targets: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average attention probability assigned to annotated chunks (diagnostic)."""
    annotated_mask = alignment_targets.sum(dim=-1) > 0
    if not bool(annotated_mask.any()):
        return mean_attention.sum() * 0.0

    normalized_target = alignment_targets / alignment_targets.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    target_probability = (mean_attention * normalized_target).sum(dim=-1)
    return target_probability[annotated_mask].mean()


def alignment_warmup_factor(epoch: int, alignment_warmup_epochs: int) -> float:
    if alignment_warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / alignment_warmup_epochs)
