"""
Loss functions for gesture classification.
"""

from __future__ import annotations

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

