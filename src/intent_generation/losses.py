"""The two frame-selection losses in frame_selection_process.png.

D = mean Euclidean pose change between chronologically selected frames.
R = mean squared Euclidean joint distance to the reconstructed trajectory.
L_selection = -alpha * D + beta * R.

With complete joints these match the screenshot's denominators exactly. Missing
joints/padding are masked and each sample is averaged before averaging the batch.
IntentResponseLoss adds causal text CE and missing-field penalties.
CombinedTrainingLoss is the trainer's single entry point for every loss term,
the weighted total, and scaling for a gradient-accumulation window.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from intent_output_formatter import EXPECTED_FIELDS, FormattedIntentOutput, FormattedTokenOutput

if TYPE_CHECKING:
    from model.pC import FrameSelection


def visible_pose_change(selected_landmarks: Tensor, selected_joint_mask: Tensor) -> Tensor:
    """Positive D(S); the total loss negates it to encourage larger movement."""
    if selected_landmarks.ndim != 4 or selected_landmarks.shape[-1] != 3 or selected_joint_mask.shape != selected_landmarks.shape[:-1]:
        raise ValueError("Expected selected_landmarks [B,M,J,3] and mask [B,M,J]")
    if selected_landmarks.shape[1] < 2:
        raise ValueError("Pose-change loss requires at least two selected frames")
    pose = torch.where(selected_joint_mask.unsqueeze(-1), selected_landmarks, 0.0).float()
    pair_mask = selected_joint_mask[:, 1:] & selected_joint_mask[:, :-1]
    distances = torch.linalg.vector_norm(pose[:, 1:] - pose[:, :-1], dim=-1)
    counts = pair_mask.sum(dim=(1, 2))
    per_sample = (distances * pair_mask).sum(dim=(1, 2)) / counts.clamp_min(1)
    return per_sample.mean()


def reconstruct_trajectory(selected_landmarks: Tensor, selected_times: Tensor,
                           source_times: Tensor, selected_joint_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Piecewise-linear interpolation at original times, constant beyond ends.

    Interval indices are discrete; pose values and interpolation fractions retain
    autograd. selected_times must be strictly increasing in the forward pass.
    Only intervals with the required endpoint joints are supervised.
    """
    if selected_landmarks.ndim != 4 or selected_landmarks.shape[-1] != 3:
        raise ValueError("Expected selected_landmarks [B,M,J,3]")
    b, m, j, _ = selected_landmarks.shape
    if m < 2 or selected_times.shape != (b, m) or source_times.ndim != 2 or source_times.shape[0] != b or selected_joint_mask.shape != (b, m, j):
        raise ValueError("Invalid trajectory reconstruction shapes or fewer than two selected frames")
    if not torch.isfinite(selected_times).all() or (selected_times[:, 1:] <= selected_times[:, :-1]).any():
        raise ValueError("Selected frame times must be finite and strictly increasing")
    if not torch.isfinite(source_times).all():
        raise ValueError("source_times must be finite; replace padded times before reconstruction")
    pose = torch.where(selected_joint_mask.unsqueeze(-1), selected_landmarks, 0.0).float()
    selected_times, source_times = selected_times.float(), source_times.float()
    right = torch.searchsorted(selected_times.detach().contiguous(), source_times.detach().contiguous(), right=True).clamp(1, m - 1)
    left = right - 1
    left_time, right_time = selected_times.gather(1, left), selected_times.gather(1, right)
    fraction = ((source_times - left_time) / (right_time - left_time).clamp_min(1e-8)).clamp(0, 1)
    def gather_pose(indices: Tensor) -> Tensor:
        return pose.gather(1, indices[:, :, None, None].expand(-1, -1, j, 3))
    p_left, p_right = gather_pose(left), gather_pose(right)
    weight = fraction[:, :, None, None]
    reconstructed = (1 - weight) * p_left + weight * p_right
    left_mask = selected_joint_mask.gather(1, left.unsqueeze(-1).expand(-1, -1, j))
    right_mask = selected_joint_mask.gather(1, right.unsqueeze(-1).expand(-1, -1, j))
    # At an exact endpoint only that endpoint's joint is required.
    requires_left = (fraction < 1).unsqueeze(-1)
    requires_right = (fraction > 0).unsqueeze(-1)
    valid = (~requires_left | left_mask) & (~requires_right | right_mask)
    return reconstructed, valid


def trajectory_reconstruction_loss(original_landmarks: Tensor, selection: FrameSelection) -> Tensor:
    """R(S); sum squared XYZ errors per joint, then mean over frames/joints."""
    if original_landmarks.shape != (*selection.source_joint_mask.shape, 3):
        raise ValueError("Original landmark shape does not match selection")
    reconstruction, reconstructed_mask = reconstruct_trajectory(
        selection.selected_landmarks, selection.selected_times,
        selection.source_times, selection.selected_joint_mask)
    mask = selection.source_joint_mask & reconstructed_mask & selection.source_frame_mask.unsqueeze(-1)
    original = torch.where(mask.unsqueeze(-1), original_landmarks, 0.0).float()
    reconstruction = torch.where(mask.unsqueeze(-1), reconstruction, 0.0)
    # Sum over XYZ, not mean: the screenshot uses squared vector norm.
    squared_distance = (original - reconstruction).square().sum(-1)
    counts = mask.sum(dim=(1, 2))
    return (squared_distance.sum(dim=(1, 2)) / counts.clamp_min(1)).mean()


class FrameSelectionLoss(nn.Module):
    """Return a differentiable total plus named D and R components.

    alpha and beta depend on coordinate scale; defaults are starting values,
    not validated hyperparameters. Samples without valid comparisons contribute
    zero for that term. No extra entropy, ordering, or diversity penalties added.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
            raise ValueError("alpha and beta must be finite and nonnegative")
        self.alpha, self.beta = alpha, beta

    def forward(self, original_landmarks: Tensor, selection: FrameSelection) -> dict[str, Tensor]:
        movement = visible_pose_change(selection.selected_landmarks, selection.selected_joint_mask)
        reconstruction = trajectory_reconstruction_loss(original_landmarks, selection)
        return {"loss": -self.alpha * movement + self.beta * reconstruction,
                "pose_change": movement, "reconstruction": reconstruction}


class IntentResponseLoss(nn.Module):
    """Causal token cross-entropy on intent/reasoning, excluding evidence.

    For each missing field add missing_weight * (1 + its teacher-forced NLL).
    The count term is discrete; corrective NLL provides an additional gradient.
    Keys/syntax for the two supervised fields are included in their token masks.
    """

    def __init__(self, missing_weight: float = 1.0):
        super().__init__()
        if not math.isfinite(missing_weight) or missing_weight < 0:
            raise ValueError("missing_weight must be finite and nonnegative")
        self.missing_weight = missing_weight

    def forward(self, response: FormattedTokenOutput,
                generated: list[FormattedIntentOutput]) -> dict[str, Tensor]:
        logits, labels = response.logits[:, :-1], response.labels[:, 1:]
        b, length = labels.shape
        if len(generated) != b:
            raise ValueError("One generated response is required for every reference")
        masks = {field: response.field_masks[field][:, 1:] & (labels != -100) for field in EXPECTED_FIELDS}
        supervised = masks[EXPECTED_FIELDS[0]] | masks[EXPECTED_FIELDS[1]]
        if not supervised.any():
            raise ValueError("No supervised intent/reasoning tokens")
        losses = F.cross_entropy(logits[supervised].float(), labels[supervised], reduction="none")
        token_losses = torch.zeros((b, length), dtype=losses.dtype, device=logits.device).masked_scatter(supervised, losses)
        per_field = {}
        result = {"text_ce": losses.mean()}
        missing = torch.tensor([[field in row.missing_fields for field in EXPECTED_FIELDS] for row in generated], dtype=losses.dtype, device=logits.device)
        for index, field in enumerate(EXPECTED_FIELDS):
            counts = masks[field].sum(-1)
            if (counts == 0).any():
                raise ValueError(f"An annotation has no supervised tokens for {field}")
            per_field[field] = (token_losses * masks[field]).sum(-1) / counts
            result[("intent" if index == 0 else "reasoning") + "_ce"] = (token_losses * masks[field]).sum() / counts.sum()
        corrective = sum(missing[:, i] * per_field[field] for i, field in enumerate(EXPECTED_FIELDS)).mean()
        result["missing_count"] = missing.sum(-1).mean()
        result["missing_penalty"] = self.missing_weight * (result["missing_count"] + corrective)
        result["loss"] = result["text_ce"] + result["missing_penalty"]
        return result


class CombinedTrainingLoss(nn.Module):
    """Compute all training/evaluation losses in one place.

    Total = intent/reasoning causal CE + missing-field penalty
            + selection_loss_weight * (-selection_alpha*D + selection_beta*R).

    Evidence is excluded by IntentResponseLoss's field masks. Existing pose
    formulas remain unchanged. All tensors retain their autograd connections.
    backward_loss scales the same total for the actual accumulation-window size;
    loss and the component values remain unscaled for consistent reporting.
    """

    def __init__(self, missing_field_weight: float = 1.0,
                 selection_loss_weight: float = 1.0,
                 selection_alpha: float = 1.0, selection_beta: float = 1.0):
        super().__init__()
        if not math.isfinite(selection_loss_weight) or selection_loss_weight < 0:
            raise ValueError("selection_loss_weight must be finite and nonnegative")
        self.response_loss = IntentResponseLoss(missing_weight=missing_field_weight)
        self.selection_loss = FrameSelectionLoss(alpha=selection_alpha, beta=selection_beta)
        self.selection_loss_weight = selection_loss_weight

    def forward(self, response: FormattedTokenOutput,
                generated: list[FormattedIntentOutput], original_landmarks: Tensor,
                selection: FrameSelection, backward_divisor: int = 1) -> dict[str, Tensor]:
        if type(backward_divisor) is not int or backward_divisor <= 0:
            raise ValueError("backward_divisor must be a positive integer")
        text = self.response_loss(response, generated)
        pose = self.selection_loss(original_landmarks, selection)
        total = text["loss"] + self.selection_loss_weight * pose["loss"]
        return {
            "loss": total,
            "backward_loss": total / backward_divisor,
            "response_loss": text["loss"],
            "text_ce": text["text_ce"],
            "intent_ce": text["intent_ce"],
            "reasoning_ce": text["reasoning_ce"],
            "missing_count": text["missing_count"],
            "missing_penalty": text["missing_penalty"],
            "selection_loss": pose["loss"],
            "pose_change": pose["pose_change"],
            "reconstruction": pose["reconstruction"],
        }
