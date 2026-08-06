"""
Landmark-based POS/NEG gesture classifier.

The public class name ``LandmarkGestureClassifier`` is kept for compatibility with
``train.py``. Inputs may arrive as ``[B, T, J, C]``, ``[B, C, T, J]``, or
``[B, T, C, J]`` and are converted internally to canonical ``[B, T, J, C]``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

NUM_UPPER_BODY_LANDMARKS = 25
MEDIAPIPE_LANDMARK_CHANNELS = 4
DEFAULT_COORD_CLAMP = 10.0
DEFAULT_NODE_FEATURE_DIM = 12
DEFAULT_NORMALIZATION_EPS = 1e-6

# MediaPipe face + upper-body subset (indices 0-24 in the stored landmark order).
DEFAULT_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (0, 1),
    (0, 4),
    (0, 9),
    (0, 10),
    (9, 10),
    (0, 7),
    (0, 8),
    (7, 11),
    (8, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (15, 16),
    (15, 12),
    (16, 11),
)

DEFAULT_GESTURE_JOINT_INDICES: tuple[int, ...] = (
    15, 16, 17, 18, 19, 20, 21, 22,
)


def build_adjacency(
    num_joints: int,
    edges: Iterable[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return a row-normalized adjacency matrix with shape [J, J]."""
    if num_joints <= 0:
        raise ValueError(f"num_joints must be positive, got {num_joints}")

    adjacency = torch.eye(num_joints, dtype=torch.float32)
    edge_list = DEFAULT_SKELETON_EDGES if edges is None else edges
    for src, dst in edge_list:
        if 0 <= src < num_joints and 0 <= dst < num_joints:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0

    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def canonicalize_landmarks(
    landmarks: torch.Tensor,
    *,
    num_joints: int,
    in_channels: int,
) -> torch.Tensor:
    """
    Convert supported landmark layouts to ``[B, T, J, C]``.

    Supported layouts:
        [B, T, J, C]
        [B, T, C, J]
        [B, C, T, J]
    """
    if landmarks.ndim != 4:
        raise ValueError(
            "Expected landmarks with ndim=4, "
            f"got ndim={landmarks.ndim} shape={tuple(landmarks.shape)}"
        )

    shape = landmarks.shape
    if shape[2] == num_joints and shape[3] == in_channels:
        return landmarks.contiguous()

    if shape[2] == in_channels and shape[3] == num_joints:
        return landmarks.permute(0, 1, 3, 2).contiguous()

    channel_axis = None
    joint_axis = None
    for axis, size in enumerate(shape):
        if size == in_channels and channel_axis is None:
            channel_axis = axis
        if size == num_joints and joint_axis is None:
            joint_axis = axis

    if channel_axis is None or joint_axis is None:
        raise ValueError(
            "Could not infer landmark layout. Expected one dimension equal to "
            f"num_joints={num_joints} and one equal to in_channels={in_channels}, "
            f"got shape={tuple(shape)}"
        )

    if channel_axis == joint_axis:
        raise ValueError(
            "Ambiguous landmark layout: joint and channel dimensions are equal "
            f"for shape={tuple(shape)}"
        )

    batch_axis = 0
    temporal_axis = {0, 1, 2, 3} - {batch_axis, channel_axis, joint_axis}
    if len(temporal_axis) != 1:
        raise ValueError(
            f"Could not infer temporal axis for landmark shape={tuple(shape)}"
        )
    temporal_axis = next(iter(temporal_axis))

    permute_order = (batch_axis, temporal_axis, joint_axis, channel_axis)
    if permute_order == (0, 1, 2, 3):
        return landmarks

    return landmarks.permute(*permute_order).contiguous()


def split_mediapipe_landmarks(
    landmarks: torch.Tensor,
    *,
    in_channels: int = MEDIAPIPE_LANDMARK_CHANNELS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split MediaPipe-style landmarks ``[B, T, J, C]`` into position, visibility,
    and validity mask.
    """
    if landmarks.ndim != 4:
        raise ValueError(
            "Expected landmarks with shape [B, T, J, C], "
            f"got ndim={landmarks.ndim} shape={tuple(landmarks.shape)}"
        )
    if in_channels < 4:
        raise ValueError(
            "MediaPipe preprocessing expects at least 4 channels (x, y, z, v), "
            f"got in_channels={in_channels}"
        )

    landmarks = torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
    position = landmarks[..., :3]
    visibility = landmarks[..., 3:4].clamp(min=0.0, max=1.0)
    valid_mask = (visibility > 0.0).float()
    position = position * valid_mask
    return position, visibility, valid_mask


def compute_frame_reference_center(
    position: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    left_shoulder_index: int = LEFT_SHOULDER_INDEX,
    right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
    left_hip_index: int = LEFT_HIP_INDEX,
    right_hip_index: int = RIGHT_HIP_INDEX,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> torch.Tensor:
    """
    Compute a per-frame body reference center with shape ``[B, T, 1, 3]``.

    Priority:
        1. Shoulder midpoint when both shoulders are visible.
        2. Mean of visible shoulder/hip reference joints.
        3. Mean of all visible landmarks.
    """
    visible_mean = (position * valid_mask).sum(dim=2, keepdim=True)
    visible_mean = visible_mean / valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)

    def joint_pair_midpoint(left_idx: int, right_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        left_pos = position[:, :, left_idx : left_idx + 1, :]
        right_pos = position[:, :, right_idx : right_idx + 1, :]
        left_mask = valid_mask[:, :, left_idx : left_idx + 1, :]
        right_mask = valid_mask[:, :, right_idx : right_idx + 1, :]
        both_visible = left_mask * right_mask
        midpoint = 0.5 * (left_pos + right_pos)
        return midpoint, both_visible

    shoulder_mid, both_shoulders = joint_pair_midpoint(
        left_shoulder_index,
        right_shoulder_index,
    )

    reference_positions = []
    reference_masks = []
    for joint_idx in (
        left_shoulder_index,
        right_shoulder_index,
        left_hip_index,
        right_hip_index,
    ):
        if joint_idx < position.shape[2]:
            reference_positions.append(position[:, :, joint_idx : joint_idx + 1, :])
            reference_masks.append(valid_mask[:, :, joint_idx : joint_idx + 1, :])

    if reference_positions:
        refs = torch.cat(reference_positions, dim=2)
        ref_masks = torch.cat(reference_masks, dim=2)
        ref_weight = ref_masks.sum(dim=2, keepdim=True).clamp_min(eps)
        reference_center = (refs * ref_masks).sum(dim=2, keepdim=True) / ref_weight
        has_reference = ref_weight > 0
    else:
        reference_center = visible_mean
        has_reference = torch.zeros_like(visible_mean[..., :1], dtype=torch.bool)

    center = torch.where(both_shoulders > 0, shoulder_mid, reference_center)
    center = torch.where(has_reference, center, visible_mean)
    return center


def compute_frame_body_scale(
    position: torch.Tensor,
    valid_mask: torch.Tensor,
    center: torch.Tensor,
    *,
    left_shoulder_index: int = LEFT_SHOULDER_INDEX,
    right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> torch.Tensor:
    """
    Compute a per-frame body scale with shape ``[B, T, 1, 1]``.

    Uses shoulder distance when both shoulders are visible, otherwise RMS distance
    of visible landmarks from the reference center.
    """
    left_pos = position[:, :, left_shoulder_index : left_shoulder_index + 1, :2]
    right_pos = position[:, :, right_shoulder_index : right_shoulder_index + 1, :2]
    left_mask = valid_mask[:, :, left_shoulder_index : left_shoulder_index + 1, :]
    right_mask = valid_mask[:, :, right_shoulder_index : right_shoulder_index + 1, :]
    both_shoulders = left_mask * right_mask

    shoulder_dist = torch.linalg.norm(left_pos - right_pos, dim=-1, keepdim=True)
    shoulder_scale = shoulder_dist

    centered = (position - center) * valid_mask
    rms_scale = torch.sqrt((centered.square() * valid_mask).sum(dim=(2, 3), keepdim=True))
    rms_scale = rms_scale / valid_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

    scale = torch.where(both_shoulders > 0, shoulder_scale, rms_scale)
    return scale.clamp_min(eps)


def normalize_pose_positions(
    position: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    left_shoulder_index: int = LEFT_SHOULDER_INDEX,
    right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
    left_hip_index: int = LEFT_HIP_INDEX,
    right_hip_index: int = RIGHT_HIP_INDEX,
    coord_clamp: float = DEFAULT_COORD_CLAMP,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> torch.Tensor:
    """Center and scale pose positions, returning normalized xyz with shape ``[B,T,J,3]``."""
    center = compute_frame_reference_center(
        position,
        valid_mask,
        left_shoulder_index=left_shoulder_index,
        right_shoulder_index=right_shoulder_index,
        left_hip_index=left_hip_index,
        right_hip_index=right_hip_index,
        eps=eps,
    )
    scale = compute_frame_body_scale(
        position,
        valid_mask,
        center,
        left_shoulder_index=left_shoulder_index,
        right_shoulder_index=right_shoulder_index,
        eps=eps,
    )
    normalized = (position - center) / scale
    normalized = normalized.clamp(min=-coord_clamp, max=coord_clamp)
    return normalized * valid_mask


def temporal_delta(features: torch.Tensor) -> torch.Tensor:
    """First-order temporal difference with the first frame set to zero."""
    delta = torch.zeros_like(features)
    delta[:, 1:] = features[:, 1:] - features[:, :-1]
    return delta


def build_node_features(
    landmarks: torch.Tensor,
    *,
    num_joints: int,
    in_channels: int = MEDIAPIPE_LANDMARK_CHANNELS,
    left_shoulder_index: int = LEFT_SHOULDER_INDEX,
    right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
    left_hip_index: int = LEFT_HIP_INDEX,
    right_hip_index: int = RIGHT_HIP_INDEX,
    coord_clamp: float = DEFAULT_COORD_CLAMP,
    include_visibility_delta: bool = True,
    include_valid_mask: bool = True,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> torch.Tensor:
    """
    Build per-node features from MediaPipe landmarks.

    Returns:
        Node features with shape ``[B, T, J, node_in_dim]`` containing:
            normalized xyz,
            velocity xyz,
            acceleration xyz,
            visibility,
            optional visibility delta,
            optional valid landmark mask.
    """
    landmarks = canonicalize_landmarks(
        landmarks,
        num_joints=num_joints,
        in_channels=in_channels,
    )
    position, visibility, valid_mask = split_mediapipe_landmarks(
        landmarks,
        in_channels=in_channels,
    )
    normalized_position = normalize_pose_positions(
        position,
        valid_mask,
        left_shoulder_index=left_shoulder_index,
        right_shoulder_index=right_shoulder_index,
        left_hip_index=left_hip_index,
        right_hip_index=right_hip_index,
        coord_clamp=coord_clamp,
        eps=eps,
    )

    velocity = temporal_delta(normalized_position)
    acceleration = temporal_delta(velocity)

    feature_parts = [
        normalized_position,
        velocity,
        acceleration,
        visibility,
    ]
    if include_visibility_delta:
        feature_parts.append(temporal_delta(visibility))
    if include_valid_mask:
        feature_parts.append(valid_mask)

    return torch.cat(feature_parts, dim=-1)


class PoseFeaturePreprocessor(nn.Module):
    """
    Differentiable MediaPipe landmark preprocessor.

    Converts raw landmark tensors to node features with shape ``[B, T, J, D]``.
    """

    def __init__(
        self,
        *,
        num_joints: int,
        in_channels: int = MEDIAPIPE_LANDMARK_CHANNELS,
        left_shoulder_index: int = LEFT_SHOULDER_INDEX,
        right_shoulder_index: int = RIGHT_SHOULDER_INDEX,
        left_hip_index: int = LEFT_HIP_INDEX,
        right_hip_index: int = RIGHT_HIP_INDEX,
        coord_clamp: float = DEFAULT_COORD_CLAMP,
        include_visibility_delta: bool = True,
        include_valid_mask: bool = True,
        eps: float = DEFAULT_NORMALIZATION_EPS,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.left_shoulder_index = left_shoulder_index
        self.right_shoulder_index = right_shoulder_index
        self.left_hip_index = left_hip_index
        self.right_hip_index = right_hip_index
        self.coord_clamp = coord_clamp
        self.include_visibility_delta = include_visibility_delta
        self.include_valid_mask = include_valid_mask
        self.eps = eps

    @property
    def node_feature_dim(self) -> int:
        dim = 10
        if self.include_visibility_delta:
            dim += 1
        if self.include_valid_mask:
            dim += 1
        return dim

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        return build_node_features(
            landmarks,
            num_joints=self.num_joints,
            in_channels=self.in_channels,
            left_shoulder_index=self.left_shoulder_index,
            right_shoulder_index=self.right_shoulder_index,
            left_hip_index=self.left_hip_index,
            right_hip_index=self.right_hip_index,
            coord_clamp=self.coord_clamp,
            include_visibility_delta=self.include_visibility_delta,
            include_valid_mask=self.include_valid_mask,
            eps=self.eps,
        )


class GraphMix(nn.Module):
    """Skeleton-aware mixing over joints."""

    def __init__(self, dim: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("adjacency", adjacency.clone(), persistent=False)
        self.edge_gate = nn.Parameter(torch.zeros(adjacency.shape[0], adjacency.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, D]
        weights = self.adjacency * torch.sigmoid(self.edge_gate)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed = torch.einsum("ij,btjd->btid", weights, x)
        return mixed


class TemporalMix(nn.Module):
    """Depthwise temporal convolution per joint."""

    def __init__(self, dim: int, *, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for same-length padding")
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )
        self.pointwise = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_joints, dim = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch_size * num_joints, dim, num_frames)
        seq = self.conv(seq)
        seq = F.gelu(seq)
        seq = seq.reshape(batch_size, num_joints, dim, num_frames).permute(0, 3, 1, 2)
        seq = self.pointwise(seq)
        return self.dropout(seq)


class GraphTemporalBlock(nn.Module):
    """Graph mix + temporal mix + feed-forward with pre-norm residuals."""

    def __init__(
        self,
        dim: int,
        adjacency: torch.Tensor,
        *,
        temporal_kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.norm_graph = nn.LayerNorm(dim)
        self.graph_mix = GraphMix(dim, adjacency)
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_mix = TemporalMix(
            dim,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.graph_mix(self.norm_graph(x))
        x = x + self.temporal_mix(self.norm_temporal(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class GestureAttentionPool(nn.Module):
    """Pool encoded landmarks over joints and time with hand emphasis."""

    def __init__(
        self,
        dim: int,
        *,
        num_joints: int,
        gesture_joint_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        joint_prior = torch.ones(num_joints, dtype=torch.float32)
        indices = gesture_joint_indices
        if indices is None and num_joints == NUM_UPPER_BODY_LANDMARKS:
            indices = DEFAULT_GESTURE_JOINT_INDICES
        if indices is not None:
            for joint_idx in indices:
                if 0 <= joint_idx < num_joints:
                    joint_prior[joint_idx] = 2.0
        self.register_buffer("joint_prior", joint_prior, persistent=False)

        self.joint_score = nn.Linear(dim, 1)
        self.temporal_score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, D]
        joint_logits = self.joint_score(x).squeeze(-1)
        joint_logits = joint_logits + self.joint_prior.view(1, 1, -1).log()
        joint_weights = joint_logits.softmax(dim=2)
        joint_pooled = (x * joint_weights.unsqueeze(-1)).sum(dim=2)

        temporal_logits = self.temporal_score(joint_pooled).squeeze(-1)
        temporal_weights = temporal_logits.softmax(dim=1)
        temporal_mean = (joint_pooled * temporal_weights.unsqueeze(-1)).sum(dim=1)
        temporal_max = joint_pooled.max(dim=1).values

        return torch.cat([temporal_mean, temporal_max], dim=-1)


class LandmarkGestureClassifier(nn.Module):
    """
    Graph-temporal landmark classifier for binary POS/NEG gesture detection.

    ``forward`` accepts landmark tensors in several layouts and returns logits
    with shape ``[B, num_classes]``.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        num_frames: int = 16,
        num_joints: int = NUM_UPPER_BODY_LANDMARKS,
        in_channels: int = 4,
        stgcn_channels: Sequence[int] = (96, 96, 128, 128),
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
        normalize_landmarks: bool = False,
        edge_importance_weighting: bool = True,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        fc_hidden_dim: int = 256,
        return_probabilities: bool = False,
        graph_edges: Iterable[tuple[int, int]] | None = None,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        use_motion_features: bool | None = None,
        temporal_attention_layers: int | None = None,
        part_groups: Sequence[Sequence[int]] | None = None,
        add_gesture_edges: bool = True,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if num_joints <= 0:
            raise ValueError(f"num_joints must be positive, got {num_joints}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if not stgcn_channels:
            raise ValueError("stgcn_channels must contain at least one channel value")
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd for same padding")

        # Legacy kwargs kept for API compatibility with older training scripts.
        _ = (
            edge_importance_weighting,
            lstm_hidden_dim,
            lstm_layers,
            bidirectional,
            num_heads,
            temporal_attention_layers,
            part_groups,
            add_gesture_edges,
        )

        model_dim = int(embed_dim if embed_dim is not None else stgcn_channels[0])
        model_depth = int(depth if depth is not None else len(stgcn_channels))
        ratio = float(mlp_ratio if mlp_ratio is not None else 2.0)
        use_motion = True if use_motion_features is None else bool(use_motion_features)

        if model_dim <= 0:
            raise ValueError(f"embed_dim/model_dim must be positive, got {model_dim}")
        if model_depth <= 0:
            raise ValueError(f"depth must be positive, got {model_depth}")

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.normalize_landmarks = normalize_landmarks
        self.return_probabilities = return_probabilities
        self.use_motion_features = use_motion
        self.model_dim = model_dim

        adjacency = build_adjacency(num_joints, graph_edges)
        self.register_buffer("adjacency", adjacency, persistent=False)

        input_dim = in_channels * (2 if use_motion else 1)
        self.input_bn = nn.BatchNorm1d(input_dim * num_joints)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                GraphTemporalBlock(
                    model_dim,
                    adjacency,
                    temporal_kernel_size=temporal_kernel_size,
                    mlp_ratio=ratio,
                    dropout=dropout,
                )
                for _ in range(model_depth)
            ]
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.pool = GestureAttentionPool(
            model_dim,
            num_joints=num_joints,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, fc_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def _split_landmark_channels(
        self,
        landmarks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.in_channels >= 4:
            coords = landmarks[..., :3]
            visibility = landmarks[..., 3:4]
        elif self.in_channels == 3:
            coords = landmarks[..., :2]
            visibility = landmarks[..., 2:3]
        elif self.in_channels == 2:
            coords = landmarks
            visibility = torch.ones_like(coords[..., :1])
        else:
            coords = landmarks
            visibility = torch.ones_like(coords[..., :1])

        visibility = torch.nan_to_num(visibility, nan=0.0).clamp(min=0.0, max=1.0)
        mask = (visibility > 0.0).float()
        return coords, visibility, mask

    def sanitize_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
        if self.in_channels <= 1:
            return landmarks

        coords, visibility, mask = self._split_landmark_channels(landmarks)
        coords = coords * mask

        if self.in_channels >= 4:
            visibility = visibility * mask
            return torch.cat([coords, visibility], dim=-1)
        if self.in_channels == 3:
            return torch.cat([coords, visibility * mask], dim=-1)
        return coords

    def normalize_pose_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = self.sanitize_landmarks(landmarks)
        if self.in_channels <= 1:
            return landmarks

        coords, visibility, mask = self._split_landmark_channels(landmarks)
        valid_count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        center = (coords * mask).sum(dim=2, keepdim=True) / valid_count
        coords = coords - center

        if self.num_joints > max(LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX):
            shoulder_dist = torch.linalg.norm(
                coords[:, :, LEFT_SHOULDER_INDEX, :2]
                - coords[:, :, RIGHT_SHOULDER_INDEX, :2],
                dim=-1,
                keepdim=True,
            ).unsqueeze(-1)
        else:
            shoulder_dist = torch.zeros_like(coords[..., :1].mean(dim=2, keepdim=True))

        fallback_scale = coords[..., :2].std(dim=2, keepdim=True).mean(dim=-1, keepdim=True)
        scale = torch.where(shoulder_dist > 1e-6, shoulder_dist, fallback_scale)
        coords = coords / scale.clamp_min(1e-6)
        coords = coords * mask

        if self.in_channels >= 4:
            return torch.cat([coords, visibility * mask], dim=-1)
        if self.in_channels == 3:
            return torch.cat([coords, visibility * mask], dim=-1)
        return coords

    def _resample_time(self, landmarks: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_joints, channels = landmarks.shape
        if num_frames == self.num_frames:
            return landmarks

        seq = landmarks.permute(0, 2, 3, 1).reshape(
            batch_size,
            num_joints * channels,
            num_frames,
        )
        seq = F.interpolate(
            seq,
            size=self.num_frames,
            mode="linear",
            align_corners=False,
        )
        return seq.reshape(
            batch_size,
            num_joints,
            channels,
            self.num_frames,
        ).permute(0, 3, 1, 2).contiguous()

    def _prepare_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        if self.normalize_landmarks:
            landmarks = self.normalize_pose_landmarks(landmarks)
        else:
            landmarks = self.sanitize_landmarks(landmarks)

        features = landmarks
        if self.use_motion_features:
            motion = self._make_motion_features(landmarks)
            features = torch.cat([landmarks, motion], dim=-1)

        batch_size, num_frames, num_joints, channels = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(
            batch_size,
            num_joints * channels,
            num_frames,
        )
        flat = self.input_bn(flat)
        return flat.reshape(
            batch_size,
            num_joints,
            channels,
            num_frames,
        ).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _make_motion_features(landmarks: torch.Tensor) -> torch.Tensor:
        motion = torch.zeros_like(landmarks)
        motion[:, 1:] = landmarks[:, 1:] - landmarks[:, :-1]
        return motion

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = canonicalize_landmarks(
            landmarks,
            num_joints=self.num_joints,
            in_channels=self.in_channels,
        )
        landmarks = self._resample_time(landmarks)
        features = self._prepare_features(landmarks)

        x = self.input_proj(features)
        x = self.input_dropout(self.input_norm(x))

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        pooled = self.pool(x)
        logits = self.classifier(pooled)

        if self.return_probabilities:
            return F.softmax(logits, dim=-1)
        return logits


def _main() -> None:
    batch_size = 4
    num_frames = 128
    num_joints = NUM_UPPER_BODY_LANDMARKS
    in_channels = 4
    num_classes = 2

    model = LandmarkGestureClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
    )

    landmarks_btjc = torch.randn(batch_size, num_frames, num_joints, in_channels)
    landmarks_bctj = landmarks_btjc.permute(0, 3, 1, 2).contiguous()
    landmarks_btcj = landmarks_btjc.permute(0, 1, 3, 2).contiguous()

    for name, sample in (
        ("BTJC", landmarks_btjc),
        ("BCTJ", landmarks_bctj),
        ("BTCJ", landmarks_btcj),
    ):
        logits = model(sample)
        print(f"{name} in={tuple(sample.shape)} out={tuple(logits.shape)}")
        assert logits.shape == (batch_size, num_classes)


if __name__ == "__main__":
    _main()
