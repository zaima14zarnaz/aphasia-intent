"""
ST-GCN gesture classification model for MediaPipe pose landmarks.

Input shape:
    [B, T, V, C]

Default output:
    logits [B, num_classes]

The class name LandmarkGestureClassifier is preserved so existing training code that
imports this model does not need to change.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT_SHOULDER_INDEX = 0
RIGHT_SHOULDER_INDEX = 1

NUM_UPPER_BODY_LANDMARKS = 14


# Default graph for a 14-joint upper-body MediaPipe-style subset.
# IMPORTANT: update this list if your dataloader uses a different joint order.
# Expected order:
#   0 left_shoulder,  1 right_shoulder,
#   2 left_elbow,     3 right_elbow,
#   4 left_wrist,     5 right_wrist,
#   6 left_pinky,     7 right_pinky,
#   8 left_index,     9 right_index,
#   10 left_thumb,    11 right_thumb,
#   12 left_hip,      13 right_hip
DEFAULT_UPPER_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),   # shoulder line
    (0, 2),   # left upper arm
    (2, 4),   # left lower arm
    (4, 6),   # left wrist -> pinky
    (4, 8),   # left wrist -> index
    (4, 10),  # left wrist -> thumb
    (1, 3),   # right upper arm
    (3, 5),   # right lower arm
    (5, 7),   # right wrist -> pinky
    (5, 9),   # right wrist -> index
    (5, 11),  # right wrist -> thumb
    (0, 12),  # left torso
    (1, 13),  # right torso
    (12, 13), # hip line
)


def _normalize_digraph(adjacency: torch.Tensor) -> torch.Tensor:
    """Column-normalize adjacency as in common ST-GCN implementations."""
    degree = adjacency.sum(dim=0)
    inv_degree = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_degree[nonzero] = degree[nonzero].pow(-1)
    return adjacency @ torch.diag(inv_degree)


def build_adjacency(
    num_joints: int,
    edges: Iterable[tuple[int, int]] | None = None,
    *,
    add_self_links: bool = True,
) -> torch.Tensor:
    """
    Build a normalized skeleton adjacency matrix with shape [1, V, V].

    The leading dimension is the graph-partition dimension K. This implementation
    uses K=1 for a compact upper-body graph. Each ST-GCN block receives its own
    learnable edge-importance matrix with the same shape.
    """
    if num_joints <= 0:
        raise ValueError(f"num_joints must be positive, got {num_joints}")

    adjacency = torch.zeros(num_joints, num_joints, dtype=torch.float32)

    if add_self_links:
        for joint_idx in range(num_joints):
            adjacency[joint_idx, joint_idx] = 1.0

    if edges is None:
        edges = DEFAULT_UPPER_BODY_EDGES

    for src, dst in edges:
        if 0 <= src < num_joints and 0 <= dst < num_joints:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0

    adjacency = _normalize_digraph(adjacency)
    return adjacency.unsqueeze(0)  # [K=1, V, V]


class GraphConv(nn.Module):
    """Spatial graph convolution used inside an ST-GCN block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.size(0) != self.kernel_size:
            raise ValueError(
                "adjacency first dimension must match graph kernel size, "
                f"got adjacency K={adjacency.size(0)} and kernel_size={self.kernel_size}"
            )

        batch_size, _, num_frames, num_joints = x.shape
        x = self.conv(x)
        x = x.view(batch_size, self.kernel_size, -1, num_frames, num_joints)
        x = torch.einsum("bkctv,kvw->bctw", x, adjacency).contiguous()
        return x


class STGCNBlock(nn.Module):
    """
    ST-GCN block:
        GraphConv -> BatchNorm -> ReLU -> Dropout -> TemporalConv -> BatchNorm
        -> residual add -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        temporal_kernel_size, spatial_kernel_size = kernel_size
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd for same padding")
        temporal_padding = (temporal_kernel_size - 1) // 2

        self.gcn = GraphConv(in_channels, out_channels, spatial_kernel_size)
        self.gcn_bn = nn.BatchNorm2d(out_channels)

        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(temporal_padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual: nn.Module | None = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        x = self.gcn(x, adjacency)
        x = self.gcn_bn(x)
        x = self.tcn(x)
        x = x + residual
        return self.relu(x)


class LandmarkGestureClassifier(nn.Module):
    """
    ST-GCN + BiLSTM gesture classifier.

    Architecture matching the figure:
        [B, T, V, C]
        -> optional landmark normalization
        -> reshape to [B, C, T, V]
        -> 10 ST-GCN blocks with per-block edge importance
        -> joint pooling
        -> BiLSTM
        -> FC layers
        -> logits or softmax probabilities

    By default this returns logits, which is what nn.CrossEntropyLoss expects.
    Set return_probabilities=True only for inference when you need softmax output.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        num_frames: int = 16,
        num_joints: int = NUM_UPPER_BODY_LANDMARKS,
        in_channels: int = 4,
        stgcn_channels: Sequence[int] = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_kernel_size: int = 9,
        dropout: float = 0.1,
        normalize_landmarks: bool = True,
        edge_importance_weighting: bool = True,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        fc_hidden_dim: int = 256,
        return_probabilities: bool = False,
        graph_edges: Iterable[tuple[int, int]] | None = None,
        # Legacy arguments kept for compatibility with the previous model.py.
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        use_motion_features: bool | None = None,
        temporal_attention_layers: int | None = None,
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
        if lstm_hidden_dim <= 0:
            raise ValueError(f"lstm_hidden_dim must be positive, got {lstm_hidden_dim}")
        if lstm_layers <= 0:
            raise ValueError(f"lstm_layers must be positive, got {lstm_layers}")

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.normalize_landmarks = normalize_landmarks
        self.edge_importance_weighting = edge_importance_weighting
        self.bidirectional = bidirectional
        self.return_probabilities = return_probabilities

        adjacency = build_adjacency(num_joints, graph_edges)
        self.register_buffer("A", adjacency, persistent=False)

        spatial_kernel_size = adjacency.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        channels = [in_channels, *stgcn_channels]
        blocks: list[nn.Module] = []
        for block_idx, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            blocks.append(
                STGCNBlock(
                    c_in,
                    c_out,
                    kernel_size,
                    stride=1,
                    dropout=dropout,
                    residual=block_idx != 0,
                )
            )
        self.st_gcn_blocks = nn.ModuleList(blocks)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.A)) for _ in self.st_gcn_blocks]
            )
        else:
            self.edge_importance = [torch.ones_like(self.A) for _ in self.st_gcn_blocks]

        stgcn_output_dim = channels[-1]
        self.lstm = nn.LSTM(
            input_size=stgcn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, num_classes),
        )

    def _validate_landmarks(self, landmarks: torch.Tensor) -> None:
        if landmarks.ndim != 4:
            raise ValueError(
                "Expected landmarks with shape [B, T, V, C], "
                f"got ndim={landmarks.ndim} with shape {tuple(landmarks.shape)}"
            )

        batch_size, num_frames, num_joints, in_channels = landmarks.shape

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if num_frames != self.num_frames:
            raise ValueError(
                f"Expected T={self.num_frames}, got T={num_frames}. "
                "If your dataloader samples a different window length, pass "
                "num_frames=<that length> when creating the model."
            )
        if num_joints != self.num_joints:
            raise ValueError(
                f"Expected V={self.num_joints}, got V={num_joints}. "
                "Check the selected landmark subset and graph_edges."
            )
        if in_channels != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={in_channels}")

    def normalize_pose_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Body-center and body-size normalize MediaPipe pose landmarks.

        The first three channels are treated as x/y/z and the fourth channel is
        treated as visibility. Visibility is preserved unchanged.
        """
        landmarks = torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

        xyz = landmarks[..., :3]
        visibility = landmarks[..., 3:4]

        visibility_mask = (visibility > 0.0).float()
        valid_count = visibility_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        center = (xyz * visibility_mask).sum(dim=2, keepdim=True) / valid_count
        xyz = xyz - center

        shoulder_dist = torch.linalg.norm(
            xyz[:, :, LEFT_SHOULDER_INDEX, :2] - xyz[:, :, RIGHT_SHOULDER_INDEX, :2],
            dim=-1,
            keepdim=True,
        ).unsqueeze(-1)
        fallback_scale = xyz[..., :2].std(dim=2, keepdim=True).mean(dim=-1, keepdim=True)
        scale = torch.where(shoulder_dist > 1e-6, shoulder_dist, fallback_scale)
        xyz = xyz / scale.clamp_min(1e-6)

        return torch.cat([xyz, visibility], dim=-1)

    def _prepare_stgcn_input(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Convert [B, T, V, C] landmarks to normalized [B, C, T, V]."""
        x = landmarks.permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]

        batch_size, channels, num_frames, num_joints = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, V, C, T]
        x = x.view(batch_size, num_joints * channels, num_frames)
        x = self.data_bn(x)
        x = x.view(batch_size, num_joints, channels, num_frames)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, T, V]
        return x

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmarks: pose landmark tensor [B, T, V, C].

        Returns:
            logits [B, num_classes] by default, or softmax probabilities when
            return_probabilities=True.
        """
        self._validate_landmarks(landmarks)

        if self.normalize_landmarks:
            landmarks = self.normalize_pose_landmarks(landmarks)

        x = self._prepare_stgcn_input(landmarks)

        for st_gcn_block, edge_importance in zip(self.st_gcn_blocks, self.edge_importance):
            x = st_gcn_block(x, self.A * edge_importance)

        # Joint/global spatial pooling: [B, C, T, V] -> [B, T, C]
        x = x.mean(dim=-1).permute(0, 2, 1).contiguous()

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        logits = self.classifier(pooled)

        if self.return_probabilities:
            return F.softmax(logits, dim=-1)
        return logits


def _main() -> None:
    batch_size = 4
    num_frames = 16
    num_joints = NUM_UPPER_BODY_LANDMARKS
    in_channels = 4
    num_classes = 5

    model = LandmarkGestureClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        normalize_landmarks=True,
    )

    landmarks = torch.randn(batch_size, num_frames, num_joints, in_channels)
    logits = model(landmarks)

    print(f"landmarks shape: {tuple(landmarks.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, num_classes)


if __name__ == "__main__":
    _main()
