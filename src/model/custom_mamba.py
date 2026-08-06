"""
Custom Mamba-style landmark gesture classifier.

Uses differentiable MediaPipe preprocessing from ``model.posemamba`` and,
optionally, an edge-aware top-k graph convolution stack for spatial mixing,
followed by a bidirectional Mamba-style temporal encoder over frame tokens.
When ``USE_GCN=False``, graph building and GCN are skipped in favor of direct
kinematic embeddings (position, velocity, acceleration).
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Sequence

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None  # type: ignore[assignment, misc]
    AutoTokenizer = None  # type: ignore[assignment, misc]

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.posemamba import (
    DEFAULT_GESTURE_JOINT_INDICES,
    NUM_UPPER_BODY_LANDMARKS,
    PoseFeaturePreprocessor,
    canonicalize_landmarks,
)

__all__ = [
    "EdgeEmbedding",
    "EdgeAwareGraphConv",
    "EdgeAwareGraphConvStack",
    "FrameJointPool",
    "SimpleTemporalSSM",
    "BidirectionalTemporalSSM",
    "MambaTemporalEncoder",
    "TemporalAttentionPool",
    "GestureClassifierHead",
    "TranscriptBERTEncoder",
    "LandmarkGestureClassifier",
]

DEFAULT_BERT_MODEL_NAME = "bert-base-uncased"
DEFAULT_TRANSCRIPT_EMBED_DIM = 256
DEFAULT_BERT_MAX_LENGTH = 128
DEFAULT_NUM_TRANSCRIPT_CHUNKS = 5
DEFAULT_MAX_TRANSCRIPT_CHUNKS = 16
DEFAULT_TRANSCRIPT_DROPOUT_PROBABILITY = 0.2
INITIAL_GATE = 0.2
INITIAL_GATE_LOGIT = math.log(INITIAL_GATE / (1.0 - INITIAL_GATE))

DEFAULT_NORMALIZATION_EPS = 1e-6
DEFAULT_TEMPORAL_EMBED_DIM = 1024
CLASSIFIER_HIDDEN_DIMS: tuple[int, int] = (512, 256)
# Temporary default: skip Mamba and use GCN -> pool -> classifier only.
USE_MAMBA_ENCODER = False
# Temporary default: skip graph building + GCN; use kinematic embeddings only.
USE_GCN = False
KINEMATIC_FEATURE_DIM = 9  # normalized xyz + velocity xyz + acceleration xyz
_MAMBA_DISABLED_WARNED = False
_MAMBA_FALLBACK_WARNED = False
_GCN_DISABLED_WARNED = False


def _sanitize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    """Replace non-finite landmark values before feature extraction."""
    return torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)


def _validate_landmark_input(
    landmarks: torch.Tensor,
    *,
    num_joints: int,
    in_channels: int,
) -> None:
    """Validate raw landmark tensors before canonicalization."""
    if landmarks.ndim != 4:
        raise ValueError(
            "Expected landmarks with shape [B, T, J, C] or a supported permuted "
            f"layout, got ndim={landmarks.ndim} shape={tuple(landmarks.shape)}"
        )
    if landmarks.shape[0] < 1:
        raise ValueError(
            "Expected a batch dimension with size >= 1, "
            f"got shape={tuple(landmarks.shape)}"
        )

    shape = landmarks.shape
    has_joint_axis = num_joints in shape
    has_channel_axis = in_channels in shape
    if not has_joint_axis or not has_channel_axis:
        raise ValueError(
            "Landmark tensor must contain dimensions matching "
            f"num_joints={num_joints} and in_channels={in_channels}, "
            f"got shape={tuple(shape)}"
        )


def _validate_canonical_landmarks(
    landmarks: torch.Tensor,
    *,
    num_joints: int,
    in_channels: int,
) -> None:
    """Validate canonical ``[B, T, J, C]`` landmark tensors."""
    if landmarks.ndim != 4:
        raise ValueError(
            f"Expected canonical landmarks with ndim=4, got {landmarks.ndim}"
        )
    if landmarks.shape[0] < 1:
        raise ValueError(
            f"Expected batch dimension >= 1, got shape={tuple(landmarks.shape)}"
        )
    if landmarks.shape[1] < 1:
        raise ValueError(
            f"Expected at least one temporal frame, got shape={tuple(landmarks.shape)}"
        )
    if landmarks.shape[2] != num_joints:
        raise ValueError(
            f"Expected num_joints={num_joints}, got {landmarks.shape[2]} "
            f"in shape={tuple(landmarks.shape)}"
        )
    if landmarks.shape[3] != in_channels:
        raise ValueError(
            f"Expected in_channels={in_channels}, got {landmarks.shape[3]} "
            f"in shape={tuple(landmarks.shape)}"
        )
    if not torch.isfinite(landmarks).all():
        raise ValueError(
            "Landmarks contain NaN or Inf values after canonicalization."
        )


def _looks_like_probabilities(logits: torch.Tensor, *, atol: float = 1e-3) -> bool:
    """Heuristic check for softmax outputs mistakenly passed as logits."""
    if logits.numel() == 0:
        return False
    detached = logits.detach()
    if float(detached.min()) < -atol or float(detached.max()) > 1.0 + atol:
        return False
    row_sums = detached.sum(dim=-1)
    return bool(
        torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol, rtol=0.0)
    )


def _validate_logits(logits: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    """Ensure classifier outputs are well-formed raw logits, not probabilities."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [B, num_classes], got {tuple(logits.shape)}"
        )
    if logits.shape[0] < 1:
        raise ValueError(
            f"Expected batch dimension >= 1 in logits, got {tuple(logits.shape)}"
        )
    if logits.shape[-1] != num_classes:
        raise ValueError(
            f"Expected logits last dimension={num_classes}, "
            f"got shape={tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Classifier produced non-finite logits containing NaN or Inf."
        )
    if _looks_like_probabilities(logits):
        raise ValueError(
            "Model output looks like softmax probabilities. "
            "CrossEntropyLoss expects raw logits; keep return_probabilities=False."
        )
    return logits


def build_topk_graph(
    positions: torch.Tensor,
    top_k: int,
    *,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a sparse top-k neighbor graph from normalized joint positions.

    Args:
        positions: Normalized xyz with shape ``[B, T, J, 3]``.
        top_k: Number of non-self neighbors per node.

    Returns:
        neighbor_idx: Long tensor ``[B, T, J, K]``.
        edge_geometry: Raw edge geometry ``[B, T, J, K, 6]`` with
            ``[dx, dy, dz, distance, inverse_distance, rbf]``.
    """
    _ = eps
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError(
            "positions must have shape [B, T, J, 3], "
            f"got {tuple(positions.shape)}"
        )

    batch_size, num_frames, num_joints, _ = positions.shape
    if num_joints < 2:
        raise ValueError(
            f"build_topk_graph requires at least 2 joints, got {num_joints}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    # Self-information is already preserved by the residual connection inside
    # EdgeAwareGraphConv, so top-k neighbors exclude the center joint.
    neighbor_count = min(top_k, num_joints - 1)
    flat_positions = positions.reshape(batch_size * num_frames, num_joints, 3)

    diff = flat_positions.unsqueeze(2) - flat_positions.unsqueeze(1)
    distance = torch.linalg.norm(diff, dim=-1)
    distance = distance.masked_fill(
        torch.eye(num_joints, device=positions.device, dtype=torch.bool).unsqueeze(0),
        float("inf"),
    )
    _, neighbor_idx = torch.topk(
        distance,
        k=neighbor_count,
        dim=-1,
        largest=False,
    )

    batch_idx = torch.arange(
        flat_positions.shape[0],
        device=positions.device,
    ).view(-1, 1, 1)
    neighbor_positions = flat_positions[batch_idx, neighbor_idx]
    center_positions = flat_positions.unsqueeze(2)
    relative = neighbor_positions - center_positions

    neighbor_distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
    inverse_distance = 1.0 / (1.0 + neighbor_distance)
    rbf = torch.exp(-neighbor_distance.square())

    edge_geometry = torch.cat(
        [relative, neighbor_distance, inverse_distance, rbf],
        dim=-1,
    )
    neighbor_idx = neighbor_idx.reshape(
        batch_size,
        num_frames,
        num_joints,
        neighbor_count,
    )
    edge_geometry = edge_geometry.reshape(
        batch_size,
        num_frames,
        num_joints,
        neighbor_count,
        edge_geometry.shape[-1],
    )
    return neighbor_idx, edge_geometry


def gather_neighbors(features: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbor node features with shape ``[B, T, J, K, D]``."""
    batch_size, num_frames, num_joints, dim = features.shape
    neighbor_count = neighbor_idx.shape[-1]
    flat_features = features.reshape(batch_size * num_frames, num_joints, dim)
    flat_idx = neighbor_idx.reshape(batch_size * num_frames, num_joints, neighbor_count)
    batch_idx = torch.arange(
        flat_features.shape[0],
        device=features.device,
    ).view(-1, 1, 1)
    neighbors = flat_features[batch_idx, flat_idx]
    return neighbors.reshape(batch_size, num_frames, num_joints, neighbor_count, dim)


class EdgeEmbedding(nn.Module):
    """
    Embed relative geometry between landmark pairs.

    Raw edge geometry contains ``[dx, dy, dz, distance, inverse_distance, rbf]``.
    """

    RAW_EDGE_DIM = 6

    def __init__(
        self,
        edge_dim: int,
        *,
        rbf_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}")
        self.edge_dim = edge_dim
        self.rbf_gamma = rbf_gamma
        self.mlp = nn.Sequential(
            nn.Linear(self.RAW_EDGE_DIM, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,
        *,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_idx, edge_geometry = build_topk_graph(positions, top_k)
        if self.rbf_gamma != 1.0:
            distance = edge_geometry[..., 3:4]
            edge_geometry = edge_geometry.clone()
            edge_geometry[..., 5:6] = torch.exp(-self.rbf_gamma * distance.square())
        edge_embed = self.mlp(edge_geometry)
        return neighbor_idx, edge_embed


class EdgeAwareGraphConv(nn.Module):
    """
    Edge-conditioned message passing over a top-k landmark graph.

    For each frame, node features are projected to ``hidden_dim``, neighbor
    messages are weighted by learned edge-aware attention, then aggregated with
    a residual connection and normalization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.neighbor_proj = nn.Linear(out_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.attn = nn.Linear(out_dim * 3, 1)
        self.residual = (
            nn.Identity()
            if in_dim == out_dim
            else nn.Linear(in_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        edge_embed: torch.Tensor,
    ) -> torch.Tensor:
        center = self.node_proj(node_features)
        neighbors = gather_neighbors(center, neighbor_idx)
        neighbor_hidden = self.neighbor_proj(neighbors)
        edge_hidden = self.edge_proj(edge_embed)
        messages = neighbor_hidden + edge_hidden

        attn_input = torch.cat(
            [
                center.unsqueeze(-2).expand_as(messages),
                neighbor_hidden,
                edge_hidden,
            ],
            dim=-1,
        )
        attn_logits = self.attn(attn_input).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        aggregated = (attn_weights.unsqueeze(-1) * messages).sum(dim=-2)

        out = self.residual(node_features) + self.dropout(aggregated)
        return self.norm(out)


class EdgeAwareGraphConvStack(nn.Module):
    """Stack of edge-aware graph convolution layers."""

    def __init__(
        self,
        node_dim: int,
        gcn_dim: int,
        *,
        edge_dim: int = 32,
        num_layers: int = 2,
        top_k: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.top_k = top_k
        self.edge_embedding = EdgeEmbedding(edge_dim)

        layers: list[EdgeAwareGraphConv] = []
        in_dim = node_dim
        for _ in range(num_layers):
            layers.append(
                EdgeAwareGraphConv(
                    in_dim,
                    gcn_dim,
                    edge_dim,
                    dropout=dropout,
                )
            )
            in_dim = gcn_dim
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_features: torch.Tensor,
        normalized_positions: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_idx, edge_embed = self.edge_embedding(
            normalized_positions,
            top_k=self.top_k,
        )
        x = node_features
        for layer in self.layers:
            x = layer(x, neighbor_idx, edge_embed)
        return x


def build_temporal_ssm(
    d_model: int,
    *,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    temporal_kernel_size: int = 5,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a single-direction temporal SSM block.

    Uses ``mamba_ssm.Mamba`` when installed; otherwise falls back to
    ``SimpleTemporalSSM`` with a one-time warning.
    """
    global _MAMBA_FALLBACK_WARNED
    try:
        from mamba_ssm import Mamba

        return Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    except ImportError:
        if not _MAMBA_FALLBACK_WARNED:
            warnings.warn(
                "mamba_ssm is not installed; using SimpleTemporalSSM fallback "
                "for temporal encoding.",
                stacklevel=2,
            )
            print(
                "WARNING: mamba_ssm is not installed; "
                "using SimpleTemporalSSM fallback for temporal encoding."
            )
            _MAMBA_FALLBACK_WARNED = True
        return SimpleTemporalSSM(
            d_model,
            d_state=d_state,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )


class SimpleTemporalSSM(nn.Module):
    """
    Pure-PyTorch selective scan fallback with the same interface as ``Mamba``.

    Input/output shape: ``[B, T, D]``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd for same padding")
        padding = (temporal_kernel_size - 1) // 2

        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=temporal_kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.decay_bias = nn.Parameter(torch.full((d_model,), 1.0))
        self.skip = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"SimpleTemporalSSM expects [B, T, D], got {tuple(x.shape)}"
            )

        candidate, gate, decay = self.in_proj(x).chunk(3, dim=-1)
        candidate = self.depthwise_conv(candidate.transpose(1, 2)).transpose(1, 2)
        candidate = F.silu(candidate)
        gate = F.silu(gate)
        alpha = torch.sigmoid(decay + self.decay_bias.view(1, 1, -1))

        state = torch.zeros(
            x.size(0),
            self.d_model,
            dtype=x.dtype,
            device=x.device,
        )
        outputs: list[torch.Tensor] = []
        for frame_idx in range(x.size(1)):
            state = (
                alpha[:, frame_idx] * state
                + (1.0 - alpha[:, frame_idx]) * candidate[:, frame_idx]
            )
            outputs.append(state)

        y = torch.stack(outputs, dim=1)
        y = gate * y + self.skip.view(1, 1, -1) * x
        return self.dropout(self.out_proj(y))


class FrameJointPool(nn.Module):
    """Pool landmark/node features within each frame: ``[B, T, J, D] -> [B, T, D]``."""

    def __init__(
        self,
        dim: int,
        *,
        num_joints: int,
        pool_type: str = "attention",
        gesture_joint_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if pool_type not in {"attention", "mean"}:
            raise ValueError(
                f"pool_type must be 'attention' or 'mean', got {pool_type!r}"
            )

        self.dim = dim
        self.pool_type = pool_type
        if pool_type == "attention":
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "mean":
            return x.mean(dim=2)

        joint_logits = self.joint_score(x).squeeze(-1)
        joint_logits = joint_logits + self.joint_prior.view(1, 1, -1).log()
        joint_weights = joint_logits.softmax(dim=2)
        return (x * joint_weights.unsqueeze(-1)).sum(dim=2)


class BidirectionalTemporalSSM(nn.Module):
    """Forward/backward temporal SSM with concat + linear fusion."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        temporal_kernel_size: int = 5,
        dropout: float = 0.1,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        if fusion not in {"concat", "gated"}:
            raise ValueError(
                f"fusion must be 'concat' or 'gated', got {fusion!r}"
            )

        self.fusion = fusion
        ssm_kwargs = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "temporal_kernel_size": temporal_kernel_size,
            "dropout": dropout,
        }
        self.forward_ssm = build_temporal_ssm(d_model, **ssm_kwargs)
        self.backward_ssm = build_temporal_ssm(d_model, **ssm_kwargs)
        if fusion == "concat":
            self.merge = nn.Linear(d_model * 2, d_model)
        else:
            self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_out = self.forward_ssm(x)
        backward_out = self.backward_ssm(torch.flip(x, dims=[1]))
        backward_out = torch.flip(backward_out, dims=[1])
        merged = torch.cat([forward_out, backward_out], dim=-1)
        if self.fusion == "concat":
            return self.merge(merged)
        gate = torch.sigmoid(self.gate(merged))
        return gate * forward_out + (1.0 - gate) * backward_out


class MambaTemporalEncoderLayer(nn.Module):
    """Pre-norm bidirectional temporal SSM block with feed-forward."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        temporal_kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(d_model * mlp_ratio))
        self.norm_ssm = nn.LayerNorm(d_model)
        self.bidirectional_ssm = BidirectionalTemporalSSM(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
            fusion=fusion,
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bidirectional_ssm(self.norm_ssm(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class MambaTemporalEncoder(nn.Module):
    """
    Bidirectional Mamba-style temporal encoder over frame tokens.

    Input: ``[B, T, temporal_embed_dim]``
    Output: ``[B, T, temporal_embed_dim]``
    """

    def __init__(
        self,
        d_model: int = DEFAULT_TEMPORAL_EMBED_DIM,
        *,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        temporal_kernel_size: int = 5,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                MambaTemporalEncoderLayer(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    temporal_kernel_size=temporal_kernel_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    fusion=fusion,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TemporalAttentionPool(nn.Module):
    """Attention pooling over time: ``[B, T, D] -> [B, D]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.temporal_score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_logits = self.temporal_score(x).squeeze(-1)
        temporal_weights = temporal_logits.softmax(dim=1)
        return (x * temporal_weights.unsqueeze(-1)).sum(dim=1)


class GestureClassifierHead(nn.Module):
    """
    Sample-level classifier head for gesture logits.

    Architecture:
        Linear(1024, 512) -> LayerNorm -> GELU -> Dropout
        -> Linear(512, 256) -> GELU -> Dropout
        -> Linear(256, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: Sequence[int] = CLASSIFIER_HIDDEN_DIMS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if len(hidden_dims) != 2:
            raise ValueError(
                f"hidden_dims must contain exactly two values, got {hidden_dims}"
            )

        hidden_dim_1, hidden_dim_2 = (int(hidden_dims[0]), int(hidden_dims[1]))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, sample_embedding: torch.Tensor) -> torch.Tensor:
        if sample_embedding.ndim != 2:
            raise ValueError(
                "GestureClassifierHead expects sample embeddings with shape "
                f"[B, D], got {tuple(sample_embedding.shape)}"
            )
        return self.net(sample_embedding)


class TranscriptBERTEncoder(nn.Module):
    """Encode segment transcripts with a pretrained BERT encoder."""

    def __init__(
        self,
        *,
        bert_model_name: str = DEFAULT_BERT_MODEL_NAME,
        output_dim: int = DEFAULT_TRANSCRIPT_EMBED_DIM,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        max_length: int = DEFAULT_BERT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                "TranscriptBERTEncoder requires the 'transformers' package. "
                "Install it with: pip install transformers"
            )

        self.bert_model_name = bert_model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        hidden_size = int(self.bert.config.hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, transcripts: Sequence[str]) -> torch.Tensor:
        if not transcripts:
            raise ValueError("transcripts must contain at least one string.")

        encoded = self.tokenizer(
            list(transcripts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.bert(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_embedding)

    def encode_chunks(
        self,
        transcript_chunks: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """Encode per-sample transcript chunks to ``[B, N, output_dim]``."""
        if not transcript_chunks:
            raise ValueError("transcript_chunks must contain at least one sample.")

        batch_size = len(transcript_chunks)
        num_chunks = len(transcript_chunks[0])
        if num_chunks <= 0:
            raise ValueError("Each sample must contain at least one transcript chunk.")

        for sample_index, sample_chunks in enumerate(transcript_chunks):
            if len(sample_chunks) != num_chunks:
                raise ValueError(
                    "All samples must provide the same number of transcript chunks; "
                    f"sample 0 has {num_chunks}, sample {sample_index} has "
                    f"{len(sample_chunks)}."
                )

        flat_chunks = [
            str(chunk_text)
            for sample_chunks in transcript_chunks
            for chunk_text in sample_chunks
        ]
        flat_embeddings = self.forward(flat_chunks)
        return flat_embeddings.view(batch_size, num_chunks, self.output_dim)


class LandmarkGestureClassifier(nn.Module):
    """
    Edge-aware graph-temporal classifier for POS/NEG gesture detection.

    Pipeline (``use_gcn=False``, current temporary default):
        raw landmarks -> canonical ``[B, T, J, C]``
        -> kinematic node features ``[B, T, J, 9]`` (position, velocity, acceleration)
        -> linear embedding ``[B, T, J, gcn_dim]``
        -> frame joint pooling ``[B, T, gcn_dim]``
        -> temporal attention pooling ``[B, gcn_dim]``
        -> projection ``[B, classifier_dim]``
        -> logits ``[B, num_classes]``

    Pipeline (``use_gcn=True``, ``use_mamba_encoder=False``):
        raw landmarks -> canonical ``[B, T, J, C]``
        -> MediaPipe node features ``[B, T, J, node_dim]``
        -> edge-aware GCN stack ``[B, T, J, gcn_dim]``
        -> frame joint pooling / temporal pooling / classifier

    Pipeline (``use_mamba_encoder=True``, ``use_transcripts=True``):
        raw landmarks -> canonical ``[B, T, J, C]``
        -> spatial encoding (GCN or kinematic) + frame pooling ``[B, T, gcn_dim]``
        -> pose projection ``[B, T, hidden_dim]``
        -> BERT chunk encoding + soft pose-to-chunk attention
        -> gated residual fusion ``[B, T, hidden_dim]``
        -> bidirectional Mamba temporal encoder
        -> temporal attention pooling ``[B, hidden_dim]``
        -> logits ``[B, num_classes]``
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
        gcn_top_k: int = 8,
        gcn_layers: int = 2,
        gcn_edge_dim: int = 32,
        temporal_embed_dim: int = DEFAULT_TEMPORAL_EMBED_DIM,
        frame_pool_type: str = "attention",
        use_gcn: bool = USE_GCN,
        use_mamba_encoder: bool = USE_MAMBA_ENCODER,
        mamba_layers: int | None = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_fusion: str = "concat",
        use_temporal_attention_pool: bool = True,
        use_transcripts: bool = True,
        bert_model_name: str = DEFAULT_BERT_MODEL_NAME,
        transcript_embed_dim: int = DEFAULT_TRANSCRIPT_EMBED_DIM,
        freeze_bert: bool = False,
        bert_max_length: int = DEFAULT_BERT_MAX_LENGTH,
        num_transcript_chunks: int = DEFAULT_NUM_TRANSCRIPT_CHUNKS,
        max_transcript_chunks: int = DEFAULT_MAX_TRANSCRIPT_CHUNKS,
        transcript_dropout_probability: float = DEFAULT_TRANSCRIPT_DROPOUT_PROBABILITY,
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
        if use_gcn:
            if gcn_top_k <= 0:
                raise ValueError(f"gcn_top_k must be positive, got {gcn_top_k}")
            if gcn_layers <= 0:
                raise ValueError(f"gcn_layers must be positive, got {gcn_layers}")
        if temporal_embed_dim <= 0:
            raise ValueError(
                f"temporal_embed_dim must be positive, got {temporal_embed_dim}"
            )
        if use_transcripts and transcript_embed_dim <= 0:
            raise ValueError(
                f"transcript_embed_dim must be positive, got {transcript_embed_dim}"
            )
        if bert_max_length <= 0:
            raise ValueError(f"bert_max_length must be positive, got {bert_max_length}")
        if num_transcript_chunks <= 0:
            raise ValueError(
                f"num_transcript_chunks must be positive, got {num_transcript_chunks}"
            )
        if max_transcript_chunks <= 0:
            raise ValueError(
                f"max_transcript_chunks must be positive, got {max_transcript_chunks}"
            )
        if num_transcript_chunks > max_transcript_chunks:
            raise ValueError(
                "num_transcript_chunks must not exceed max_transcript_chunks: "
                f"{num_transcript_chunks} > {max_transcript_chunks}"
            )
        if not 0.0 <= transcript_dropout_probability <= 1.0:
            raise ValueError(
                "transcript_dropout_probability must be in [0, 1], got "
                f"{transcript_dropout_probability}"
            )
        if frame_pool_type not in {"attention", "mean"}:
            raise ValueError(
                f"frame_pool_type must be 'attention' or 'mean', got {frame_pool_type!r}"
            )

        _ = (
            normalize_landmarks,
            edge_importance_weighting,
            lstm_hidden_dim,
            lstm_layers,
            num_heads,
            use_motion_features,
            temporal_attention_layers,
            part_groups,
            add_gesture_edges,
            graph_edges,
            bidirectional,
        )

        gcn_dim = int(embed_dim if embed_dim is not None else stgcn_channels[0])
        encoder_depth = int(
            mamba_layers
            if mamba_layers is not None
            else (depth if depth is not None else len(stgcn_channels))
        )
        ratio = float(mlp_ratio if mlp_ratio is not None else 2.0)

        if gcn_dim <= 0:
            raise ValueError(f"embed_dim/gcn_dim must be positive, got {gcn_dim}")
        if use_mamba_encoder and encoder_depth <= 0:
            raise ValueError(f"mamba_layers/depth must be positive, got {encoder_depth}")

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.return_probabilities = return_probabilities
        self.gcn_dim = gcn_dim
        self.temporal_embed_dim = temporal_embed_dim
        self.gcn_top_k = gcn_top_k
        self.use_gcn = use_gcn
        self.use_mamba_encoder = use_mamba_encoder
        self.use_transcripts = use_transcripts
        self.transcript_embed_dim = transcript_embed_dim if use_transcripts else 0
        self.num_transcript_chunks = num_transcript_chunks if use_transcripts else 0
        self.max_transcript_chunks = max_transcript_chunks if use_transcripts else 0
        self.transcript_dropout_probability = transcript_dropout_probability
        self.use_temporal_attention_pool = use_temporal_attention_pool
        self.classifier_dim = temporal_embed_dim
        self.kinematic_feature_dim = KINEMATIC_FEATURE_DIM
        hidden_dim = temporal_embed_dim

        self.pose_feature_proj = nn.Sequential(
            nn.LayerNorm(gcn_dim),
            nn.Linear(gcn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.preprocessor = PoseFeaturePreprocessor(
            num_joints=num_joints,
            in_channels=in_channels,
        )
        node_in_dim = self.preprocessor.node_feature_dim

        if use_gcn:
            self.gcn_stack = EdgeAwareGraphConvStack(
                node_dim=node_in_dim,
                gcn_dim=gcn_dim,
                edge_dim=gcn_edge_dim,
                num_layers=gcn_layers,
                top_k=gcn_top_k,
                dropout=dropout,
            )
            self.kinematic_proj = None
        else:
            global _GCN_DISABLED_WARNED
            if not _GCN_DISABLED_WARNED:
                warnings.warn(
                    "GCN spatial encoder is disabled; using kinematic "
                    "(position, velocity, acceleration) embeddings -> pool "
                    "-> classifier pipeline only.",
                    stacklevel=2,
                )
                _GCN_DISABLED_WARNED = True
            self.gcn_stack = None
            self.kinematic_proj = nn.Sequential(
                nn.LayerNorm(KINEMATIC_FEATURE_DIM),
                nn.Linear(KINEMATIC_FEATURE_DIM, gcn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.frame_joint_pool = FrameJointPool(
            gcn_dim,
            num_joints=num_joints,
            pool_type=frame_pool_type,
        )

        if use_transcripts:
            self.transcript_encoder = TranscriptBERTEncoder(
                bert_model_name=bert_model_name,
                output_dim=transcript_embed_dim,
                dropout=dropout,
                freeze_bert=freeze_bert,
                max_length=bert_max_length,
            )
            self.text_value_projection = nn.Linear(transcript_embed_dim, hidden_dim)
            self.text_key_projection = nn.Linear(transcript_embed_dim, hidden_dim)
            self.pose_query_projection = nn.Linear(hidden_dim, hidden_dim)
            self.chunk_position_embedding = nn.Parameter(
                torch.zeros(1, max_transcript_chunks, hidden_dim)
            )
            self.text_context_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.zeros_(self.gate_projection.weight)
            nn.init.constant_(self.gate_projection.bias, INITIAL_GATE_LOGIT)
        else:
            self.transcript_encoder = None
            self.text_value_projection = None
            self.text_key_projection = None
            self.pose_query_projection = None
            self.chunk_position_embedding = None
            self.text_context_projection = None
            self.gate_projection = None

        if use_mamba_encoder:
            self.gcn_sample_proj = None
            self.temporal_proj = None
            self.temporal_encoder = MambaTemporalEncoder(
                d_model=temporal_embed_dim,
                num_layers=encoder_depth,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                temporal_kernel_size=temporal_kernel_size,
                expand=mamba_expand,
                mlp_ratio=ratio,
                dropout=dropout,
                fusion=mamba_fusion,
            )
            pool_dim = temporal_embed_dim
        else:
            global _MAMBA_DISABLED_WARNED
            if not _MAMBA_DISABLED_WARNED:
                if use_gcn:
                    pipeline = "GCN -> temporal pool -> classifier"
                else:
                    pipeline = (
                        "kinematic embeddings -> pool -> classifier"
                    )
                warnings.warn(
                    f"Mamba temporal encoder is disabled; using {pipeline} "
                    "pipeline only.",
                    stacklevel=2,
                )
                _MAMBA_DISABLED_WARNED = True
            self.temporal_proj = None
            self.temporal_encoder = None
            self.gcn_sample_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, temporal_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            pool_dim = temporal_embed_dim

        self.sample_pool = (
            TemporalAttentionPool(pool_dim)
            if use_temporal_attention_pool
            else None
        )
        _ = fc_hidden_dim
        self.classifier = GestureClassifierHead(
            temporal_embed_dim,
            num_classes,
            dropout=dropout,
        )

    def _prepare_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Canonicalize landmarks and resample time without crashing on T mismatch."""
        _validate_landmark_input(
            landmarks,
            num_joints=self.num_joints,
            in_channels=self.in_channels,
        )
        if (
            landmarks.ndim == 4
            and landmarks.shape[2] == self.num_joints
            and landmarks.shape[3] == self.in_channels
        ):
            landmarks = landmarks.contiguous()
        else:
            landmarks = canonicalize_landmarks(
                landmarks,
                num_joints=self.num_joints,
                in_channels=self.in_channels,
            )
        landmarks = _sanitize_landmarks(landmarks)
        _validate_canonical_landmarks(
            landmarks,
            num_joints=self.num_joints,
            in_channels=self.in_channels,
        )
        return self._resample_time(landmarks)

    def _encode_frame_tokens(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Return frame-level tokens with shape ``[B, T, gcn_dim]``."""
        landmarks = self._prepare_landmarks(landmarks)
        node_features = self.preprocessor(landmarks)
        if self.use_gcn:
            normalized_positions = node_features[..., :3]
            spatial = self.gcn_stack(node_features, normalized_positions)
        else:
            kinematic = node_features[..., : self.kinematic_feature_dim]
            spatial = self.kinematic_proj(kinematic)
        return self.frame_joint_pool(spatial)

    def _encode_pose_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Return pose sequence features with shape ``[B, T, hidden_dim]``."""
        frame_tokens = self._encode_frame_tokens(landmarks)
        return self.pose_feature_proj(frame_tokens)

    def _apply_modality_dropout(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a per-sample keep mask with shape ``[B, 1, 1]``."""
        if (
            not self.training
            or not self.use_transcripts
            or self.transcript_dropout_probability <= 0.0
        ):
            return torch.ones(batch_size, 1, 1, device=device, dtype=dtype)

        keep_mask = torch.rand(batch_size, device=device, dtype=dtype)
        keep_mask = (keep_mask >= self.transcript_dropout_probability).to(dtype)
        return keep_mask.view(batch_size, 1, 1)

    def _fuse_pose_transcript_chunks(
        self,
        pose_features: torch.Tensor,
        transcript_chunks: Sequence[Sequence[str]] | None,
        transcript_chunk_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Soft-align transcript chunks to pose timesteps and fuse with a gated residual.

        Returns:
            fused_pose: ``[B, T, hidden_dim]``
            attention_weights: ``[B, T, N]`` or ``None``
            modality_gate: ``[B, T, hidden_dim]`` or ``None``
            chunk_mask: ``[B, N]`` or ``None``
        """
        if not self.use_transcripts or self.transcript_encoder is None:
            return pose_features, None, None, None

        batch_size, num_frames, hidden_dim = pose_features.shape
        attention_weights: torch.Tensor | None = None
        modality_gate: torch.Tensor | None = None
        chunk_mask = transcript_chunk_mask

        if transcript_chunks is None:
            return pose_features, None, None, chunk_mask

        if len(transcript_chunks) != batch_size:
            raise ValueError(
                "Expected one transcript-chunk list per batch item, got "
                f"{len(transcript_chunks)} lists for batch_size={batch_size}."
            )

        num_chunks = len(transcript_chunks[0])
        if num_chunks != self.num_transcript_chunks:
            raise ValueError(
                "Expected transcript chunks with length "
                f"{self.num_transcript_chunks}, got {num_chunks}."
            )
        if num_chunks > self.max_transcript_chunks:
            raise ValueError(
                f"num_transcript_chunks={num_chunks} exceeds configured "
                f"max_transcript_chunks={self.max_transcript_chunks}."
            )

        chunk_embeddings = self.transcript_encoder.encode_chunks(transcript_chunks)
        assert chunk_embeddings.shape == (
            batch_size,
            num_chunks,
            self.transcript_embed_dim,
        )

        chunk_values = self.text_value_projection(chunk_embeddings)
        chunk_keys = self.text_key_projection(chunk_embeddings)
        position_embedding = self.chunk_position_embedding[:, :num_chunks, :]
        chunk_values = chunk_values + position_embedding
        chunk_keys = chunk_keys + position_embedding

        queries = self.pose_query_projection(pose_features)
        attention_logits = torch.matmul(
            queries,
            chunk_keys.transpose(-1, -2),
        ) / math.sqrt(hidden_dim)

        if transcript_chunk_mask is None:
            chunk_mask = torch.ones(
                batch_size,
                num_chunks,
                dtype=torch.bool,
                device=pose_features.device,
            )
        else:
            if transcript_chunk_mask.shape != (batch_size, num_chunks):
                raise ValueError(
                    "transcript_chunk_mask must have shape "
                    f"[{batch_size}, {num_chunks}], got "
                    f"{tuple(transcript_chunk_mask.shape)}."
                )
            chunk_mask = transcript_chunk_mask.to(
                device=pose_features.device,
                dtype=torch.bool,
            )

        valid_chunks = chunk_mask.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~valid_chunks, float("-inf"))
        sample_has_valid_chunks = valid_chunks.any(dim=-1, keepdim=True)

        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_weights = torch.where(
            sample_has_valid_chunks,
            attention_weights,
            torch.zeros_like(attention_weights),
        )
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        text_context = torch.matmul(attention_weights, chunk_values)

        gate_input = torch.cat([pose_features, text_context], dim=-1)
        modality_gate = torch.sigmoid(self.gate_projection(gate_input))

        transcript_active = sample_has_valid_chunks.to(dtype=pose_features.dtype)
        modality_keep = self._apply_modality_dropout(
            batch_size,
            device=pose_features.device,
            dtype=pose_features.dtype,
        )
        effective_gate = modality_gate * transcript_active * modality_keep

        if not sample_has_valid_chunks.all():
            text_context = text_context * transcript_active

        fused_pose = pose_features + effective_gate * self.text_context_projection(
            text_context
        )
        assert fused_pose.shape == pose_features.shape
        return fused_pose, attention_weights, modality_gate, chunk_mask

    def encode_temporal(
        self,
        landmarks: torch.Tensor,
        *,
        transcript_chunks: Sequence[Sequence[str]] | None = None,
        transcript_chunk_mask: torch.Tensor | None = None,
        transcripts: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return the sample-level embedding ``[B, temporal_embed_dim]``."""
        _ = transcripts
        pose_features = self._encode_pose_features(landmarks)
        fused_pose, _, _, _ = self._fuse_pose_transcript_chunks(
            pose_features,
            transcript_chunks,
            transcript_chunk_mask,
        )

        if self.use_mamba_encoder:
            encoded = self.temporal_encoder(fused_pose)
            if self.sample_pool is not None:
                sample_embedding = self.sample_pool(encoded)
            else:
                sample_embedding = encoded.mean(dim=1)
        else:
            if self.sample_pool is not None:
                pooled = self.sample_pool(fused_pose)
            else:
                pooled = fused_pose.mean(dim=1)
            sample_embedding = self.gcn_sample_proj(pooled)

        if sample_embedding.shape[-1] != self.temporal_embed_dim:
            raise RuntimeError(
                "Encoder returned unexpected embedding dimension: "
                f"expected {self.temporal_embed_dim}, got {sample_embedding.shape[-1]}"
            )
        if not torch.isfinite(sample_embedding).all():
            raise RuntimeError("Encoder produced non-finite sample embeddings.")
        return sample_embedding

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

    def forward(
        self,
        landmarks: torch.Tensor,
        *,
        transcript_chunks: Sequence[Sequence[str]] | None = None,
        transcript_chunk_mask: torch.Tensor | None = None,
        transcripts: Sequence[str] | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full gesture classification pipeline.

        Steps:
            1. Accept/canonicalize landmarks from training batches.
            2. Normalize pose and build kinematic node features.
            3. Apply GCN spatial encoding when ``use_gcn=True``; otherwise embed
               position/velocity/acceleration directly.
            4. Soft-align BERT-encoded transcript chunks to pose timesteps and
               fuse them with a gated residual before the temporal encoder.
            5. Pool over joints/time and classify.
               When ``use_mamba_encoder=True``, run the Mamba temporal encoder
               before pooling.
            6. Return raw logits ``[B, num_classes]`` for CrossEntropyLoss.
               Softmax is applied only for inference when
               ``return_probabilities=True`` and ``self.training`` is False.
        """
        _ = transcripts
        pose_features = self._encode_pose_features(landmarks)
        fused_pose, attention_weights, modality_gate, chunk_mask = (
            self._fuse_pose_transcript_chunks(
                pose_features,
                transcript_chunks,
                transcript_chunk_mask,
            )
        )

        if self.use_mamba_encoder:
            encoded = self.temporal_encoder(fused_pose)
            if self.sample_pool is not None:
                sample_embedding = self.sample_pool(encoded)
            else:
                sample_embedding = encoded.mean(dim=1)
        else:
            if self.sample_pool is not None:
                pooled = self.sample_pool(fused_pose)
            else:
                pooled = fused_pose.mean(dim=1)
            sample_embedding = self.gcn_sample_proj(pooled)

        logits = self.classifier(sample_embedding)
        logits = _validate_logits(logits, num_classes=self.num_classes)

        if return_attention:
            if attention_weights is None:
                batch_size = landmarks.shape[0]
                num_chunks = self.num_transcript_chunks
                attention_weights = landmarks.new_zeros(
                    batch_size,
                    self.num_frames,
                    num_chunks,
                )
            if modality_gate is None:
                modality_gate = landmarks.new_zeros(
                    pose_features.shape[0],
                    pose_features.shape[1],
                    self.temporal_embed_dim,
                )
            if chunk_mask is None:
                chunk_mask = torch.zeros(
                    landmarks.shape[0],
                    self.num_transcript_chunks,
                    dtype=torch.bool,
                    device=landmarks.device,
                )
            if self.return_probabilities:
                if self.training:
                    raise RuntimeError(
                        "return_probabilities=True cannot be used while training. "
                        "CrossEntropyLoss requires raw logits from the model."
                    )
                logits = F.softmax(logits, dim=-1)
            return logits, attention_weights, modality_gate, chunk_mask

        if self.return_probabilities:
            if self.training:
                raise RuntimeError(
                    "return_probabilities=True cannot be used while training. "
                    "CrossEntropyLoss requires raw logits from the model."
                )
            return F.softmax(logits, dim=-1)
        return logits


def _assert_no_self_neighbors(neighbor_idx: torch.Tensor) -> None:
    """Ensure top-k neighbor indices never include the center joint."""
    num_joints = neighbor_idx.shape[2]
    center_idx = torch.arange(num_joints, device=neighbor_idx.device)
    center_idx = center_idx.view(1, 1, num_joints, 1)
    if bool((neighbor_idx == center_idx).any()):
        raise AssertionError("neighbor_idx must not contain self-neighbors")


def _validate_chunk_alignment_model(
    *,
    batch_size: int = 2,
    num_frames: int = 16,
    num_joints: int = NUM_UPPER_BODY_LANDMARKS,
    in_channels: int = 4,
    num_classes: int = 2,
    num_transcript_chunks: int = 5,
    max_transcript_chunks: int = 16,
) -> None:
    """Validate soft pose-to-chunk alignment shapes, masking, and gradients."""
    hidden_dim = 128
    text_dim = 64
    landmarks = torch.randn(batch_size, num_frames, num_joints, in_channels)

    def _make_model(num_chunks: int) -> LandmarkGestureClassifier:
        return LandmarkGestureClassifier(
            num_classes=num_classes,
            num_frames=num_frames,
            num_joints=num_joints,
            in_channels=in_channels,
            temporal_embed_dim=hidden_dim,
            embed_dim=64,
            gcn_layers=2,
            gcn_top_k=4,
            use_gcn=True,
            use_mamba_encoder=True,
            mamba_layers=1,
            use_transcripts=True,
            transcript_embed_dim=text_dim,
            num_transcript_chunks=num_chunks,
            max_transcript_chunks=max_transcript_chunks,
            transcript_dropout_probability=0.0,
        )

    model = _make_model(num_transcript_chunks)
    model.eval()

    transcript_chunks = [
        [f"sample{b}_chunk{c}" for c in range(num_transcript_chunks)]
        for b in range(batch_size)
    ]
    transcript_chunk_mask = torch.tensor(
        [
            [True, True, False, True, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    pose_features = model._encode_pose_features(landmarks)
    assert pose_features.shape == (batch_size, num_frames, hidden_dim)

    fused_pose, attention_weights, modality_gate, chunk_mask = (
        model._fuse_pose_transcript_chunks(
            pose_features,
            transcript_chunks,
            transcript_chunk_mask,
        )
    )
    assert fused_pose.shape == pose_features.shape
    assert attention_weights is not None
    assert attention_weights.shape == (batch_size, num_frames, num_transcript_chunks)
    assert chunk_mask is not None
    assert chunk_mask.shape == (batch_size, num_transcript_chunks)
    assert torch.isfinite(attention_weights).all()
    assert torch.isfinite(fused_pose).all()

    invalid_chunk_attention = attention_weights[:, :, ~transcript_chunk_mask.any(dim=0)]
    if invalid_chunk_attention.numel() > 0:
        assert torch.allclose(
            invalid_chunk_attention,
            torch.zeros_like(invalid_chunk_attention),
        )

    valid_rows = transcript_chunk_mask.any(dim=-1)
    if bool(valid_rows.any()):
        valid_attention = attention_weights[valid_rows]
        valid_mask = transcript_chunk_mask[valid_rows]
        summed = valid_attention.sum(dim=-1)
        assert torch.allclose(
            summed,
            torch.ones_like(summed),
            atol=1e-5,
        )
        masked_invalid = valid_attention.masked_select(~valid_mask.unsqueeze(1))
        assert torch.allclose(masked_invalid, torch.zeros_like(masked_invalid))

    empty_row = 1
    assert torch.allclose(
        attention_weights[empty_row],
        torch.zeros_like(attention_weights[empty_row]),
    )
    assert torch.allclose(
        fused_pose[empty_row],
        pose_features[empty_row],
        atol=1e-5,
    )

    initial_gate = torch.sigmoid(model.gate_projection.bias.detach())
    assert torch.allclose(
        initial_gate,
        torch.full_like(initial_gate, INITIAL_GATE),
        atol=1e-4,
    )

    logits, attn_out, gate_out, mask_out = model(
        landmarks,
        transcript_chunks=transcript_chunks,
        transcript_chunk_mask=transcript_chunk_mask,
        return_attention=True,
    )
    assert logits.shape == (batch_size, num_classes)
    assert attn_out.shape == (batch_size, num_frames, num_transcript_chunks)
    assert gate_out.shape == (batch_size, num_frames, hidden_dim)
    assert mask_out.shape == (batch_size, num_transcript_chunks)
    assert attn_out.requires_grad
    assert gate_out.requires_grad

    model.train()
    model.transcript_dropout_probability = 0.0
    logits = model(
        landmarks,
        transcript_chunks=transcript_chunks,
        transcript_chunk_mask=transcript_chunk_mask,
    )
    loss = logits.sum()
    loss.backward()
    assert model.text_value_projection.weight.grad is not None
    assert model.text_key_projection.weight.grad is not None
    assert model.pose_query_projection.weight.grad is not None
    assert model.gate_projection.weight.grad is not None
    assert model.transcript_encoder.proj[1].weight.grad is not None

    alt_num_chunks = 3
    alt_model = _make_model(alt_num_chunks)
    alt_chunks = [
        [f"alt{b}_chunk{c}" for c in range(alt_num_chunks)]
        for b in range(batch_size)
    ]
    alt_mask = torch.ones(batch_size, alt_num_chunks, dtype=torch.bool)
    alt_logits = alt_model(
        landmarks,
        transcript_chunks=alt_chunks,
        transcript_chunk_mask=alt_mask,
    )
    assert alt_logits.shape == (batch_size, num_classes)


def _run_model_sanity_check(
    *,
    use_gcn: bool,
    batch_size: int,
    num_frames: int,
    num_joints: int,
    in_channels: int,
    num_classes: int,
) -> None:
    model = LandmarkGestureClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        gcn_top_k=8,
        gcn_layers=2,
        use_gcn=use_gcn,
        use_mamba_encoder=False,
    )

    landmarks = torch.randn(batch_size, num_frames, num_joints, in_channels)
    node_features = model.preprocessor(model._prepare_landmarks(landmarks))
    logits = model(landmarks)
    sample_embedding = model.encode_temporal(landmarks)
    frame_tokens = model._encode_frame_tokens(landmarks)

    print(f"use_gcn: {model.use_gcn}")
    print(f"node features: {tuple(node_features.shape)}")
    if model.use_gcn:
        neighbor_idx, edge_geometry = build_topk_graph(
            node_features[..., :3],
            top_k=model.gcn_top_k,
        )
        _assert_no_self_neighbors(neighbor_idx)
        inverse_distance = edge_geometry[..., 4]
        assert torch.isfinite(inverse_distance).all(), (
            "inverse_distance must be finite"
        )
        assert float(inverse_distance.max()) <= 1.0 + 1e-6, (
            f"inverse_distance must be <= 1.0, got max={float(inverse_distance.max())}"
        )

        _, edge_embed = model.gcn_stack.edge_embedding(
            node_features[..., :3],
            top_k=model.gcn_top_k,
        )
        gcn_out = model.gcn_stack(node_features, node_features[..., :3])
        print(f"neighbor idx: {tuple(neighbor_idx.shape)}")
        print(f"edge embed: {tuple(edge_embed.shape)}")
        print(f"gcn out: {tuple(gcn_out.shape)}")
        assert gcn_out.shape == (batch_size, num_frames, num_joints, model.gcn_dim)
    else:
        kinematic = node_features[..., : model.kinematic_feature_dim]
        embedded = model.kinematic_proj(kinematic)
        print(f"kinematic features: {tuple(kinematic.shape)}")
        print(f"kinematic embedding: {tuple(embedded.shape)}")

    print(f"frame tokens: {tuple(frame_tokens.shape)}")
    print(f"sample embedding: {tuple(sample_embedding.shape)}")
    print(f"logits: {tuple(logits.shape)}")

    assert logits.shape == (batch_size, num_classes)
    assert torch.isfinite(logits).all(), "logits must be finite raw values"
    assert frame_tokens.shape == (batch_size, num_frames, model.gcn_dim)
    assert sample_embedding.shape == (batch_size, model.temporal_embed_dim)

    short_landmarks = torch.randn(batch_size, 64, num_joints, in_channels)
    short_logits = model(short_landmarks)
    assert short_logits.shape == (batch_size, num_classes)
    assert torch.isfinite(short_logits).all()


def _main() -> None:
    batch_size = 2
    num_frames = 128
    num_joints = NUM_UPPER_BODY_LANDMARKS
    in_channels = 4
    num_classes = 2

    print("=== sanity check: use_gcn=False ===")
    _run_model_sanity_check(
        use_gcn=False,
        batch_size=batch_size,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        num_classes=num_classes,
    )

    print("\n=== sanity check: use_gcn=True ===")
    _run_model_sanity_check(
        use_gcn=True,
        batch_size=batch_size,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        num_classes=num_classes,
    )

    print("\n=== sanity check: chunk alignment ===")
    _validate_chunk_alignment_model(
        batch_size=batch_size,
        num_frames=16,
        num_joints=num_joints,
        in_channels=in_channels,
        num_classes=num_classes,
        num_transcript_chunks=5,
    )

    model = LandmarkGestureClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        use_gcn=True,
    )
    try:
        bad_joints = torch.randn(batch_size, num_frames, num_joints + 1, in_channels)
        model(bad_joints)
        raise AssertionError("Expected joint-count mismatch to raise ValueError")
    except ValueError as exc:
        print(f"joint mismatch check ok: {exc}")

    try:
        bad_batch = torch.randn(num_frames, num_joints, in_channels)
        model(bad_batch)
        raise AssertionError("Expected missing batch dimension to raise ValueError")
    except ValueError as exc:
        print(f"batch dimension check ok: {exc}")

    try:
        build_topk_graph(
            torch.randn(1, 4, 1, 3),
            top_k=1,
        )
        raise AssertionError("Expected num_joints < 2 to raise ValueError")
    except ValueError as exc:
        print(f"single-joint graph check ok: {exc}")


if __name__ == "__main__":
    _main()
