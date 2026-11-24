"""OpenPose motion encoding -> ST temporal selection -> Qwen2.5-VL + LoRA.

Default landmark order: BODY_25 (0:25), left hand (25:46), right hand (46:67).
Pass XY or actual XYZ coordinates, plus visibility/confidence separately.
Only A2 consumes continuous motion tokens and the transcript; A1/B/C are text.
The RGB/vision-encoder branch is unused. No optimizer or loss is defined here.

selector = TemporalFrameSelector(num_joints=67, num_frames=8)
llm = QwenIntentModel(num_frames=8, motion_dim=selector.motion_dim)
pipeline = AphasiaIntentPipeline(selector, llm)
result = pipeline(landmarks_xy, transcripts, target_jsons=targets,
                  visibility=openpose_confidence)
# Shift logits/labels/field_masks for causal CE in the training code.

Trainable: part input MLPs, temporal/cross-part Transformers, selector MLP,
motion projector, and language-attention LoRA. Base Qwen/text embeddings and
vision weights are frozen. New motion modules require training; untrained
motion vectors do not inherit the original RGB model's zero-shot alignment.

Generation/JSON parsing is discrete: downstream C loss does not backpropagate
through generated A2/B text. A2 caption CE trains the complete motion path.
Every selected time contributes 7 motion slots; 8 times give 56 slots.
Missing parts are masked. Selected contextual tokens can summarize other times
because temporal encoding precedes selection, as requested.

Requirements: torch, transformers==4.57.1, peft==0.17.1, accelerate.
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROMPT_ORDER = ("Prompt A1", "Prompt A2", "Prompt B", "Prompt C")
PROMPT_FIELDS = {
    "Prompt A1": ("conversation topic",),
    "Prompt A2": ("gesture movement description",),
    "Prompt B": ("gesture meaning",),
    "Prompt C": ("communicative intent", "reasoning", "gesture contribution"),
}
DEFAULT_PROMPTS = {
    "Prompt A1": "What is the topic of the conversation according to this transcript?",
    "Prompt A2": "Describe the participant's gesture movement from the ordered OpenPose motion "
    "tokens. Focus on left/right hand trajectories, finger articulation, palm "
    "orientation, and head movement. Use the transcript as speech context; describe only "
    "movement supported by the motion input. There are no RGB images or appearance "
    "cues.\n"
    "\n"
    "Transcript:\n"
    "<INPUT_TRANSCRIPT>",
    "Prompt B": "You are given the description of a gesture and the topic of the conversation when "
    "the gesture was being made. Infer the meaning of the gesture. The inferred meaning "
    "should be supported by both the gesture movement description and the conversation "
    "topic.",
    "Prompt C": "Infer the communicative intent of the person based on the meaning of the gesture, "
    "the topic of conversation, and the transcript. Support the intent with evidence in "
    "these inputs. Explain the reasoning behind the inferred communicative intent and how "
    "the gesture contributes to that intent.",
}
# Compatibility constants refer to the final stage only; use PROMPT_FIELDS for all.
INTENT_PROMPT = DEFAULT_PROMPTS["Prompt C"]
JSON_FIELDS = PROMPT_FIELDS["Prompt C"]


def openpose_person_to_tensors(
    person: Mapping[str, Any], *, device=None
) -> tuple[Tensor, Tensor]:
    """Convert one standard OpenPose person JSON to body/left/right order.

    Returns XY [67,2] and confidence [67]. Empty detection arrays are masked
    with confidence zero; malformed nonempty arrays raise an explicit error.
    No tracking/person selection is performed: the caller selects the person.
    """
    arrays = []
    for key, count in (
        ("pose_keypoints_2d", 25),
        ("hand_left_keypoints_2d", 21),
        ("hand_right_keypoints_2d", 21),
    ):
        values = person.get(key, [])
        if len(values) == 0:
            array = torch.zeros(count, 3, dtype=torch.float32, device=device)
        else:
            array = torch.as_tensor(values, dtype=torch.float32, device=device)
            if array.numel() != count * 3:
                raise ValueError(f"{key} must contain {count} (x,y,confidence) triples")
            array = array.reshape(count, 3)
        arrays.append(array)
    combined = torch.cat(arrays, dim=0)
    return combined[:, :2], combined[:, 2]


@dataclass(frozen=True)
class PartSpec:
    name: str
    kind: str
    joints: tuple[int, ...]
    anchor: tuple[int, ...]
    orientation: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class OpenPoseLayout:
    num_joints: int
    parts: tuple[PartSpec, ...]
    shoulders: tuple[int, int] = (5, 2)  # OpenPose left, right
    body_format: str = "BODY_25"

    @classmethod
    def body_hands(
        cls,
        body_format: str = "BODY_25",
        *,
        left_hand_start: int | None = None,
        right_hand_start: int | None = None,
        num_joints: int | None = None,
        face_indices: Sequence[int] = (),
    ) -> OpenPoseLayout:
        """Body, left hand (21), right hand (21) by default.

        BODY_25 gives 67 joints; COCO_18 gives 60. If face or other joints are
        included, provide the actual hand offsets and total joint count. These
        offsets describe YOUR array order, not OpenPose's separate JSON arrays.
        """
        if body_format not in ("BODY_25", "COCO_18"):
            raise ValueError(
                "body_format must be BODY_25 or COCO_18; define a custom OpenPoseLayout otherwise"
            )
        nbody = 25 if body_format == "BODY_25" else 18
        left = nbody if left_hand_start is None else left_hand_start
        right = left + 21 if right_hand_start is None else right_hand_start
        total = max(left + 21, right + 21, nbody, max(face_indices, default=-1) + 1)
        total = total if num_joints is None else num_joints
        if (
            left < nbody
            or right < nbody
            or set(range(left, left + 21)) & set(range(right, right + 21))
        ):
            raise ValueError(
                "Hand offsets must be distinct and outside the body landmark block"
            )
        head = (0, 15, 16, 17, 18) if nbody == 25 else (0, 14, 15, 16, 17)
        rear, lear = (17, 18) if nbody == 25 else (16, 17)
        parts = []
        for side, start, shoulder, elbow, wrist in (
            ("left", left, 5, 6, 7),
            ("right", right, 2, 3, 4),
        ):
            # Global hand/arm trajectory, finger articulation, and palm geometry
            # remain separate. Overlapping wrist/MCP landmarks are intentional.
            parts.extend(
                (
                    PartSpec(
                        side + "_hand",
                        "hand",
                        (shoulder, elbow, wrist, start),
                        (start,),
                    ),
                    PartSpec(
                        side + "_fingers",
                        "finger",
                        tuple(range(start + 1, start + 21)),
                        (start,),
                    ),
                    PartSpec(
                        side + "_palm",
                        "palm",
                        tuple(start + i for i in (0, 5, 9, 13, 17)),
                        (start,),
                        (start, start + 5, start + 17),
                    ),
                )
            )
        parts.append(
            PartSpec(
                "head",
                "head",
                tuple(dict.fromkeys((*head, *face_indices))),
                (1,),
                (rear, lear, 0),
            )
        )
        layout = cls(total, tuple(parts), (5, 2), body_format)
        layout.validate()
        return layout

    def validate(self) -> None:
        if type(self.num_joints) is not int or self.num_joints < 1 or not self.parts:
            raise ValueError(
                "Layout must have positive num_joints and at least one part"
            )
        if len({p.name for p in self.parts}) != len(self.parts):
            raise ValueError("Part names must be unique")
        used = list(self.shoulders)
        for part in self.parts:
            if (
                part.kind not in ("hand", "finger", "palm", "head")
                or not part.joints
                or not part.anchor
            ):
                raise ValueError(
                    "Every part needs a supported kind, joint indices, and anchor indices"
                )
            used.extend((*part.joints, *part.anchor, *(part.orientation or ())))
        if any(type(i) is not int or not 0 <= i < self.num_joints for i in used):
            raise ValueError("Layout contains joint indices outside num_joints")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MotionBatch:
    tokens: Tensor  # [B,T,P,D]; after selection T becomes M
    mask: Tensor  # [B,T,P], bool: observed anatomical part at that time
    times: Tensor  # [B,T], in caller's units (indices or seconds)
    part_names: tuple[str, ...]

    @property
    def shape(self):
        return self.tokens.shape

    @property
    def ndim(self):
        return self.tokens.ndim

    def detach(self) -> MotionBatch:
        return MotionBatch(
            self.tokens.detach(),
            self.mask.detach(),
            self.times.detach(),
            self.part_names,
        )

    def __getitem__(self, index) -> MotionBatch:
        # Batch-only indexing, preserving a batch dimension even for an integer.
        index = [index] if isinstance(index, int) else index
        return MotionBatch(
            self.tokens[index], self.mask[index], self.times[index], self.part_names
        )

    def to(self, device) -> MotionBatch:
        return MotionBatch(
            self.tokens.to(device),
            self.mask.to(device),
            self.times.to(device),
            self.part_names,
        )


def prepare_landmarks(
    landmarks: Tensor,
    layout: OpenPoseLayout,
    frame_mask: Tensor | None = None,
    landmark_mask: Tensor | None = None,
    visibility: Tensor | None = None,
    frame_times: Tensor | None = None,
    visibility_threshold: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validate observed values; zero missing/padded values before arithmetic."""
    if (
        landmarks.ndim != 4
        or landmarks.shape[2] != layout.num_joints
        or landmarks.shape[-1] not in (2, 3)
        or not landmarks.is_floating_point()
    ):
        raise ValueError(
            f"Expected floating XY/XYZ [B,T,{layout.num_joints},2 or 3]; confidence must be separate"
        )
    b, t, j, channels = landmarks.shape
    if b < 1 or t < 1:
        raise ValueError("Landmark batches and sequences must be nonempty")
    device = landmarks.device
    frames = (
        torch.ones(b, t, dtype=torch.bool, device=device)
        if frame_mask is None
        else frame_mask.to(device=device, dtype=torch.bool)
    )
    joints = (
        torch.ones(b, t, j, dtype=torch.bool, device=device)
        if landmark_mask is None
        else landmark_mask.to(device=device, dtype=torch.bool)
    )
    if frames.shape != (b, t) or joints.shape != (b, t, j):
        raise ValueError("Incorrect frame_mask or landmark_mask shape")
    joints = joints & frames[..., None]
    if visibility is not None:
        visibility = visibility.to(device)
        if visibility.shape != (b, t, j):
            raise ValueError("visibility must be [B,T,J]")
        valid_confidence = visibility[joints]
        if (
            not torch.isfinite(valid_confidence).all()
            or ((valid_confidence < 0) | (valid_confidence > 1)).any()
        ):
            raise ValueError("Observed OpenPose confidence must be finite and in [0,1]")
        joints = joints & (visibility > visibility_threshold)
    if not torch.isfinite(landmarks[joints]).all():
        raise ValueError(
            "Observed coordinates must be finite; mask missing coordinates explicitly"
        )
    xyz = torch.where(joints[..., None], landmarks, 0.0).float()
    if channels == 2:
        xyz = F.pad(xyz, (0, 1))  # planar geometry only; no inferred depth
    times = (
        torch.arange(t, device=device, dtype=torch.float32).expand(b, -1)
        if frame_times is None
        else frame_times.to(device=device, dtype=torch.float32)
    )
    if times.shape != (b, t):
        raise ValueError("frame_times must be [B,T]")
    for i in range(b):
        real = times[i, frames[i]]
        if not torch.isfinite(real).all() or (real[1:] <= real[:-1]).any():
            raise ValueError("Real timestamps must be finite and strictly increasing")
    return xyz, joints, frames, torch.where(frames, times, 0.0)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    return (values * mask[..., None]).sum(dim) / mask.sum(dim).clamp_min(1)[..., None]


def _safe_transformer(encoder: nn.Module, values: Tensor, valid: Tensor) -> Tensor:
    """Prevent all-masked attention rows from producing NaNs or fake tokens."""
    safe = valid.clone()
    empty = ~safe.any(-1)
    safe[empty, 0] = True
    values = encoder(values, src_key_padding_mask=~safe)
    return torch.where(valid[..., None], values, 0.0)


class PartWiseMotionEncoder(nn.Module):
    """Figure (a)/(b): per-part geometry -> temporal attention -> cross-part attention.

    Seven streams retain left/right identity. Temporal encoders are shared by
    left/right sides of the same anatomical category. Cross-part attention runs
    at each matched time, never across mismatched times. Body-relative positions
    preserve cross-hand distances; wrist-relative offsets capture articulation.
    Palm/head orientation features use only measured geometry. With XY input,
    out-of-plane palm/head rotation cannot be recovered from missing depth.
    """

    def __init__(
        self,
        layout: OpenPoseLayout | None = None,
        motion_dim: int = 256,
        num_heads: int = 8,
        temporal_layers: int = 2,
        cross_part_layers: int = 1,
        dropout: float = 0.1,
        visibility_threshold: float = 0.0,
    ):
        super().__init__()
        self.layout = layout or OpenPoseLayout.body_hands()
        self.layout.validate()
        if (
            type(motion_dim) is not int
            or motion_dim < 2
            or motion_dim % 2
            or type(num_heads) is not int
            or num_heads < 1
            or motion_dim % num_heads
        ):
            raise ValueError("motion_dim must be even and divisible by num_heads")
        if temporal_layers < 1 or cross_part_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Transformer depths must be positive and dropout in [0,1)")
        if not 0 <= visibility_threshold < 1:
            raise ValueError("visibility_threshold must be in [0,1)")
        self.motion_dim = motion_dim
        self.visibility_threshold = visibility_threshold
        self.part_names = tuple(p.name for p in self.layout.parts)
        self.part_inputs = nn.ModuleDict(
            {
                p.name: nn.Sequential(
                    nn.Linear(len(p.joints) * 10 + 7, motion_dim),
                    nn.GELU(),
                    nn.LayerNorm(motion_dim),
                )
                for p in self.layout.parts
            }
        )

        def block(depth):
            layer = nn.TransformerEncoderLayer(
                motion_dim,
                num_heads,
                4 * motion_dim,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                layer, depth, norm=nn.LayerNorm(motion_dim), enable_nested_tensor=False
            )

        self.temporal = nn.ModuleDict(
            {
                kind: block(temporal_layers)
                for kind in sorted({p.kind for p in self.layout.parts})
            }
        )
        self.cross_part = block(cross_part_layers)
        self.part_ids = nn.Embedding(len(self.layout.parts), motion_dim)
        frequencies = torch.exp(
            torch.arange(0, motion_dim, 2).float() * (-math.log(10000.0) / motion_dim)
        )
        self.register_buffer("time_frequencies", frequencies, persistent=False)

    def forward(
        self,
        landmarks: Tensor,
        *,
        frame_mask: Tensor | None = None,
        landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None,
        frame_times: Tensor | None = None,
    ) -> MotionBatch:
        xyz, valid, frames, times = prepare_landmarks(
            landmarks,
            self.layout,
            frame_mask,
            landmark_mask,
            visibility,
            frame_times,
            self.visibility_threshold,
        )
        b, t, _, _ = xyz.shape
        if (~frames.any(-1)).any():
            raise ValueError("Every clip needs a real frame")
        # Remove translation and clip-level body scale without erasing relative
        # left/right positions or replacing missing joints with observed zeros.
        l, r = self.layout.shoulders
        have_shoulders = valid[:, :, l] & valid[:, :, r]
        center = (xyz[:, :, l] + xyz[:, :, r]) / 2
        fallback = _masked_mean(xyz, valid, 2)
        center = torch.where(have_shoulders[..., None], center, fallback)
        lengths = (xyz[:, :, l] - xyz[:, :, r]).norm(dim=-1)
        scale = (lengths * have_shoulders).sum(-1) / have_shoulders.sum(-1).clamp_min(1)
        # A degenerate/no-shoulder clip retains its supplied coordinate scale.
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        pose = torch.where(
            valid[..., None],
            (xyz - center[:, :, None]) / scale[:, None, None, None],
            0.0,
        )
        velocity = torch.zeros_like(pose)
        adjacent = valid[:, 1:] & valid[:, :-1]
        dt = (times[:, 1:] - times[:, :-1]).clamp_min(1e-6)
        velocity[:, 1:] = torch.where(
            adjacent[..., None],
            (pose[:, 1:] - pose[:, :-1]) / dt[:, :, None, None],
            0.0,
        )
        first = times.masked_fill(~frames, torch.inf).amin(-1)
        last = times.masked_fill(~frames, -torch.inf).amax(-1)
        relative_time = (times - first[:, None]) / (last - first).clamp_min(1e-6)[
            :, None
        ]
        relative_time = torch.where(frames, relative_time, 0.0)
        angles = relative_time[..., None] * self.time_frequencies * 2 * math.pi
        time_ids = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        tokens, masks = [], []
        for index, part in enumerate(self.layout.parts):
            joint_ids = list(part.joints)
            anchor_ids = list(part.anchor)
            part_valid = valid[:, :, joint_ids]
            anchor_valid = valid[:, :, anchor_ids]
            anchor = _masked_mean(pose[:, :, anchor_ids], anchor_valid, 2)
            local = pose[:, :, joint_ids] - anchor[:, :, None]
            local_valid = part_valid & anchor_valid.any(-1)[..., None]
            local = torch.where(local_valid[..., None], local, 0.0)
            features = torch.cat(
                (
                    pose[:, :, joint_ids],
                    local,
                    velocity[:, :, joint_ids],
                    part_valid[..., None].float(),
                ),
                -1,
            ).flatten(-2)
            geometry = pose.new_zeros(b, t, 7)
            if part.orientation is not None:
                a, c, d = part.orientation
                axis = pose[:, :, c] - pose[:, :, a]
                other = pose[:, :, d] - pose[:, :, a]
                normal = torch.linalg.cross(axis, other)
                geometry_valid = (
                    valid[:, :, a]
                    & valid[:, :, c]
                    & valid[:, :, d]
                    & (axis.norm(dim=-1) > 1e-6)
                    & (normal.norm(dim=-1) > 1e-6)
                )
                geometry = torch.cat(
                    (
                        F.normalize(axis, dim=-1),
                        F.normalize(normal, dim=-1),
                        geometry_valid[..., None].float(),
                    ),
                    -1,
                )
                geometry = torch.where(geometry_valid[..., None], geometry, 0.0)
            features = torch.cat((features, geometry), -1)
            module = self.part_inputs[part.name]
            values = module(features.to(next(module.parameters()).dtype))
            observed = part_valid.any(-1) & frames
            values = (
                values + time_ids.to(values) + self.part_ids.weight[index].to(values)
            )
            values = _safe_transformer(self.temporal[part.kind], values, observed)
            tokens.append(values)
            masks.append(observed)
        tokens = torch.stack(tokens, 2)
        masks = torch.stack(masks, 2)
        p = len(self.layout.parts)
        tokens = _safe_transformer(
            self.cross_part, tokens.reshape(b * t, p, -1), masks.reshape(b * t, p)
        )
        return MotionBatch(tokens.reshape(b, t, p, -1), masks, times, self.part_names)


@dataclass
class FrameSelection:
    indices: Tensor
    selection_weights: Tensor
    soft_weights: Tensor
    scores: Tensor
    temporal_weights: Tensor
    selected_landmarks: Tensor
    selected_joint_mask: Tensor
    selected_times: Tensor
    source_times: Tensor
    source_joint_mask: Tensor
    source_frame_mask: Tensor
    selected_rgb: Tensor | None = None  # compatibility field; always None
    selected_motion: MotionBatch | None = None


class TemporalFrameSelector(nn.Module):
    """Encode all landmarks, then use the original MLP/ST selection mechanism.

    The scorer now receives a pooled motion encoding and its next-time
    difference (2*motion_dim inputs). Distinct hard choices and chronological
    sorting are preserved. The same ST weights select every anatomical stream.
    The encoder is owned here so selector.parameters()/state_dict() include it.
    """

    def __init__(
        self,
        num_joints: int = 67,
        num_frames: int = 8,
        hidden_dim: int = 256,
        temperature: float = 1.0,
        stochastic: bool = True,
        visibility_threshold: float = 0.0,
        motion_dim: int = 256,
        layout: OpenPoseLayout | None = None,
        num_heads: int = 8,
        temporal_layers: int = 2,
        cross_part_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if any(
            type(n) is not int or n < 1 for n in (num_joints, num_frames, hidden_dim)
        ):
            raise ValueError("Joint/frame/hidden sizes must be positive integers")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if layout is None:
            if num_joints not in (67, 60):
                raise ValueError(
                    "Supply an explicit OpenPoseLayout for layouts other than BODY_25+hands (67) or COCO_18+hands (60)"
                )
            layout = OpenPoseLayout.body_hands(
                "BODY_25" if num_joints == 67 else "COCO_18"
            )
        if layout.num_joints != num_joints:
            raise ValueError("layout.num_joints and num_joints disagree")
        self.num_joints, self.num_frames = num_joints, num_frames
        self.temperature, self.stochastic = temperature, stochastic
        self.visibility_threshold = visibility_threshold
        self.motion_dim = motion_dim
        self.encoder = PartWiseMotionEncoder(
            layout,
            motion_dim,
            num_heads,
            temporal_layers,
            cross_part_layers,
            dropout,
            visibility_threshold,
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * motion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        landmarks: Tensor,
        frame_mask: Tensor | None = None,
        landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None,
        frame_times: Tensor | None = None,
        temperature: float | None = None,
    ) -> FrameSelection:
        encoded = self.encoder(
            landmarks,
            frame_mask=frame_mask,
            landmark_mask=landmark_mask,
            visibility=visibility,
            frame_times=frame_times,
        )
        pose, joint_mask, frames, times = prepare_landmarks(
            landmarks,
            self.encoder.layout,
            frame_mask,
            landmark_mask,
            visibility,
            frame_times,
            self.visibility_threshold,
        )
        b, t, j, _ = pose.shape
        valid_frames = frames & encoded.mask.any(-1)
        if (valid_frames.sum(-1) < self.num_frames).any():
            raise ValueError(
                f"Need at least {self.num_frames} frames with observed motion parts in every clip"
            )
        pooled = _masked_mean(encoded.tokens, encoded.mask, 2)
        valid_diff = valid_frames[:, :-1] & valid_frames[:, 1:]
        diff = torch.where(valid_diff[..., None], pooled[:, 1:] - pooled[:, :-1], 0.0)
        diff = F.pad(diff, (0, 0, 0, 1))
        scores = (
            self.scorer(
                torch.cat((pooled, diff), -1).to(next(self.scorer.parameters()).dtype)
            )
            .squeeze(-1)
            .float()
        )
        scores = scores.masked_fill(~valid_frames, -torch.inf)
        tau = self.temperature if temperature is None else temperature
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be finite and positive")
        ranking = scores
        if self.training and self.stochastic:
            u = torch.rand_like(scores).clamp_(1e-6, 1 - 1e-6)
            ranking = scores - torch.log(-torch.log(u))
        available = valid_frames.clone()
        indices, rows, soft_rows = [], [], []
        for _ in range(self.num_frames):
            remaining = ranking.masked_fill(~available, -torch.inf)
            index = remaining.argmax(-1)
            hard = F.one_hot(index, num_classes=t).to(scores.dtype)
            soft = F.softmax(remaining / tau, -1)
            row = hard + (soft - soft.detach()) if self.training else hard
            indices.append(index)
            rows.append(row)
            soft_rows.append(soft)
            available = available & ~hard.bool()
        indices = torch.stack(indices, 1)
        order = indices.argsort(1)
        indices = indices.gather(1, order)
        weights = torch.stack(rows, 1).gather(1, order[..., None].expand(-1, -1, t))
        soft_weights = torch.stack(soft_rows, 1).gather(
            1, order[..., None].expand(-1, -1, t)
        )
        selected_pose = torch.einsum("bmt,btjc->bmjc", weights, pose)
        selected_joints = joint_mask.gather(1, indices[..., None].expand(-1, -1, j))
        selected_times = torch.einsum("bmt,bt->bm", weights, times)
        p = encoded.tokens.shape[2]
        selected_mask = encoded.mask.gather(1, indices[..., None].expand(-1, -1, p))
        selected_tokens = torch.einsum(
            "bmt,btpd->bmpd", weights.to(encoded.tokens.dtype), encoded.tokens
        )
        selected_tokens = torch.where(selected_mask[..., None], selected_tokens, 0.0)
        motion = MotionBatch(
            selected_tokens, selected_mask, selected_times, encoded.part_names
        )
        return FrameSelection(
            indices,
            weights,
            soft_weights,
            scores,
            F.softmax(scores / tau, -1),
            selected_pose,
            selected_joints,
            selected_times,
            times,
            joint_mask,
            frames,
            None,
            motion,
        )


@dataclass
class PreparedQwenBatch:
    model_inputs: dict[str, Tensor]
    labels: Tensor | None
    prompt_lengths: Tensor
    field_masks: dict[str, Tensor] | None = None


@dataclass
class PreparedHierarchy:
    prompts: dict[str, PreparedQwenBatch]
    responses: list[dict[str, dict[str, str]]] | None
    targets: dict[str, list[dict[str, str]]] | None
    conditioning: str


@dataclass
class PromptOutput:
    output_fields: tuple[str, ...]
    model_output: Any
    labels: Tensor | None
    field_masks: dict[str, Tensor] | None
    responses: list[dict[str, str]] | None

    @property
    def logits(self) -> Tensor:
        return self.model_output.logits


@dataclass
class HierarchicalOutput:
    prompts: dict[str, PromptOutput]
    responses: list[dict[str, dict[str, str]]] | None
    targets: dict[str, list[dict[str, str]]] | None
    conditioning: str
    selection: FrameSelection | None = None


class JSONGenerationError(ValueError):
    def __init__(
        self,
        raw_output: str,
        message: str = "Model did not generate the required JSON object",
        prompt_name: str | None = None,
        sample_index: int | None = None,
    ):
        super().__init__(
            f'{prompt_name or "Generation"}, sample {sample_index}: {message}'
        )
        self.raw_output = raw_output
        self.prompt_name = prompt_name
        self.sample_index = sample_index


class MotionScale(nn.Module):
    """Initialize projected token scale near text-embedding scale; learn it."""

    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.02)))

    def forward(self, values):
        return values * self.log_scale.exp().clamp(max=10.0)


class TextOnlyProcessor:
    """Native tokenizer/chat template with no image-processing dependency."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, **kwargs):
        # The older wrapper builds text content blocks for processor compatibility.
        plain = []
        for item in messages:
            content = item["content"]
            if isinstance(content, list):
                content = "".join(block["text"] for block in content)
            plain.append({**item, "content": content})
        return self.tokenizer.apply_chat_template(plain, **kwargs)

    def save_pretrained(self, directory):
        return self.tokenizer.save_pretrained(directory)


class QwenIntentModel(nn.Module):
    """One shared Qwen2.5-VL-7B model, language-attention LoRA, four prompts.
    prepare_inputs returns all four stage batches, not a single C batch.
    forward(prepared=...) returns all four supervised model outputs without loss.
    forward(**one_stage.model_inputs) remains available for stagewise training.
    generate_json returns every prompt's complete response JSON per sample.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        training_mode: str = "lora",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = True,
        num_frames: int = 8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        model: nn.Module | None = None,
        processor: Any | None = None,
        local_files_only: bool = False,
        output_fields: Sequence[str] = JSON_FIELDS,
        prompts_path: str | Path | None = None,
        motion_dim: int = 256,
        part_names: Sequence[str] | None = None,
    ):
        super().__init__()
        if training_mode != "lora":
            raise ValueError(
                "QwenIntentModel supports LoRA training only; training_mode must be 'lora'"
            )
        if (
            type(lora_rank) is not int
            or lora_rank <= 0
            or not math.isfinite(lora_alpha)
            or lora_alpha <= 0
            or not 0 <= lora_dropout < 1
        ):
            raise ValueError(
                "LoRA rank/alpha must be positive; dropout must be in [0,1)"
            )
        if tuple(output_fields) != JSON_FIELDS:
            raise ValueError(
                "The four-prompt architecture requires all C fields and all A1/A2/B fields"
            )
        self.output_fields = tuple(output_fields)
        self.prompts = self._load_prompts(prompts_path)
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor together, or neither")
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if dtype is None:
            dtype = (
                (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                if device.type == "cuda"
                else torch.float32
            )
        if model is None:
            try:
                from transformers import (
                    AutoTokenizer,
                    Qwen2_5_VLForConditionalGeneration,
                )
            except ImportError as exc:
                raise ImportError(
                    "Install transformers==4.57.1 and accelerate"
                ) from exc
            processor = TextOnlyProcessor(
                AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, local_files_only=local_files_only
                )
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=local_files_only,
            )
        self.model = model.to(device)
        self.processor = processor
        self.num_frames, self.training_mode = num_frames, training_mode
        self.motion_dim = motion_dim
        if type(motion_dim) is not int or motion_dim < 1:
            raise ValueError("motion_dim must be a positive integer")
        self.part_names = tuple(
            part_names or (p.name for p in OpenPoseLayout.body_hands().parts)
        )
        if not self.part_names or len(set(self.part_names)) != len(self.part_names):
            raise ValueError("part_names must be nonempty and unique")
        self.motion_token = "<|pose_motion|>"
        # Add the motion marker without replacing Qwen's existing special tokens.
        # add_tokens avoids the renamed add_special_tokens replacement keyword.
        added = self.processor.tokenizer.add_tokens(
            [self.motion_token], special_tokens=True
        )
        if self.model.get_input_embeddings().weight.shape[0] != len(
            self.processor.tokenizer
        ):
            self.model.resize_token_embeddings(
                len(self.processor.tokenizer), mean_resizing=False
            )
        self.motion_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.motion_token
        )
        self.model.requires_grad_(False)
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA training requires peft==0.17.1") from exc
        targets = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
            and ".self_attn." in name
            and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
            and ".visual." not in name
        ]
        if not targets:
            raise ValueError("Cannot find Qwen language attention projections for LoRA")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(
                ("lora_A." in name or "lora_B." in name) and ".visual." not in name
            )
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No trainable LoRA adapter parameters were created")
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        embedding_dim = self.model.get_input_embeddings().weight.shape[1]
        self.motion_projector = nn.Sequential(
            nn.LayerNorm(motion_dim),
            nn.Linear(motion_dim, motion_dim),
            nn.GELU(),
            nn.Linear(motion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            MotionScale(),
        ).to(device=device, dtype=dtype)
        self.train()

    @staticmethod
    def _load_prompts(path: str | Path | None) -> dict[str, str]:
        if path is None:
            return dict(DEFAULT_PROMPTS)
        with Path(path).open(encoding="utf-8") as handle:
            config = json.load(handle)
        prompts = {}
        for name in PROMPT_ORDER:
            item = config[name]
            if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(f"{name} needs a nonempty prompt string")
            if tuple(item.get("json_fields", ())) != PROMPT_FIELDS[name]:
                raise ValueError(f"{name} json_fields must be {PROMPT_FIELDS[name]}")
            prompts[name] = item["prompt"]
        return prompts

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _validate_json(obj: Any, fields: Sequence[str] = JSON_FIELDS) -> None:
        if (
            not isinstance(obj, dict)
            or set(obj) != set(fields)
            or any(not isinstance(obj[k], str) for k in fields)
        ):
            raise ValueError(
                f"Expected exactly these JSON string fields: {tuple(fields)}"
            )
        if any("<|" in obj[k] for k in fields):
            raise ValueError("Response contains a reserved Qwen control token")

    def _batch_objects(
        self, name: str, values: Sequence[Mapping[str, str] | str], batch_size: int
    ) -> list[dict[str, str]]:
        if isinstance(values, (str, bytes, Mapping)) or len(values) != batch_size:
            raise ValueError(f"{name} requires one response object per clip")
        result = []
        for value in values:
            obj = json.loads(value) if isinstance(value, str) else dict(value)
            self._validate_json(obj, PROMPT_FIELDS[name])
            result.append(obj)
        return result

    def _messages(
        self, name: str, inputs: Mapping[str, str], target: str | None = None
    ) -> list[dict]:
        replacements = {f"<INPUT_{key}>": value for key, value in inputs.items()}
        replacements["<INPUT_VIDEO_FRAMES>"] = (
            "(Pose motion tokens are supplied instead of RGB frames.)"
        )
        replacements["<INPUT_MOTION_TOKENS>"] = (
            "(Continuous motion vectors are supplied before the text.)"
        )
        template = self.prompts[name]

        def substitute(match):
            if match.group(0) not in replacements:
                raise ValueError(
                    f"{name}: unsupported input placeholder {match.group(0)}"
                )
            return replacements[match.group(0)]

        instruction = re.sub(r"<INPUT_[A-Z_]+>", substitute, template)
        for key, value in inputs.items():
            if f"<INPUT_{key}>" not in template:
                instruction += f'\n\n{key.replace("_"," ").title()}:\n{value}'
        instruction += (
            "\n\nReturn only a valid JSON object with exactly these string fields:\n"
            + json.dumps({field: "..." for field in PROMPT_FIELDS[name]})
        )
        if "<|" in instruction or (target is not None and "<|" in target):
            raise ValueError("Prompt/input/target contains a reserved control token")
        if name == "Prompt A2":
            # Placeholders only reserve positions. _prepare_stage replaces their
            # embeddings with projected continuous motion vectors, never text.
            instruction = (
                "Input modality: OpenPose motion vectors only; no RGB images are present. "
                "Tokens are ordered by selected time, then by anatomical stream: "
                + ", ".join(self.part_names)
                + ".\n"
                + self.motion_token * (self.num_frames * len(self.part_names))
                + "\n"
                + instruction
            )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        ]
        if target is not None:
            messages.append({"role": "assistant", "content": target})
        return messages

    def _render(
        self,
        name: str,
        messages: list[dict],
        image_tokens: int = 0,
        generation_prompt: bool = True,
    ) -> str:
        # image_tokens remains an unused compatibility argument for old callers.
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )

    def _prepare_stage(
        self,
        name: str,
        inputs: Sequence[Mapping[str, str]],
        motion: MotionBatch | None = None,
        targets: Sequence[Mapping[str, str] | str] | None = None,
    ) -> PreparedQwenBatch:
        if name not in PROMPT_FIELDS or not inputs:
            raise ValueError("Supply a valid prompt name and a nonempty input batch")
        b = len(inputs)
        if name == "Prompt A2":
            motion = self._motion(motion)
            if motion.shape[0] != b:
                raise ValueError("A2 motion and text batch sizes differ")
        elif motion is not None:
            raise ValueError("Only A2 accepts motion tokens")
        objects = None if targets is None else self._batch_objects(name, targets, b)
        fields = PROMPT_FIELDS[name]
        prompts, full_texts, field_ranges = [], [], []
        for i, stage_inputs in enumerate(inputs):
            if any(not isinstance(value, str) for value in stage_inputs.values()):
                raise TypeError("Prompt text inputs must be strings")
            prompts.append(
                self._render(
                    name, self._messages(name, stage_inputs), generation_prompt=True
                )
            )
            if objects is not None:
                target, boundaries = "{", []
                for k, field in enumerate(fields):
                    boundaries.append(len(target) if k else 0)
                    target += (
                        (", " if k else "")
                        + json.dumps(field)
                        + ": "
                        + json.dumps(objects[i][field], ensure_ascii=False)
                    )
                target += "}"
                full = self._render(
                    name,
                    self._messages(name, stage_inputs, target),
                    generation_prompt=False,
                )
                if not full.startswith(prompts[-1]):
                    raise ValueError(
                        "Chat template does not preserve the assistant prefix"
                    )
                full_texts.append(full)
                boundaries.append(len(full) - len(prompts[-1]))
                field_ranges.append(
                    [
                        (
                            len(prompts[-1]) + boundaries[k],
                            len(prompts[-1]) + boundaries[k + 1],
                        )
                        for k in range(len(fields))
                    ]
                )
        tokenizer = self.processor.tokenizer
        if objects is not None and not tokenizer.is_fast:
            raise ValueError("Field-specific training requires a fast tokenizer")
        tokens = tokenizer(
            full_texts if objects is not None else prompts,
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
            return_offsets_mapping=objects is not None,
        )
        ids, attention = tokens["input_ids"].to(self.device), tokens[
            "attention_mask"
        ].to(self.device)
        labels = None if objects is None else ids.clone()
        if labels is not None:
            labels[attention == 0] = -100
        lengths = []
        for i, prompt in enumerate(prompts):
            prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            active = ids[i, attention[i].bool()].tolist()
            if active[: len(prefix)] != prefix:
                raise ValueError(
                    "Prompt tokens are not an exact prefix of the target sequence"
                )
            lengths.append(len(prefix))
            if labels is not None:
                offset = ids.shape[1] - len(active)
                labels[i, offset : offset + len(prefix)] = -100
        field_masks = None
        if labels is not None:
            field_masks = {
                field: torch.zeros_like(labels, dtype=torch.bool) for field in fields
            }
            for i, ranges in enumerate(field_ranges):
                for position in (labels[i] != -100).nonzero().flatten().tolist():
                    start, end = tokens["offset_mapping"][i, position].tolist()
                    overlaps = [
                        max(0, min(end, hi) - max(start, lo)) for lo, hi in ranges
                    ]
                    owner = (
                        overlaps.index(max(overlaps))
                        if max(overlaps) > 0
                        else len(ranges) - 1
                    )
                    field_masks[fields[owner]][i, position] = True
        model_inputs = {"input_ids": ids, "attention_mask": attention}
        if name == "Prompt A2":
            slots = ids == self.motion_token_id
            n = self.num_frames * len(self.part_names)
            if not torch.all(slots.sum(-1) == n):
                raise ValueError(
                    "A2 must contain exactly one placeholder per selected time and part"
                )
            embedding = self.model.get_input_embeddings()
            text_vectors = embedding(ids)
            projector_dtype = next(self.motion_projector.parameters()).dtype
            projected = self.motion_projector(motion.tokens.to(projector_dtype)).to(
                text_vectors.dtype
            )
            # Masked-scatter preserves gradients through projector, ST selector,
            # temporal encoders and cross-part encoders into caption CE.
            replacement = projected.reshape(-1, projected.shape[-1])
            model_inputs["inputs_embeds"] = text_vectors.masked_scatter(
                slots[..., None], replacement
            )
            attention = attention.clone()
            attention[slots] = motion.mask.reshape(-1).to(attention.dtype)
            model_inputs["attention_mask"] = attention
            if labels is not None:
                labels[slots] = -100
        # Explicit 1-D text-style RoPE positions for soft motion prefixes. For
        # generation these are removed so Qwen updates cached positions itself.
        model_inputs["position_ids"] = (attention.long().cumsum(-1) - 1).clamp_min(0)
        return PreparedQwenBatch(
            model_inputs, labels, torch.tensor(lengths, device=self.device), field_masks
        )

    @staticmethod
    def _inputs_for(
        name: str,
        transcripts: Sequence[str],
        responses: Sequence[Mapping[str, Mapping[str, str]]],
    ) -> list[dict[str, str]]:
        if name in ("Prompt A1", "Prompt A2"):
            return [{"TRANSCRIPT": text} for text in transcripts]
        if len(responses) != len(transcripts):
            raise ValueError("Transcript/upstream response batch lengths differ")
        if name == "Prompt B":
            return [
                {
                    "GESTURE_MOVEMENT_DESCRIPTION": item["Prompt A2"][
                        "gesture movement description"
                    ],
                    "CONVERSATION_TOPIC": item["Prompt A1"]["conversation topic"],
                }
                for item in responses
            ]
        if name == "Prompt C":
            return [
                {
                    "GESTURE_MEANING": item["Prompt B"]["gesture meaning"],
                    "CONVERSATION_TOPIC": item["Prompt A1"]["conversation topic"],
                    "TRANSCRIPT": text,
                }
                for item, text in zip(responses, transcripts)
            ]
        raise ValueError(f"Unknown prompt {name}")

    def _validate_batch(self, motion: MotionBatch, transcripts: Sequence[str]) -> int:
        motion = self._motion(motion)
        b = motion.shape[0]
        if isinstance(transcripts, str) or len(transcripts) != b:
            raise ValueError("transcripts must contain one string per clip")
        if any(not isinstance(text, str) or "<|" in text for text in transcripts):
            raise ValueError(
                "Transcripts must be strings without reserved control tokens"
            )
        return b

    def prepare_inputs(
        self,
        motion: MotionBatch,
        transcripts: Sequence[str],
        target_jsons: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
        *,
        conditioning: str = "generated",
        response_jsons: Sequence[Mapping[str, Mapping[str, str]]] | None = None,
        max_new_tokens: int | Mapping[str, int] = 512,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> PreparedHierarchy:
        """Prepare all four prompts, assistant labels, and per-field token masks.
        Generated evidence is the default, including during training. Explicit
        target conditioning is available for teacher-forced upstream evidence.
        Responses and supervision targets are separate to avoid target leakage.
        No token truncation is performed. C's target is never included in its input.
        """
        b = self._validate_batch(motion, transcripts)
        if conditioning not in ("generated", "targets"):
            raise ValueError("conditioning must be 'generated' or 'targets'")
        targets = None
        if target_jsons is not None:
            if not isinstance(target_jsons, Mapping) or set(target_jsons) != set(
                PROMPT_ORDER
            ):
                raise ValueError(
                    f"target_jsons must be a mapping with all four keys: {PROMPT_ORDER}"
                )
            targets = {
                name: self._batch_objects(name, target_jsons[name], b)
                for name in PROMPT_ORDER
            }
        responses = None
        if conditioning == "targets":
            if targets is None or response_jsons is not None:
                raise ValueError(
                    "Target conditioning requires targets and no generated response_jsons"
                )
            evidence = [
                {name: targets[name][i] for name in PROMPT_ORDER} for i in range(b)
            ]
        else:
            if response_jsons is None:
                responses = self.generate_json(
                    motion,
                    transcripts,
                    max_new_tokens=max_new_tokens,
                    **dict(generation_kwargs or {}),
                )
            else:
                if (
                    isinstance(response_jsons, (str, bytes, Mapping))
                    or len(response_jsons) != b
                ):
                    raise ValueError(
                        "response_jsons needs one complete hierarchy per clip"
                    )
                responses = []
                for item in response_jsons:
                    if set(item) != set(PROMPT_ORDER):
                        raise ValueError(
                            "Each response hierarchy needs all four prompt keys"
                        )
                    copy = {}
                    for name in PROMPT_ORDER:
                        obj = dict(item[name])
                        self._validate_json(obj, PROMPT_FIELDS[name])
                        copy[name] = obj
                    responses.append(copy)
            evidence = responses
        prepared = {}
        for name in PROMPT_ORDER:
            prepared[name] = self._prepare_stage(
                name,
                self._inputs_for(name, transcripts, evidence),
                motion if name == "Prompt A2" else None,
                None if targets is None else targets[name],
            )
        return PreparedHierarchy(prepared, responses, targets, conditioning)

    def forward(self, prepared: PreparedHierarchy | None = None, **model_inputs):
        """Return four prompt outputs, or raw Qwen outputs for a single stage.
        For each PromptOutput, logits are [B,L,V]; labels are [B,L] with prompt
        and padding -100; masks are [B,L] by JSON field. For causal CE compare
        logits[:, :-1] against labels[:, 1:] and field_masks[field][:, 1:].
        Field masks partition the assistant JSON syntax/values and end token,
        matching the original wrapper's field-specific supervision convention.
        """
        if prepared is None:
            if "labels" in model_inputs:
                raise ValueError(
                    "Keep labels outside forward; calculate loss in train.py"
                )
            return self.model(**model_inputs, use_cache=False, return_dict=True)
        if model_inputs:
            raise ValueError("Supply either prepared=... or single-stage model inputs")
        outputs = {}
        for name in PROMPT_ORDER:
            batch = prepared.prompts[name]
            model_output = self.model(
                **batch.model_inputs, use_cache=False, return_dict=True
            )
            responses = (
                None
                if prepared.responses is None
                else [item[name] for item in prepared.responses]
            )
            outputs[name] = PromptOutput(
                PROMPT_FIELDS[name],
                model_output,
                batch.labels,
                batch.field_masks,
                responses,
            )
        return HierarchicalOutput(
            outputs, prepared.responses, prepared.targets, prepared.conditioning
        )

    def _generate_stage(
        self,
        name: str,
        inputs: Sequence[Mapping[str, str]],
        motion: MotionBatch | None,
        max_new_tokens: int,
        generation_kwargs: Mapping[str, Any],
    ) -> list[dict[str, str]]:
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        batch = self._prepare_stage(name, inputs, motion)
        opts = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        opts.update(generation_kwargs)
        if opts.get("num_return_sequences", 1) != 1 or opts.get(
            "return_dict_in_generate", False
        ):
            raise ValueError("Use one tensor generation result per sample")
        gen_inputs = {
            k: v for k, v in batch.model_inputs.items() if k != "position_ids"
        }
        generated = self.model.generate(**gen_inputs, **opts)
        continuation = generated[:, batch.model_inputs["input_ids"].shape[1] :]
        texts = self.processor.tokenizer.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(texts) != len(inputs):
            raise ValueError("Generation batch size changed unexpectedly")
        result = []
        for i, text in enumerate(texts):
            cleaned = text.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            try:
                obj = json.loads(cleaned)
                self._validate_json(obj, PROMPT_FIELDS[name])
            except (ValueError, TypeError) as exc:
                raise JSONGenerationError(
                    text, prompt_name=name, sample_index=i
                ) from exc
            result.append(obj)
        return result

    @torch.no_grad()
    def generate_json(
        self,
        motion: MotionBatch,
        transcripts: Sequence[str],
        max_new_tokens: int | Mapping[str, int] = 512,
        **generation_kwargs,
    ) -> list[dict[str, dict[str, str]]]:
        """Generate A1, A2, B, C in order and return all complete response JSONs.
        A1 <- transcript; A2 <- selected motion tokens + transcript; B <- A2 description + A1 topic;
        C <- B meaning + A1 topic + transcript. A1/A2 do not see each other's
        response. No annotation is used. JSON errors include stage/sample/raw text.
        Restores the model's previous training/eval mode even if a stage fails.
        """
        b = self._validate_batch(motion, transcripts)
        responses = [{} for _ in range(b)]
        was_training = self.training
        self.eval()
        try:
            for name in PROMPT_ORDER:
                limit = (
                    max_new_tokens[name]
                    if isinstance(max_new_tokens, Mapping)
                    else max_new_tokens
                )
                objects = self._generate_stage(
                    name,
                    self._inputs_for(name, transcripts, responses),
                    motion if name == "Prompt A2" else None,
                    limit,
                    generation_kwargs,
                )
                for item, obj in zip(responses, objects):
                    item[name] = obj
            return responses
        finally:
            self.train(was_training)

    def generate_text(
        self, motion: MotionBatch, transcripts: Sequence[str], **kwargs
    ) -> list[dict[str, str]]:
        """Return each stage's validated response serialized as JSON text.
        Unlike the original single-C API this returns four strings per sample.
        Use generate_json for structured fields; neither method calculates loss.
        """
        return [
            {name: json.dumps(obj[name], ensure_ascii=False) for name in PROMPT_ORDER}
            for obj in self.generate_json(motion, transcripts, **kwargs)
        ]

    def _motion(self, motion: MotionBatch) -> MotionBatch:
        if not isinstance(motion, MotionBatch):
            raise TypeError(
                "A2 requires selector(...).selected_motion (MotionBatch), not RGB or flattened coordinates"
            )
        b, m, p, d = motion.tokens.shape
        if (
            b < 1
            or m != self.num_frames
            or p != len(self.part_names)
            or d != self.motion_dim
            or motion.part_names != self.part_names
            or motion.mask.shape != (b, m, p)
            or motion.times.shape != (b, m)
        ):
            raise ValueError(
                "Motion shape/part order differs from the configured selector and LVLM"
            )
        if motion.mask.dtype != torch.bool or not motion.mask.any(-1).all():
            raise ValueError(
                "Each selected time needs at least one observed anatomical part"
            )
        if (
            not torch.isfinite(motion.tokens).all()
            or not torch.isfinite(motion.times).all()
        ):
            raise ValueError("Motion tokens and timestamps must be finite")
        return motion.to(self.device)

    def save_motion_interface(self, directory: str | Path) -> None:
        """LoRA save_pretrained does not cover this separately owned projector."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.motion_projector.state_dict(), directory / "projector.pt")
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "motion_dim": self.motion_dim,
                    "part_names": self.part_names,
                    "num_frames": self.num_frames,
                    "motion_token": self.motion_token,
                },
                indent=2,
            )
        )

    def load_motion_interface(self, directory: str | Path) -> None:
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if (
            config["motion_dim"] != self.motion_dim
            or tuple(config["part_names"]) != self.part_names
            or config["num_frames"] != self.num_frames
            or config["motion_token"] != self.motion_token
        ):
            raise ValueError(
                "Saved motion interface configuration differs from current model"
            )
        self.motion_projector.load_state_dict(
            torch.load(
                directory / "projector.pt", map_location=self.device, weights_only=True
            )
        )


class AphasiaIntentPipeline(nn.Module):
    """OpenPose encoding -> chronological ST selection -> Qwen motion prefix."""

    def __init__(self, selector: TemporalFrameSelector, llm: QwenIntentModel):
        super().__init__()
        if (
            selector.num_frames != llm.num_frames
            or selector.motion_dim != llm.motion_dim
            or selector.encoder.part_names != llm.part_names
        ):
            raise ValueError(
                "Encoder, selector and LVLM motion dimensions/part order must agree"
            )
        self.selector, self.llm = selector, llm

    def forward(
        self,
        landmarks: Tensor,
        transcripts: Sequence[str],
        target_jsons: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
        *,
        frame_mask: Tensor | None = None,
        landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None,
        frame_times: Tensor | None = None,
        temperature: float | None = None,
        **prepare_kwargs,
    ) -> HierarchicalOutput:
        selection = self.selector(
            landmarks,
            frame_mask=frame_mask,
            landmark_mask=landmark_mask,
            visibility=visibility,
            frame_times=frame_times,
            temperature=temperature,
        )
        prepared = self.llm.prepare_inputs(
            selection.selected_motion, transcripts, target_jsons, **prepare_kwargs
        )
        result = self.llm(prepared=prepared)
        result.selection = selection
        return result

    @torch.no_grad()
    def generate_json(
        self,
        landmarks: Tensor,
        transcripts: Sequence[str],
        *,
        frame_mask: Tensor | None = None,
        landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None,
        frame_times: Tensor | None = None,
        **generation_kwargs,
    ) -> list[dict[str, dict[str, str]]]:
        was_training = self.training
        self.eval()
        try:
            selection = self.selector(
                landmarks,
                frame_mask=frame_mask,
                landmark_mask=landmark_mask,
                visibility=visibility,
                frame_times=frame_times,
            )
            return self.llm.generate_json(
                selection.selected_motion, transcripts, **generation_kwargs
            )
        finally:
            self.train(was_training)

    def save_trainable(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.selector.state_dict(), directory / "selector_encoder.pt")
        (directory / "layout.json").write_text(
            json.dumps(self.selector.encoder.layout.to_dict(), indent=2)
        )
        self.llm.model.save_pretrained(directory / "qwen_adapter")
        self.llm.processor.save_pretrained(directory / "qwen_adapter")
        self.llm.save_motion_interface(directory / "motion_interface")

    def load_trainable(self, directory: str | Path) -> None:
        """Restore a save_trainable checkpoint into the same architecture/config."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        directory = Path(directory)
        saved_layout = json.loads((directory / "layout.json").read_text())
        current_layout = json.loads(json.dumps(self.selector.encoder.layout.to_dict()))
        if saved_layout != current_layout:
            raise ValueError(
                "Saved OpenPose joint layout differs from the current layout"
            )
        self.selector.load_state_dict(
            torch.load(
                directory / "selector_encoder.pt",
                map_location=next(self.selector.parameters()).device,
                weights_only=True,
            )
        )
        adapter_file = directory / "qwen_adapter" / "adapter_model.safetensors"
        if adapter_file.is_file():
            adapter = load_file(str(adapter_file), device=str(self.llm.device))
        else:
            adapter = torch.load(
                directory / "qwen_adapter" / "adapter_model.bin",
                map_location=self.llm.device,
                weights_only=True,
            )
        set_peft_model_state_dict(self.llm.model, adapter)
        self.llm.load_motion_interface(directory / "motion_interface")
