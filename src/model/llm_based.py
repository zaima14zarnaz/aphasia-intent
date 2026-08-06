"""
LLaMA-style continuous-pose gesture classifier.

This model is designed as a drop-in POS/NEG classifier for the current
``train.py`` loop:

    logits = model(batch["landmarks"])

Expected input shape is ``[B, T, J, C]`` where the current dataloader returns
upper-body landmarks with ``J=8`` and ``C=4``. The model avoids VQ-VAE
tokenization for now: normalized continuous pose, velocity, acceleration, and
joint-validity features are encoded by a temporal transformer and projected
into LLaMA's embedding space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from transformers import LlamaConfig, LlamaModel
except ImportError as exc:  # pragma: no cover - dependency is environment-specific.
    LlamaConfig = None  # type: ignore[assignment]
    LlamaModel = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


@dataclass(frozen=True)
class PoseLlamaGestureConfig:
    num_classes: int = 2
    num_frames: int = 128
    num_joints: int = 8
    in_channels: int = 4
    pose_feature_dim: int = 12
    pose_hidden_dim: int = 256
    pose_encoder_layers: int = 4
    pose_encoder_heads: int = 8
    pose_dropout: float = 0.1
    llama_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_pretrained_llama: bool = False
    local_files_only: bool = False
    hf_token: str | None = None
    freeze_llama: bool = True
    llama_torch_dtype: str | None = "float16"
    topk_ratio: float = 0.15
    topk_min: int = 1
    classifier_dropout: float = 0.2
    eps: float = 1e-6


class PosePreprocessor(nn.Module):
    """
    Apply gesture-oriented preprocessing inside the model.

    The dataloader already selects upper-body joints in this order:
    left shoulder, right shoulder, left elbow, right elbow, left wrist,
    right wrist, left hip, right hip.

    The output per joint contains:
        normalized xyz/confidence, velocity xyz/confidence,
        acceleration xyz, valid-joint flag
    for a total of 11 channels by default.
    """

    def __init__(
        self,
        *,
        left_shoulder_idx: int = 0,
        right_shoulder_idx: int = 1,
        left_hip_idx: int = 6,
        right_hip_idx: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.left_shoulder_idx = left_shoulder_idx
        self.right_shoulder_idx = right_shoulder_idx
        self.left_hip_idx = left_hip_idx
        self.right_hip_idx = right_hip_idx
        self.eps = eps

    def forward(self, landmarks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if landmarks.ndim != 4:
            raise ValueError(
                "Expected landmarks with shape [B, T, J, C], "
                f"got {tuple(landmarks.shape)}"
            )
        if landmarks.shape[-1] < 2:
            raise ValueError("Expected at least x/y landmark channels.")

        x = landmarks.float()
        finite = torch.isfinite(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        coords = x[..., :3]
        if coords.shape[-1] < 3:
            pad = coords.new_zeros(*coords.shape[:-1], 3 - coords.shape[-1])
            coords = torch.cat([coords, pad], dim=-1)

        if x.shape[-1] >= 4:
            confidence = x[..., 3:4].clamp(min=0.0, max=1.0)
            valid = finite[..., :2].all(dim=-1, keepdim=True) & (confidence > 0.0)
        else:
            confidence = finite[..., :2].all(dim=-1, keepdim=True).float()
            valid = confidence > 0.0

        valid_float = valid.float()
        coords = coords * valid_float

        left_shoulder = coords[:, :, self.left_shoulder_idx]
        right_shoulder = coords[:, :, self.right_shoulder_idx]
        left_hip = coords[:, :, self.left_hip_idx]
        right_hip = coords[:, :, self.right_hip_idx]

        shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
        hip_mid = 0.5 * (left_hip + right_hip)
        root = 0.5 * (shoulder_mid + hip_mid)

        shoulder_width = (left_shoulder[..., :2] - right_shoulder[..., :2]).norm(
            dim=-1,
            keepdim=True,
        )
        torso_length = (shoulder_mid[..., :2] - hip_mid[..., :2]).norm(
            dim=-1,
            keepdim=True,
        )
        scale = torch.maximum(shoulder_width, torso_length).clamp_min(self.eps)

        norm_coords = coords - root.unsqueeze(2)
        norm_coords_xy = norm_coords[..., :2] / scale.unsqueeze(2)
        norm_coords_z = norm_coords[..., 2:3] / scale.unsqueeze(2)
        norm_coords = torch.cat([norm_coords_xy, norm_coords_z], dim=-1)
        norm_coords = norm_coords * valid_float

        velocity = torch.zeros_like(norm_coords)
        velocity[:, 1:] = norm_coords[:, 1:] - norm_coords[:, :-1]
        velocity = velocity * valid_float

        acceleration = torch.zeros_like(norm_coords)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        acceleration = acceleration * valid_float

        features = torch.cat(
            [
                norm_coords,
                confidence * valid_float,
                velocity,
                confidence * valid_float,
                acceleration,
                valid_float,
            ],
            dim=-1,
        )
        frame_valid = valid.any(dim=2).squeeze(-1)
        return features, frame_valid


class SinusoidalTemporalPosition(nn.Module):
    """Fixed temporal position encoding for pose tokens."""

    def __init__(self, hidden_dim: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max position length "
                f"{self.pe.shape[1]}."
            )
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class ContinuousPoseEncoder(nn.Module):
    """Encode normalized continuous landmarks into temporal pose tokens."""

    def __init__(
        self,
        *,
        num_joints: int,
        pose_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_joints * pose_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_encoding = SinusoidalTemporalPosition(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        pose_features: torch.Tensor,
        frame_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_joints, feature_dim = pose_features.shape
        x = pose_features.reshape(batch_size, num_frames, num_joints * feature_dim)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        src_key_padding_mask = None
        if frame_valid is not None:
            src_key_padding_mask = ~frame_valid.bool()

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.out_norm(x)


class ContinuousPoseLlamaGestureClassifier(nn.Module):
    """
    Continuous pose -> temporal transformer -> LLaMA embedding space -> POS/NEG.

    The LLaMA sequence is assembled as:
        [TASK_SOFT] [POSE_1] ... [POSE_T] [GEST_LABEL_TOKEN_SOFT]

    Since LLaMA is decoder-only and causal, the gesture-label token is placed at
    the end so its hidden state can attend to the whole pose-token window.

    Classification combines:
        1. hidden state at the learned gesture-label token,
        2. validity-masked mean context over pose-token hidden states,
        3. MIL/top-k pooled temporal evidence over pose-token frame logits.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_frames: int = 128,
        num_joints: int = 8,
        in_channels: int = 4,
        *,
        config: PoseLlamaGestureConfig | None = None,
        **overrides,
    ) -> None:
        super().__init__()
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ImportError(
                "llm_based.py requires the 'transformers' package to use LLaMA."
            ) from _TRANSFORMERS_IMPORT_ERROR

        # Backward-compatible aliases for train scripts written for the BERT
        # version of this file.
        if "bert_model_name" in overrides and "llama_model_name" not in overrides:
            overrides["llama_model_name"] = overrides.pop("bert_model_name")
        if "use_pretrained_bert" in overrides and "use_pretrained_llama" not in overrides:
            overrides["use_pretrained_llama"] = overrides.pop("use_pretrained_bert")
        if "freeze_bert" in overrides and "freeze_llama" not in overrides:
            overrides["freeze_llama"] = overrides.pop("freeze_bert")

        cfg_kwargs = {
            "num_classes": num_classes,
            "num_frames": num_frames,
            "num_joints": num_joints,
            "in_channels": in_channels,
        }
        cfg_kwargs.update(overrides)
        self.config = config or PoseLlamaGestureConfig(**cfg_kwargs)

        self.preprocessor = PosePreprocessor(eps=self.config.eps)
        self.pose_encoder = ContinuousPoseEncoder(
            num_joints=self.config.num_joints,
            pose_feature_dim=self.config.pose_feature_dim,
            hidden_dim=self.config.pose_hidden_dim,
            num_layers=self.config.pose_encoder_layers,
            num_heads=self.config.pose_encoder_heads,
            dropout=self.config.pose_dropout,
        )

        if self.config.use_pretrained_llama:
            self.llama = LlamaModel.from_pretrained(
                self.config.llama_model_name,
                local_files_only=self.config.local_files_only,
                token=self.config.hf_token,
                torch_dtype=self._resolve_llama_dtype(self.config.llama_torch_dtype),
            )
        else:
            llama_config = LlamaConfig(
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                num_key_value_heads=12,
                intermediate_size=3072,
                max_position_embeddings=max(512, self.config.num_frames + 8),
                hidden_dropout_prob=self.config.pose_dropout,
                attention_dropout=self.config.pose_dropout,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            self.llama = LlamaModel(llama_config)

        llama_dim = int(self.llama.config.hidden_size)
        self.pose_to_llama = nn.Sequential(
            nn.Linear(self.config.pose_hidden_dim, llama_dim),
            nn.LayerNorm(llama_dim),
            nn.GELU(),
            nn.Dropout(self.config.pose_dropout),
            nn.Linear(llama_dim, llama_dim),
        )
        self.task_token = nn.Parameter(torch.zeros(1, 1, llama_dim))
        self.gesture_label_token = nn.Parameter(torch.zeros(1, 1, llama_dim))

        self.frame_score_head = nn.Linear(llama_dim, self.config.num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(llama_dim * 3),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(llama_dim * 3, llama_dim),
            nn.GELU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(llama_dim, self.config.num_classes),
        )

        self._init_special_tokens()
        if self.config.freeze_llama:
            self.freeze_llama()

    @staticmethod
    def _resolve_llama_dtype(dtype_name: str | None) -> torch.dtype | str | None:
        if dtype_name is None:
            return None
        normalized = dtype_name.lower()
        if normalized == "auto":
            return "auto"
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(
            "llama_torch_dtype must be one of: None, 'auto', 'float16', "
            "'bfloat16', or 'float32'."
        )

    def _init_special_tokens(self) -> None:
        nn.init.normal_(self.task_token, mean=0.0, std=0.02)
        nn.init.normal_(self.gesture_label_token, mean=0.0, std=0.02)

    def freeze_llama(self) -> None:
        for parameter in self.llama.parameters():
            parameter.requires_grad = False

    def unfreeze_llama(self) -> None:
        for parameter in self.llama.parameters():
            parameter.requires_grad = True

    def freeze_bert(self) -> None:
        self.freeze_llama()

    def unfreeze_bert(self) -> None:
        self.unfreeze_llama()

    def _topk_pool(
        self,
        frame_logits: torch.Tensor,
        frame_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_valid.bool()
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid] = True

        frame_logits = torch.nan_to_num(
            frame_logits.float(),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        num_frames = frame_logits.shape[1]
        k = max(
            self.config.topk_min,
            int(round(num_frames * self.config.topk_ratio)),
        )
        k = min(k, num_frames)

        masked_logits = frame_logits.masked_fill(
            ~valid.unsqueeze(-1),
            -20.0,
        )
        topk_values = masked_logits.topk(k=k, dim=1).values
        pooled = topk_values.mean(dim=1)
        return torch.nan_to_num(pooled, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(
        self,
        landmarks: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if landmarks.shape[1] > self.config.num_frames:
            raise ValueError(
                f"Input has T={landmarks.shape[1]} frames, but this model was "
                f"configured for num_frames={self.config.num_frames}."
            )

        pose_features, frame_valid = self.preprocessor(landmarks)
        encoder_frame_valid = frame_valid.clone()
        all_invalid = ~encoder_frame_valid.any(dim=1)
        if all_invalid.any():
            encoder_frame_valid[all_invalid] = True

        pose_tokens = self.pose_encoder(
            pose_features,
            frame_valid=encoder_frame_valid,
        )
        pose_embeds = self.pose_to_llama(pose_tokens)

        batch_size = pose_embeds.shape[0]
        task = self.task_token.expand(batch_size, -1, -1)
        label_token = self.gesture_label_token.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([task, pose_embeds, label_token], dim=1)

        special_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=landmarks.device,
        )
        attention_mask = torch.cat(
            [special_mask, frame_valid.bool(), special_mask],
            dim=1,
        ).long()

        llama_dtype = next(self.llama.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=llama_dtype)

        llama_outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        head_dtype = next(self.frame_score_head.parameters()).dtype
        sequence_output = llama_outputs.last_hidden_state.to(dtype=head_dtype)
        sequence_output = torch.nan_to_num(
            sequence_output,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        pose_start = task.shape[1]
        pose_end = pose_start + pose_embeds.shape[1]
        pose_hidden = sequence_output[:, pose_start:pose_end]
        label_hidden = sequence_output[:, -1]

        frame_logits = self.frame_score_head(pose_hidden)
        frame_logits = torch.nan_to_num(
            frame_logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        mil_logits = self._topk_pool(frame_logits, frame_valid)
        mil_logits = mil_logits.to(dtype=head_dtype)
        mil_context = mil_logits @ self.frame_score_head.weight
        valid_float = frame_valid.unsqueeze(-1).to(dtype=pose_hidden.dtype)
        pose_context = (pose_hidden * valid_float).sum(dim=1) / valid_float.sum(
            dim=1
        ).clamp_min(1.0)
        pose_context = torch.nan_to_num(
            pose_context,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        logits = self.classifier(
            torch.cat([label_hidden, pose_context, mil_context], dim=-1)
        )
        logits = logits + mil_logits
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        if return_details:
            return {
                "logits": logits,
                "frame_logits": frame_logits,
                "mil_logits": mil_logits,
                "frame_valid": frame_valid,
                "label_hidden": label_hidden,
                "pose_context": pose_context,
            }
        return logits


# Alias with the same naming style as the existing LandmarkGestureClassifier.
LLMBasedGestureClassifier = ContinuousPoseLlamaGestureClassifier


def build_llm_based_gesture_classifier(
    *,
    num_classes: int = 2,
    num_frames: int = 128,
    num_joints: int = 8,
    in_channels: int = 4,
    **kwargs,
) -> ContinuousPoseLlamaGestureClassifier:
    return ContinuousPoseLlamaGestureClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        **kwargs,
    )


if __name__ == "__main__":
    model = ContinuousPoseLlamaGestureClassifier(
        num_classes=2,
        num_frames=128,
        num_joints=8,
        in_channels=4,
        use_pretrained_llama=False,
    )
    dummy_landmarks = torch.randn(2, 128, 8, 4)
    dummy_logits = model(dummy_landmarks)
    print(f"logits shape: {tuple(dummy_logits.shape)}")

