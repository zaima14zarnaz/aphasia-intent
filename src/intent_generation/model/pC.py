"""Landmark-driven frame selection and a trainable Qwen2.5-VL wrapper.

See TRAINING_USAGE.md. No optimizer, training loop, or loss calculation lives here.
Training uses forward logits; generate_json() returns complete JSON objects.
Validated against transformers==4.57.1 and peft==0.17.1.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "C.json"
with _PROMPT_PATH.open(encoding="utf-8") as _prompt_file:
    _prompt_c = json.load(_prompt_file)["Prompt C"]
INTENT_PROMPT = _prompt_c["prompt"]
JSON_FIELDS = tuple(_prompt_c["json_fields"])
if not isinstance(INTENT_PROMPT, str) or not INTENT_PROMPT.strip():
    raise ValueError(f"Prompt C is missing a prompt string: {_PROMPT_PATH}")
if not JSON_FIELDS or len(set(JSON_FIELDS)) != len(JSON_FIELDS) or any(not isinstance(field, str) or not field for field in JSON_FIELDS):
    raise ValueError(f"Prompt C json_fields must be nonempty unique strings: {_PROMPT_PATH}")


@dataclass
class FrameSelection:
    indices: Tensor                 # [B,M], positions in the input tensor, ascending
    selection_weights: Tensor       # [B,M,T], hard forward / soft surrogate backward
    soft_weights: Tensor            # [B,M,T], score-rank distributions, time-reordered
    scores: Tensor                  # [B,T], invalid frames have -inf
    temporal_weights: Tensor        # [B,T], masked softmax of frame scores
    selected_landmarks: Tensor      # [B,M,J,3]
    selected_joint_mask: Tensor     # [B,M,J], True = visible/present
    selected_times: Tensor          # [B,M], original frame indices/times if supplied
    source_times: Tensor            # [B,T]
    source_joint_mask: Tensor       # [B,T,J]
    source_frame_mask: Tensor       # [B,T], excludes batch padding
    selected_rgb: Tensor | None     # [B,M,3,H,W]


class TemporalFrameSelector(nn.Module):
    """Pose + next-frame pose difference -> MLP -> distinct temporal samples.

    Training: sequential softmax distributions with hard exclusion of already
    selected indices, optionally adding one Gumbel perturbation per frame.
    Straight-through one-hot matrices keep actual source frames in the forward
    pass and use softmax derivatives in backward. This is a biased gradient
    approximation, not the exact derivative of discrete top-k or sorting.

    Eval: deterministic top-M scores, chronologically sorted. At least M frames
    with one valid joint are required; short clips raise rather than duplicate.
    Neither extraction nor root/body-scale normalization is performed here.
    """

    def __init__(self, num_joints: int = 33, num_frames: int = 8,
                 hidden_dim: int = 256, temperature: float = 1.0,
                 stochastic: bool = True, visibility_threshold: float = 0.0):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (num_joints, num_frames, hidden_dim)):
            raise ValueError("Joint/frame/hidden sizes must be positive integers")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= visibility_threshold < 1:
            raise ValueError("visibility_threshold must be in [0,1)")
        self.num_joints, self.num_frames = num_joints, num_frames
        self.temperature, self.stochastic = temperature, stochastic
        self.visibility_threshold = visibility_threshold
        self.scorer = nn.Sequential(
            nn.Linear(num_joints * 6, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, landmarks: Tensor, frame_mask: Tensor | None = None,
                landmark_mask: Tensor | None = None, visibility: Tensor | None = None,
                rgb: Tensor | None = None, frame_times: Tensor | None = None,
                temperature: float | None = None) -> FrameSelection:
        if landmarks.ndim != 4 or landmarks.shape[2:] != (self.num_joints, 3):
            raise ValueError(f"landmarks must be [B,T,{self.num_joints},3] XYZ only")
        b, t, j, _ = landmarks.shape
        if b == 0 or t == 0 or not landmarks.is_floating_point():
            raise ValueError("Expected a nonempty floating-point landmark tensor")
        device = landmarks.device
        frame_mask = torch.ones((b, t), dtype=torch.bool, device=device) if frame_mask is None else frame_mask.to(device=device, dtype=torch.bool)
        landmark_mask = torch.ones((b, t, j), dtype=torch.bool, device=device) if landmark_mask is None else landmark_mask.to(device=device, dtype=torch.bool)
        if frame_mask.shape != (b, t) or landmark_mask.shape != (b, t, j):
            raise ValueError("Incorrect frame_mask or landmark_mask shape")
        joint_mask = landmark_mask & frame_mask.unsqueeze(-1)
        if visibility is not None:
            if visibility.shape != (b, t, j):
                raise ValueError("visibility must be [B,T,J]")
            visibility = visibility.to(device)
            if not torch.isfinite(visibility[joint_mask]).all() or ((visibility[joint_mask] < 0) | (visibility[joint_mask] > 1)).any():
                raise ValueError("Valid visibility scores must be finite and in [0,1]")
            joint_mask = joint_mask & (visibility > self.visibility_threshold)
        if not torch.isfinite(landmarks[joint_mask]).all():
            raise ValueError("Valid landmarks must be finite")
        valid_frames = frame_mask & joint_mask.any(-1)
        if (valid_frames.sum(-1) < self.num_frames).any():
            raise ValueError(f"Need at least {self.num_frames} frames with visible/present landmarks in every clip")
        if frame_times is None:
            times = torch.arange(t, device=device, dtype=torch.float32).expand(b, -1)
        else:
            if frame_times.shape != (b, t):
                raise ValueError("frame_times must be [B,T]")
            times = frame_times.to(device=device, dtype=torch.float32)
        for batch in range(b):
            real_times = times[batch, frame_mask[batch]]
            if not torch.isfinite(real_times).all() or (real_times[1:] <= real_times[:-1]).any():
                raise ValueError("Real frame_times must be finite and strictly increasing")
        # Replace ignored coordinates/times before arithmetic (NaN * 0 is NaN).
        times = torch.where(frame_mask, times, 0.0)
        pose = torch.where(joint_mask.unsqueeze(-1), landmarks, 0.0).float()
        valid_difference = joint_mask[:, :-1] & joint_mask[:, 1:]
        difference = torch.where(valid_difference.unsqueeze(-1), pose[:, 1:] - pose[:, :-1], 0.0)
        difference = F.pad(difference, (0, 0, 0, 0, 0, 1))
        features = torch.cat((pose.flatten(2), difference.flatten(2)), dim=-1)
        scores = self.scorer(features.to(dtype=next(self.scorer.parameters()).dtype)).squeeze(-1).float()
        scores = scores.masked_fill(~valid_frames, -torch.inf)
        tau = self.temperature if temperature is None else temperature
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperature must be finite and positive")
        ranking_scores = scores
        if self.training and self.stochastic:
            u = torch.rand_like(scores).clamp_(1e-6, 1 - 1e-6)
            ranking_scores = scores - torch.log(-torch.log(u))
        available = valid_frames.clone()
        hard_indices, rows, soft_rows = [], [], []
        for _ in range(self.num_frames):
            remaining = ranking_scores.masked_fill(~available, -torch.inf)
            index = remaining.argmax(-1)
            hard = F.one_hot(index, num_classes=t).to(scores.dtype)
            soft = F.softmax(remaining / tau, dim=-1)
            row = hard + (soft - soft.detach()) if self.training else hard
            hard_indices.append(index)
            rows.append(row)
            soft_rows.append(soft)
            # Use a new mask: autograd saved the previous one for masked_fill.
            available = available & ~hard.bool()
        indices = torch.stack(hard_indices, dim=1)
        order = indices.argsort(dim=1)
        indices = indices.gather(1, order)
        weights = torch.stack(rows, 1).gather(1, order.unsqueeze(-1).expand(-1, -1, t))
        soft_weights = torch.stack(soft_rows, 1).gather(1, order.unsqueeze(-1).expand(-1, -1, t))
        selected_pose = torch.einsum("bmt,btjc->bmjc", weights, pose)
        selected_mask = joint_mask.gather(1, indices.unsqueeze(-1).expand(-1, -1, j))
        selected_times = torch.einsum("bmt,bt->bm", weights, times)
        selected_rgb = None
        if rgb is not None:
            if rgb.ndim != 5 or rgb.shape[:3] != (b, t, 3) or rgb.device != device or not rgb.is_floating_point():
                raise ValueError("rgb must be floating [B,T,3,H,W], on the landmarks device")
            # No gather, NumPy, PIL, or detach: RGB retains the selection gradient.
            selected_rgb = torch.einsum("bmt,btchw->bmchw", weights.to(rgb.dtype), rgb)
        return FrameSelection(indices, weights, soft_weights, scores,
                              F.softmax(scores / tau, -1), selected_pose,
                              selected_mask, selected_times, times, joint_mask,
                              frame_mask, selected_rgb)


@dataclass
class PreparedQwenBatch:
    model_inputs: dict[str, Tensor]
    labels: Tensor | None         # [B,L], prompt/pad=-100, for external causal CE
    prompt_lengths: Tensor
    field_masks: dict[str, Tensor] | None = None


class JSONGenerationError(ValueError):
    def __init__(self, raw_output: str, message: str = "Model did not generate the required JSON object"):
        super().__init__(message)
        self.raw_output = raw_output


class QwenIntentModel(nn.Module):
    """Qwen2.5-VL-7B-Instruct with LoRA-only language-model training.

    Base language and vision weights stay frozen. Only language attention LoRA
    adapters are trainable inside this wrapper; the separate selector can train.

    forward(**prepared.model_inputs) returns Qwen's logits/model output without
    calculating loss. generate_json() is inference-only and returns whole JSON
    objects. Autoregressive token generation/JSON parsing is not differentiable.

    Images are eight ordered image inputs, not a regularly sampled video. All
    resizing, normalization, and Qwen patchification use torch to preserve input
    gradients. CPU/PIL processor paths are deliberately avoided for image data.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 training_mode: str = "lora", device: str | torch.device | None = None,
                 dtype: torch.dtype | None = None, gradient_checkpointing: bool = True,
                 num_frames: int = 8, lora_rank: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, model: nn.Module | None = None,
                 processor: Any | None = None, local_files_only: bool = False,
                 output_fields: Sequence[str] = JSON_FIELDS):
        super().__init__()
        if training_mode != "lora":
            raise ValueError("QwenIntentModel supports LoRA training only; training_mode must be 'lora'")
        if type(lora_rank) is not int or lora_rank <= 0 or not math.isfinite(lora_alpha) or lora_alpha <= 0 or not 0 <= lora_dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive; dropout must be in [0,1)")
        self.output_fields = tuple(output_fields)
        if not self.output_fields or len(set(self.output_fields)) != len(self.output_fields) or set(self.output_fields) - set(JSON_FIELDS):
            raise ValueError(f"output_fields must be a nonempty subset of {JSON_FIELDS}")
        self.prompt = INTENT_PROMPT
        if self.output_fields != JSON_FIELDS:
            self.prompt = INTENT_PROMPT.split("Output format json:")[0] + "Output format json: " + json.dumps({f: "..." for f in self.output_fields})
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor together, or neither")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None:
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
        if model is None:
            try:
                from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError("Install transformers==4.57.1 and accelerate; see TRAINING_USAGE.md") from exc
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False, local_files_only=local_files_only)
            processor.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
            # No inference-only device_map='auto'; place explicitly for training.
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, attn_implementation="sdpa", local_files_only=local_files_only)
        self.model = model.to(device)
        self.processor = processor
        self.num_frames, self.training_mode = num_frames, training_mode
        self.model.requires_grad_(False)
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA training requires peft==0.17.1") from exc
        targets = [name for name, module in self.model.named_modules()
                   if isinstance(module, nn.Linear) and ".self_attn." in name
                   and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
                   and ".visual." not in name]
        if not targets:
            raise ValueError("Cannot find Qwen language attention projections for LoRA")
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank,
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                            target_modules=targets, bias="none")
        self.model = get_peft_model(self.model, config)
        # Enforce the invariant even if the injected model had other trainables.
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(("lora_A." in name or "lora_B." in name) and ".visual." not in name)
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No trainable LoRA adapter parameters were created")
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        ip = self.processor.image_processor
        self.patch_size = int(ip.patch_size)
        self.merge_size = int(ip.merge_size)
        self.temporal_patch_size = int(ip.temporal_patch_size)
        # Legacy processors expose attributes; newer PIL processors keep these
        # pixel-area limits in size (also supports Transformers' SizeDict).
        min_pixels = getattr(ip, "min_pixels", None)
        max_pixels = getattr(ip, "max_pixels", None)
        size = getattr(ip, "size", None)
        try:
            self.min_pixels = int(min_pixels if min_pixels is not None else size["shortest_edge"])
            self.max_pixels = int(max_pixels if max_pixels is not None else size["longest_edge"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Qwen image processor requires min_pixels/max_pixels or size with shortest_edge/longest_edge") from exc
        if not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Qwen image pixel limits must satisfy 0 < minimum <= maximum")
        self.register_buffer("image_mean", torch.tensor(ip.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(ip.image_std).view(1, 3, 1, 1), persistent=False)
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        self.image_token = image_token
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if self.image_token_id != self.model.config.image_token_id:
            raise ValueError("Tokenizer and model image token IDs differ")
        self.train()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _image_inputs(self, rgb: Tensor) -> tuple[Tensor, Tensor, int]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (self.num_frames, 3) or rgb.shape[0] == 0:
            raise ValueError(f"RGB must be [B,{self.num_frames},3,H,W]")
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or rgb.min() < -1e-5 or rgb.max() > 1 + 1e-5:
            raise ValueError("Supply unnormalized floating RGB in [0,1]")
        b, m, _, h, w = rgb.shape
        if min(h, w) <= 0 or max(h, w) / min(h, w) > 200:
            raise ValueError("Invalid image dimensions/aspect ratio")
        factor = self.patch_size * self.merge_size
        rh, rw = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
        if rh * rw > self.max_pixels:
            ratio = math.sqrt(h * w / self.max_pixels)
            rh, rw = max(factor, math.floor(h / ratio / factor) * factor), max(factor, math.floor(w / ratio / factor) * factor)
        elif rh * rw < self.min_pixels:
            ratio = math.sqrt(self.min_pixels / (h * w))
            rh, rw = math.ceil(h * ratio / factor) * factor, math.ceil(w * ratio / factor) * factor
        images = rgb.reshape(b * m, 3, h, w).to(device=self.device, dtype=torch.float32)
        if (rh, rw) != (h, w):
            images = F.interpolate(images, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True)
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        gh, gw = rh // self.patch_size, rw // self.patch_size
        # Official Qwen patch ordering, with each independent image repeated
        # temporal_patch_size times as in the standard still-image processor.
        p, s, tp = self.patch_size, self.merge_size, self.temporal_patch_size
        patches = images[:, None].expand(-1, tp, -1, -1, -1)
        patches = patches.reshape(b * m, 1, tp, 3, gh // s, s, p, gw // s, s, p)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(b * m * gh * gw, 3 * tp * p * p)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=self.device).expand(b * m, -1).contiguous()
        return patches, grid, gh * gw // (s * s)

    def _messages(self, transcript: str, target: str | None = None) -> list[dict]:
        if not isinstance(transcript, str):
            raise TypeError("Each transcript must be a string")
        # Reserve chat/multimodal control strings for template-generated tokens.
        if "<|" in transcript or (target is not None and "<|" in target):
            raise ValueError("Transcript/target contains a reserved Qwen control token")
        content = [{"type": "text", "text": self.prompt + "\nTranscript:\n" + transcript}]
        for frame in range(self.num_frames):
            content.extend(({"type": "text", "text": f"\nFrame {frame + 1}: "}, {"type": "image"}))
        messages = [{"role": "user", "content": content}]
        if target is not None:
            messages.append({"role": "assistant", "content": target})
        return messages

    def _render(self, messages: list[dict], image_tokens: int, generation_prompt: bool) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        parts = text.split(self.image_token)
        if len(parts) != self.num_frames + 1:
            raise ValueError("Unexpected number of image placeholders from the chat template")
        return (self.image_token * image_tokens).join(parts)

    def prepare_inputs(self, rgb: Tensor, transcripts: Sequence[str],
                       target_jsons: Sequence[Mapping[str, str] | str] | None = None) -> PreparedQwenBatch:
        """Build inputs and optional assistant-only labels; compute no losses.

        target_jsons must explicitly contain the configured output_fields.
        Full JSON objects stay together. No annotation-to-evidence conversion is
        invented. Do not truncate multimodal sequences: choose smaller image
        sizes/batches or handle overlength examples in your training code.
        """
        pixels, grid, image_tokens = self._image_inputs(rgb)
        b = rgb.shape[0]
        if isinstance(transcripts, str) or len(transcripts) != b:
            raise ValueError("transcripts must contain one string per clip")
        if target_jsons is not None and len(target_jsons) != b:
            raise ValueError("target_jsons must contain one object per clip")
        prompts, full_texts, field_ranges = [], [], []
        for i, transcript in enumerate(transcripts):
            prompts.append(self._render(self._messages(transcript), image_tokens, True))
            if target_jsons is not None:
                obj = json.loads(target_jsons[i]) if isinstance(target_jsons[i], str) else dict(target_jsons[i])
                self._validate_json(obj, self.output_fields)
                target, boundaries = "{", []
                for k, field in enumerate(self.output_fields):
                    boundaries.append(len(target) if k else 0)
                    target += (", " if k else "") + json.dumps(field) + ": " + json.dumps(obj[field], ensure_ascii=False)
                target += "}"
                full_texts.append(self._render(self._messages(transcript, target), image_tokens, False))
                if not full_texts[-1].startswith(prompts[-1]):
                    raise ValueError("Chat template does not preserve the assistant prefix")
                boundaries.append(len(full_texts[-1]) - len(prompts[-1]))
                field_ranges.append([(len(prompts[-1]) + boundaries[k], len(prompts[-1]) + boundaries[k + 1]) for k in range(len(self.output_fields))])
        tokenizer = self.processor.tokenizer
        if target_jsons is not None and not tokenizer.is_fast:
            raise ValueError("Field-specific training requires a fast tokenizer for character offsets")
        texts = full_texts if target_jsons is not None else prompts
        tokens = tokenizer(texts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt",
                           return_offsets_mapping=target_jsons is not None)
        ids, attention = tokens["input_ids"], tokens["attention_mask"]
        lengths, labels = [], None
        if target_jsons is not None:
            labels = ids.clone()
            labels[attention == 0] = -100
        for i, prompt in enumerate(prompts):
            prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            active = ids[i, attention[i].bool()].tolist()
            if active[:len(prefix)] != prefix:
                raise ValueError("Prompt tokens are not an exact prefix of the target sequence")
            lengths.append(len(prefix))
            expected = self.num_frames * image_tokens
            if active.count(self.image_token_id) != expected:
                raise ValueError("Image-token count does not match patch grid")
            if labels is not None:
                offset = ids.shape[1] - len(active)  # left padding
                labels[i, offset:offset + len(prefix)] = -100
        field_masks = None
        if labels is not None:
            field_masks = {field: torch.zeros_like(labels, dtype=torch.bool) for field in self.output_fields}
            for i, ranges in enumerate(field_ranges):
                for position in (labels[i] != -100).nonzero().flatten().tolist():
                    start, end = tokens["offset_mapping"][i, position].tolist()
                    overlaps = [max(0, min(end, hi) - max(start, lo)) for lo, hi in ranges]
                    owner = overlaps.index(max(overlaps)) if max(overlaps) > 0 else len(ranges) - 1
                    field_masks[self.output_fields[owner]][i, position] = True
            field_masks = {field: mask.to(self.device) for field, mask in field_masks.items()}
        inputs = {"input_ids": ids.to(self.device), "attention_mask": attention.to(self.device),
                  "pixel_values": pixels, "image_grid_thw": grid}
        return PreparedQwenBatch(inputs, None if labels is None else labels.to(self.device),
                                 torch.tensor(lengths, device=self.device), field_masks)

    def forward(self, **model_inputs):
        """Return Qwen outputs (including differentiable logits), without loss."""
        if "labels" in model_inputs:
            raise ValueError("Keep labels outside forward; compute generation loss in train.py")
        return self.model(**model_inputs, use_cache=False, return_dict=True)

    @staticmethod
    def _validate_json(obj: Any, fields: Sequence[str] = JSON_FIELDS) -> None:
        if not isinstance(obj, dict) or set(obj) != set(fields) or any(not isinstance(obj[k], str) for k in fields):
            raise ValueError(f"Expected exactly these JSON string fields: {tuple(fields)}")

    @torch.no_grad()
    def generate_text(self, rgb: Tensor, transcripts: Sequence[str],
                      max_new_tokens: int = 512, **generation_kwargs) -> list[str]:
        """Return the complete generated response per clip, for external parsing."""
        was_training = self.training
        self.eval()
        try:
            prepared = self.prepare_inputs(rgb, transcripts)
            opts = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True,
                    "pad_token_id": self.processor.tokenizer.pad_token_id}
            opts.update(generation_kwargs)
            ids = self.model.generate(**prepared.model_inputs, **opts)
            continuation = ids[:, prepared.model_inputs["input_ids"].shape[1]:]
            return self.processor.tokenizer.batch_decode(continuation, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
        finally:
            self.train(was_training)

    def generate_json(self, rgb: Tensor, transcripts: Sequence[str], **kwargs) -> list[dict[str, str]]:
        """Return whole parsed objects, never individual fields or attributes.

        JSON prompting is not a syntax guarantee. Malformed/truncated responses
        raise JSONGenerationError with raw_output for train.py/caller handling.
        """
        objects = []
        for text in self.generate_text(rgb, transcripts, **kwargs):
            cleaned = text.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
            try:
                obj = json.loads(cleaned)
                self._validate_json(obj, self.output_fields)
            except (ValueError, TypeError) as exc:
                raise JSONGenerationError(text) from exc
            objects.append(obj)
        return objects
