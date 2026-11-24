"""Landmark frame selection + one shared Qwen2.5-VL LoRA model, four prompts.
Architecture preserved: pose/next-frame differences -> MLP -> straight-through
frame selection; frozen Qwen vision/base language weights; language-attention
LoRA only; differentiable torch image resizing, normalization and patchification.
Only A2 receives the selected RGB frames. A1/B/C are text-only calls.
There is no optimizer, training loop, or loss computation in this file.
Requirements: torch, transformers==4.57.1, peft==0.17.1, accelerate.
Training (all four target objects must come from your annotations):
    selection = selector(landmarks, rgb=rgb, frame_mask=frame_mask)
    prepared = llm.prepare_inputs(selection.selected_rgb, transcripts, targets)
    result = llm(prepared=prepared)
    # result.prompts['Prompt A2'].logits / labels / field_masks
    # result.responses[i]['Prompt C']['communicative intent']
``targets`` is a mapping with keys 'Prompt A1', 'Prompt A2', 'Prompt B',
'Prompt C'; each value is a batch-length sequence of response JSON objects.
A1: {'conversation topic': str}; A2: {'gesture movement description': str};
B: {'gesture meaning': str}; C: {'communicative intent': str, 'reasoning': str,
'gesture contribution': str}. These are four response objects (six total fields).
Default conditioning='generated' runs the actual four-stage generation chain,
then builds four teacher-forced sequences against the supplied targets. The
returned responses are generated predictions, not the annotation targets.
This costs four generation calls plus four supervised forwards per batch.
Pass previously generated response_jsons to avoid repeating generation.
Optional conditioning='targets' explicitly uses annotation evidence for B/C,
skips generation, and returns responses=None; targets are still available.
Generation/JSON parsing is nondifferentiable: C loss cannot backpropagate through
B/A1/A2 text. Each stage's own supervised logits retain LoRA gradients; A2 logits
also retain the selector-to-RGB gradient. Shift logits/labels/masks for causal CE
in train.py. Autoregressive generation can fail JSON validation; handle
JSONGenerationError in the caller rather than silently replacing evidence.
Optional prompts_path loads the existing A1_A2_B_C.json. Built-in defaults make
this file standalone. Placeholders from the earlier prompt templates are
recognized; plain instructions also work. Do not pass placeholder strings as
image data: A2 attaches actual selected tensors as Qwen image inputs.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
PROMPT_ORDER = ('Prompt A1', 'Prompt A2', 'Prompt B', 'Prompt C')
PROMPT_FIELDS = {
    'Prompt A1': ('conversation topic',),
    'Prompt A2': ('gesture movement description',),
    'Prompt B': ('gesture meaning',),
    'Prompt C': ('communicative intent', 'reasoning', 'gesture contribution'),
}
DEFAULT_PROMPTS = {
    'Prompt A1': 'What is the topic of the conversation according to this transcript?',
    'Prompt A2': (
        'You are given a set of video frames in chronological order, numbered from '
        'f1 to fN. The person in the frames is making a gesture. Describe the '
        'movement of the gesture across the frames. Focus on the hands, fingers, '
        'changes in palm orientation, and head.'
    ),
    'Prompt B': (
        'You are given the description of a gesture and the topic of the conversation '
        'when the gesture was being made. Infer the meaning of the gesture. The '
        'inferred meaning should be supported by both the gesture movement '
        'description and the conversation topic.'
    ),
    'Prompt C': (
        'Infer the communicative intent of the person based on the meaning of the '
        'gesture, the topic of conversation, and the transcript. Support the intent '
        'with evidence in these inputs. Explain the reasoning behind the inferred '
        'communicative intent and how the gesture contributes to that intent.'
    ),
}
# Compatibility constants refer to the final stage only; use PROMPT_FIELDS for all.
INTENT_PROMPT = DEFAULT_PROMPTS['Prompt C']
JSON_FIELDS = PROMPT_FIELDS['Prompt C']
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
    selected_rgb: Tensor | None
class TemporalFrameSelector(nn.Module):
    """Original pose/difference MLP and distinct chronological ST selections.
    Landmarks are [B,T,J,3], RGB is floating [B,T,3,H,W] in [0,1].
    No pose extraction/root normalization is performed here. At least M frames
    with a valid joint are required. Hard forward / soft surrogate backward is
    a biased gradient approximation, not the derivative of discrete top-k.
    """
    def __init__(self, num_joints: int = 33, num_frames: int = 8,
        hidden_dim: int = 256, temperature: float = 1.0,
        stochastic: bool = True, visibility_threshold: float = 0.0):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (num_joints, num_frames, hidden_dim)):
            raise ValueError('Joint/frame/hidden sizes must be positive integers')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if not 0 <= visibility_threshold < 1:
            raise ValueError('visibility_threshold must be in [0,1)')
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
            raise ValueError(f'landmarks must be [B,T,{self.num_joints},3] XYZ only')
        b, t, j, _ = landmarks.shape
        if b == 0 or t == 0 or not landmarks.is_floating_point():
            raise ValueError('Expected a nonempty floating-point landmark tensor')
        device = landmarks.device
        frame_mask = (torch.ones((b, t), dtype=torch.bool, device=device) if frame_mask is None
else frame_mask.to(device=device, dtype=torch.bool))
        landmark_mask = (torch.ones((b, t, j), dtype=torch.bool, device=device) if landmark_mask is None
else landmark_mask.to(device=device, dtype=torch.bool))
        if frame_mask.shape != (b, t) or landmark_mask.shape != (b, t, j):
            raise ValueError('Incorrect frame_mask or landmark_mask shape')
        joint_mask = landmark_mask & frame_mask.unsqueeze(-1)
        if visibility is not None:
            if visibility.shape != (b, t, j):
                raise ValueError('visibility must be [B,T,J]')
            visibility = visibility.to(device)
            if (not torch.isfinite(visibility[joint_mask]).all()
                    or ((visibility[joint_mask] < 0) | (visibility[joint_mask] > 1)).any()):
                raise ValueError('Valid visibility scores must be finite and in [0,1]')
            joint_mask = joint_mask & (visibility > self.visibility_threshold)
        if not torch.isfinite(landmarks[joint_mask]).all():
            raise ValueError('Valid landmarks must be finite')
        valid_frames = frame_mask & joint_mask.any(-1)
        if (valid_frames.sum(-1) < self.num_frames).any():
            raise ValueError(f'Need at least {self.num_frames} frames with visible/present landmarks in every clip')
        if frame_times is None:
            times = torch.arange(t, device=device, dtype=torch.float32).expand(b, -1)
        else:
            if frame_times.shape != (b, t):
                raise ValueError('frame_times must be [B,T]')
            times = frame_times.to(device=device, dtype=torch.float32)
        for batch in range(b):
            real_times = times[batch, frame_mask[batch]]
            if not torch.isfinite(real_times).all() or (real_times[1:] <= real_times[:-1]).any():
                raise ValueError('Real frame_times must be finite and strictly increasing')
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
            raise ValueError('temperature must be finite and positive')
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
            available = available & ~hard.bool()
        indices = torch.stack(hard_indices, dim=1)
        order = indices.argsort(dim=1)
        indices = indices.gather(1, order)
        weights = torch.stack(rows, 1).gather(1, order.unsqueeze(-1).expand(-1, -1, t))
        soft_weights = torch.stack(soft_rows, 1).gather(1, order.unsqueeze(-1).expand(-1, -1, t))
        selected_pose = torch.einsum('bmt,btjc->bmjc', weights, pose)
        selected_mask = joint_mask.gather(1, indices.unsqueeze(-1).expand(-1, -1, j))
        selected_times = torch.einsum('bmt,bt->bm', weights, times)
        selected_rgb = None
        if rgb is not None:
            if (rgb.ndim != 5 or rgb.shape[:3] != (b, t, 3)
                    or rgb.device != device or not rgb.is_floating_point()):
                raise ValueError('rgb must be floating [B,T,3,H,W], on the landmarks device')
            selected_rgb = torch.einsum('bmt,btchw->bmchw', weights.to(rgb.dtype), rgb)
        return FrameSelection(indices, weights, soft_weights, scores,
                              F.softmax(scores / tau, -1), selected_pose,
                              selected_mask, selected_times, times, joint_mask,
                              frame_mask, selected_rgb)
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
    def __init__(self, raw_output: str, message: str = 'Model did not generate the required JSON object',
        prompt_name: str | None = None, sample_index: int | None = None):
        super().__init__(f'{prompt_name or "Generation"}, sample {sample_index}: {message}')
        self.raw_output = raw_output
        self.prompt_name = prompt_name
        self.sample_index = sample_index
class QwenIntentModel(nn.Module):
    """One shared Qwen2.5-VL-7B model, language-attention LoRA, four prompts.
    prepare_inputs returns all four stage batches, not a single C batch.
    forward(prepared=...) returns all four supervised model outputs without loss.
    forward(**one_stage.model_inputs) remains available for stagewise training.
    generate_json returns every prompt's complete response JSON per sample.
    """
    def __init__(self, model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        training_mode: str = 'lora', device: str | torch.device | None = None,
        dtype: torch.dtype | None = None, gradient_checkpointing: bool = True,
        num_frames: int = 8, lora_rank: int = 16, lora_alpha: int = 32,
        lora_dropout: float = 0.05, model: nn.Module | None = None,
        processor: Any | None = None, local_files_only: bool = False,
        output_fields: Sequence[str] = JSON_FIELDS,
        prompts_path: str | Path | None = None):
        super().__init__()
        if training_mode != 'lora':
            raise ValueError("QwenIntentModel supports LoRA training only; training_mode must be 'lora'")
        if (type(lora_rank) is not int or lora_rank <= 0 or not math.isfinite(lora_alpha)
                or lora_alpha <= 0 or not 0 <= lora_dropout < 1):
            raise ValueError('LoRA rank/alpha must be positive; dropout must be in [0,1)')
        if tuple(output_fields) != JSON_FIELDS:
            raise ValueError('The four-prompt architecture requires all C fields and all A1/A2/B fields')
        self.output_fields = tuple(output_fields)
        self.prompts = self._load_prompts(prompts_path)
        if type(num_frames) is not int or num_frames < 1:
            raise ValueError('num_frames must be a positive integer')
        if (model is None) != (processor is None):
            raise ValueError('Inject both model and processor together, or neither')
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if dtype is None:
            dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if device.type == 'cuda' else torch.float32)
        if model is None:
            try:
                from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError('Install transformers==4.57.1 and accelerate') from exc
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False, local_files_only=local_files_only)
            processor.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, dtype=dtype, attn_implementation='sdpa', local_files_only=local_files_only)
        self.model = model.to(device)
        self.processor = processor
        self.num_frames, self.training_mode = num_frames, training_mode
        self.model.requires_grad_(False)
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError('LoRA training requires peft==0.17.1') from exc
        targets = [name for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and '.self_attn.' in name
                   and name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
                   and '.visual.' not in name]
        if not targets:
            raise ValueError('Cannot find Qwen language attention projections for LoRA')
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias='none')
        self.model = get_peft_model(self.model, config)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(('lora_A.' in name or 'lora_B.' in name) and '.visual.' not in name)
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('No trainable LoRA adapter parameters were created')
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        ip = self.processor.image_processor
        self.patch_size = int(ip.patch_size)
        self.merge_size = int(ip.merge_size)
        self.temporal_patch_size = int(ip.temporal_patch_size)
        min_pixels, max_pixels = getattr(ip, 'min_pixels', None), getattr(ip, 'max_pixels', None)
        size = getattr(ip, 'size', None)
        try:
            self.min_pixels = int(min_pixels if min_pixels is not None else size['shortest_edge'])
            self.max_pixels = int(max_pixels if max_pixels is not None else size['longest_edge'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Qwen image processor requires min/max_pixels or shortest/longest_edge size') from exc
        if not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError('Qwen image pixel limits must satisfy 0 < minimum <= maximum')
        self.register_buffer('image_mean', torch.tensor(ip.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('image_std', torch.tensor(ip.image_std).view(1, 3, 1, 1), persistent=False)
        self.image_token = getattr(self.processor, 'image_token', '<|image_pad|>')
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id != self.model.config.image_token_id:
            raise ValueError('Tokenizer and model image token IDs differ')
        self.train()
    @staticmethod
    def _load_prompts(path: str | Path | None) -> dict[str, str]:
        if path is None:
            return dict(DEFAULT_PROMPTS)
        with Path(path).open(encoding='utf-8') as handle:
            config = json.load(handle)
        prompts = {}
        for name in PROMPT_ORDER:
            item = config[name]
            if not isinstance(item.get('prompt'), str) or not item['prompt'].strip():
                raise ValueError(f'{name} needs a nonempty prompt string')
            if tuple(item.get('json_fields', ())) != PROMPT_FIELDS[name]:
                raise ValueError(f'{name} json_fields must be {PROMPT_FIELDS[name]}')
            prompts[name] = item['prompt']
        return prompts
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
    def _image_inputs(self, rgb: Tensor) -> tuple[Tensor, Tensor, int]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (self.num_frames, 3) or rgb.shape[0] == 0:
            raise ValueError(f'RGB must be [B,{self.num_frames},3,H,W]')
        if (not rgb.is_floating_point() or not torch.isfinite(rgb).all()
                or rgb.min() < -1e-5 or rgb.max() > 1 + 1e-5):
            raise ValueError('Supply unnormalized floating RGB in [0,1]')
        b, m, _, h, w = rgb.shape
        if min(h, w) <= 0 or max(h, w) / min(h, w) > 200:
            raise ValueError('Invalid image dimensions/aspect ratio')
        factor = self.patch_size * self.merge_size
        rh, rw = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
        if rh * rw > self.max_pixels:
            ratio = math.sqrt(h * w / self.max_pixels)
            rh = max(factor, math.floor(h / ratio / factor) * factor)
            rw = max(factor, math.floor(w / ratio / factor) * factor)
        elif rh * rw < self.min_pixels:
            ratio = math.sqrt(self.min_pixels / (h * w))
            rh, rw = math.ceil(h * ratio / factor) * factor, math.ceil(w * ratio / factor) * factor
        images = rgb.reshape(b * m, 3, h, w).to(device=self.device, dtype=torch.float32)
        if (rh, rw) != (h, w):
            images = F.interpolate(images, size=(rh, rw), mode='bicubic', align_corners=False, antialias=True)
        images = (images - self.image_mean.to(images)) / self.image_std.to(images)
        gh, gw = rh // self.patch_size, rw // self.patch_size
        p, s, tp = self.patch_size, self.merge_size, self.temporal_patch_size
        patches = images[:, None].expand(-1, tp, -1, -1, -1)
        patches = patches.reshape(b * m, 1, tp, 3, gh // s, s, p, gw // s, s, p)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(b * m * gh * gw, 3 * tp * p * p)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=self.device).expand(b * m, -1).contiguous()
        return patches, grid, gh * gw // (s * s)
    @staticmethod
    def _validate_json(obj: Any, fields: Sequence[str] = JSON_FIELDS) -> None:
        if (not isinstance(obj, dict) or set(obj) != set(fields)
                or any(not isinstance(obj[k], str) for k in fields)):
            raise ValueError(f'Expected exactly these JSON string fields: {tuple(fields)}')
        if any('<|' in obj[k] for k in fields):
            raise ValueError('Response contains a reserved Qwen control token')
    def _batch_objects(self, name: str, values: Sequence[Mapping[str, str] | str],
        batch_size: int) -> list[dict[str, str]]:
        if isinstance(values, (str, bytes, Mapping)) or len(values) != batch_size:
            raise ValueError(f'{name} requires one response object per clip')
        result = []
        for value in values:
            obj = json.loads(value) if isinstance(value, str) else dict(value)
            self._validate_json(obj, PROMPT_FIELDS[name])
            result.append(obj)
        return result
    def _messages(self, name: str, inputs: Mapping[str, str],
        target: str | None = None) -> list[dict]:
        replacements = {f'<INPUT_{key}>': value for key, value in inputs.items()}
        replacements['<INPUT_VIDEO_FRAMES>'] = '(Ordered frames attached below.)'
        instruction = self.prompts[name]
        # Support both plain instructions and the explicit templates from the chat.
        # Replace in a single pass, so text inside an input is never re-templated.
        if re.search(r'<INPUT_[A-Z_]+>', instruction):
            def substitute(match):
                if match.group(0) not in replacements:
                    raise ValueError(f'{name}: unsupported input placeholder {match.group(0)}')
                return replacements[match.group(0)]
            instruction = re.sub(r'<INPUT_[A-Z_]+>', substitute, instruction)
            # Ensure every required input is included even in a partial template.
            missing = [key for key in inputs if f'<INPUT_{key}>' not in self.prompts[name]]
        else:
            missing = list(inputs)
        for key in missing:
            instruction += f'\n\n{key.replace("_", " ").title()}:\n{inputs[key]}'
        instruction += ('\n\nReturn only a valid JSON object with exactly these string fields:\n'
                        + json.dumps({field: '...' for field in PROMPT_FIELDS[name]}))
        if '<|' in instruction or (target is not None and '<|' in target):
            raise ValueError('Prompt/input/target contains a reserved Qwen control token')
        content = [{'type': 'text', 'text': instruction}]
        if name == 'Prompt A2':
            for frame in range(self.num_frames):
                content.extend(({'type': 'text', 'text': f'\nFrame f{frame + 1}: '}, {'type': 'image'}))
        messages = [{'role': 'user', 'content': content}]
        if target is not None:
            messages.append({'role': 'assistant', 'content': target})
        return messages
    def _render(self, name: str, messages: list[dict], image_tokens: int,
        generation_prompt: bool) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=generation_prompt)
        parts = text.split(self.image_token)
        expected = self.num_frames if name == 'Prompt A2' else 0
        if len(parts) != expected + 1:
            raise ValueError(f'{name}: unexpected number of image placeholders')
        return (self.image_token * image_tokens).join(parts)
    def _prepare_stage(self, name: str, inputs: Sequence[Mapping[str, str]],
        rgb: Tensor | None = None,
        targets: Sequence[Mapping[str, str] | str] | None = None) -> PreparedQwenBatch:
        if name not in PROMPT_FIELDS or not inputs:
            raise ValueError('Supply a valid prompt name and a nonempty input batch')
        b = len(inputs)
        pixels = grid = None
        image_tokens = 0
        if name == 'Prompt A2':
            if rgb is None or rgb.shape[0] != b:
                raise ValueError('A2 needs a selected RGB tensor for each clip')
            pixels, grid, image_tokens = self._image_inputs(rgb)
        elif rgb is not None:
            raise ValueError('Only A2 accepts RGB frames')
        objects = None if targets is None else self._batch_objects(name, targets, b)
        fields = PROMPT_FIELDS[name]
        prompts, full_texts, field_ranges = [], [], []
        for i, stage_inputs in enumerate(inputs):
            if any(not isinstance(value, str) for value in stage_inputs.values()):
                raise TypeError('Prompt inputs must be strings')
            prompts.append(self._render(name, self._messages(name, stage_inputs), image_tokens, True))
            if objects is not None:
                target, boundaries = '{', []
                for k, field in enumerate(fields):
                    boundaries.append(len(target) if k else 0)
                    target += (', ' if k else '') + json.dumps(field) + ': ' + json.dumps(objects[i][field], ensure_ascii=False)
                target += '}'
                full_text = self._render(name, self._messages(name, stage_inputs, target), image_tokens, False)
                if not full_text.startswith(prompts[-1]):
                    raise ValueError('Chat template does not preserve the assistant prefix')
                full_texts.append(full_text)
                boundaries.append(len(full_text) - len(prompts[-1]))
                field_ranges.append([(len(prompts[-1]) + boundaries[k], len(prompts[-1]) + boundaries[k + 1])
                for k in range(len(fields))])
        tokenizer = self.processor.tokenizer
        if objects is not None and not tokenizer.is_fast:
            raise ValueError('Field-specific training requires a fast tokenizer for character offsets')
        tokens = tokenizer(full_texts if objects is not None else prompts, padding=True,
            truncation=False, add_special_tokens=False, return_tensors='pt',
            return_offsets_mapping=objects is not None)
        ids, attention = tokens['input_ids'], tokens['attention_mask']
        lengths, labels = [], None
        if objects is not None:
            labels = ids.clone()
            labels[attention == 0] = -100
        for i, prompt in enumerate(prompts):
            prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
            active = ids[i, attention[i].bool()].tolist()
            if active[:len(prefix)] != prefix:
                raise ValueError('Prompt tokens are not an exact prefix of the target sequence')
            lengths.append(len(prefix))
            expected = self.num_frames * image_tokens if name == 'Prompt A2' else 0
            if active.count(self.image_token_id) != expected:
                raise ValueError(f'{name}: image-token count does not match patch grid')
            if labels is not None:
                offset = ids.shape[1] - len(active)
                labels[i, offset:offset + len(prefix)] = -100
        field_masks = None
        if labels is not None:
            field_masks = {field: torch.zeros_like(labels, dtype=torch.bool) for field in fields}
            for i, ranges in enumerate(field_ranges):
                for position in (labels[i] != -100).nonzero().flatten().tolist():
                    start, end = tokens['offset_mapping'][i, position].tolist()
                    overlaps = [max(0, min(end, hi) - max(start, lo)) for lo, hi in ranges]
                    owner = overlaps.index(max(overlaps)) if max(overlaps) > 0 else len(ranges) - 1
                    field_masks[fields[owner]][i, position] = True
            field_masks = {field: mask.to(self.device) for field, mask in field_masks.items()}
        model_inputs = {'input_ids': ids.to(self.device), 'attention_mask': attention.to(self.device)}
        if pixels is not None:
            model_inputs.update(pixel_values=pixels, image_grid_thw=grid)
        return PreparedQwenBatch(model_inputs, None if labels is None else labels.to(self.device),
                                 torch.tensor(lengths, device=self.device), field_masks)
    @staticmethod
    def _inputs_for(name: str, transcripts: Sequence[str],
        responses: Sequence[Mapping[str, Mapping[str, str]]]) -> list[dict[str, str]]:
        if name == 'Prompt A1':
            return [{'TRANSCRIPT': text} for text in transcripts]
        if name == 'Prompt A2':
            return [{} for _ in transcripts]
        if name == 'Prompt B':
            return [{'GESTURE_MOVEMENT_DESCRIPTION': item['Prompt A2']['gesture movement description'],
                     'CONVERSATION_TOPIC': item['Prompt A1']['conversation topic']} for item in responses]
        if name == 'Prompt C':
            return [{'GESTURE_MEANING': item['Prompt B']['gesture meaning'],
                     'CONVERSATION_TOPIC': item['Prompt A1']['conversation topic'],
                     'TRANSCRIPT': text} for item, text in zip(responses, transcripts)]
        raise ValueError(f'Unknown prompt {name}')
    def _validate_batch(self, rgb: Tensor, transcripts: Sequence[str]) -> int:
        if rgb.ndim != 5 or rgb.shape[0] == 0 or rgb.shape[1:3] != (self.num_frames, 3):
            raise ValueError(f'Expected selected RGB [B,{self.num_frames},3,H,W]')
        b = rgb.shape[0]
        if isinstance(transcripts, str) or len(transcripts) != b:
            raise ValueError('transcripts must contain one string per clip')
        if any(not isinstance(text, str) or '<|' in text for text in transcripts):
            raise ValueError('Transcripts must be strings without reserved Qwen control tokens')
        return b
    def prepare_inputs(self, rgb: Tensor, transcripts: Sequence[str],
        target_jsons: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
                       *, conditioning: str = 'generated',
        response_jsons: Sequence[Mapping[str, Mapping[str, str]]] | None = None,
        max_new_tokens: int | Mapping[str, int] = 512,
        generation_kwargs: Mapping[str, Any] | None = None) -> PreparedHierarchy:
        """Prepare all four prompts, assistant labels, and per-field token masks.
        Generated evidence is the default, including during training. Explicit
        target conditioning is available for teacher-forced upstream evidence.
        Responses and supervision targets are separate to avoid target leakage.
        No token truncation is performed. C's target is never included in its input.
        """
        b = self._validate_batch(rgb, transcripts)
        if conditioning not in ('generated', 'targets'):
            raise ValueError("conditioning must be 'generated' or 'targets'")
        targets = None
        if target_jsons is not None:
            if not isinstance(target_jsons, Mapping) or set(target_jsons) != set(PROMPT_ORDER):
                raise ValueError(f'target_jsons must be a mapping with all four keys: {PROMPT_ORDER}')
            targets = {name: self._batch_objects(name, target_jsons[name], b) for name in PROMPT_ORDER}
        responses = None
        if conditioning == 'targets':
            if targets is None or response_jsons is not None:
                raise ValueError('Target conditioning requires targets and no generated response_jsons')
            evidence = [{name: targets[name][i] for name in PROMPT_ORDER} for i in range(b)]
        else:
            if response_jsons is None:
                responses = self.generate_json(rgb, transcripts, max_new_tokens=max_new_tokens,
                                               **dict(generation_kwargs or {}))
            else:
                if isinstance(response_jsons, (str, bytes, Mapping)) or len(response_jsons) != b:
                    raise ValueError('response_jsons needs one complete hierarchy per clip')
                responses = []
                for item in response_jsons:
                    if set(item) != set(PROMPT_ORDER):
                        raise ValueError('Each response hierarchy needs all four prompt keys')
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
                name, self._inputs_for(name, transcripts, evidence),
                rgb if name == 'Prompt A2' else None,
                None if targets is None else targets[name])
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
            if 'labels' in model_inputs:
                raise ValueError('Keep labels outside forward; calculate loss in train.py')
            return self.model(**model_inputs, use_cache=False, return_dict=True)
        if model_inputs:
            raise ValueError('Supply either prepared=... or single-stage model inputs')
        outputs = {}
        for name in PROMPT_ORDER:
            batch = prepared.prompts[name]
            model_output = self.model(**batch.model_inputs, use_cache=False, return_dict=True)
            responses = None if prepared.responses is None else [item[name] for item in prepared.responses]
            outputs[name] = PromptOutput(PROMPT_FIELDS[name], model_output, batch.labels,
                                          batch.field_masks, responses)
        return HierarchicalOutput(outputs, prepared.responses, prepared.targets, prepared.conditioning)
    def _generate_stage(self, name: str, inputs: Sequence[Mapping[str, str]],
        rgb: Tensor | None, max_new_tokens: int,
        generation_kwargs: Mapping[str, Any]) -> list[dict[str, str]]:
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
        batch = self._prepare_stage(name, inputs, rgb)
        opts = {'max_new_tokens': max_new_tokens, 'do_sample': False, 'use_cache': True,
                'pad_token_id': self.processor.tokenizer.pad_token_id}
        opts.update(generation_kwargs)
        if opts.get('num_return_sequences', 1) != 1 or opts.get('return_dict_in_generate', False):
            raise ValueError('Use one tensor generation result per input sample')
        generated = self.model.generate(**batch.model_inputs, **opts)
        continuation = generated[:, batch.model_inputs['input_ids'].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(continuation, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        if len(texts) != len(inputs):
            raise ValueError('Generation batch size changed unexpectedly')
        objects = []
        for i, text in enumerate(texts):
            cleaned = text.strip()
            if cleaned.startswith('```') and cleaned.endswith('```'):
                cleaned = '\n'.join(cleaned.splitlines()[1:-1]).strip()
            try:
                obj = json.loads(cleaned)
                self._validate_json(obj, PROMPT_FIELDS[name])
            except (ValueError, TypeError) as exc:
                raise JSONGenerationError(text, prompt_name=name, sample_index=i) from exc
            objects.append(obj)
        return objects
    @torch.no_grad()
    def generate_json(self, rgb: Tensor, transcripts: Sequence[str],
        max_new_tokens: int | Mapping[str, int] = 512,
                      **generation_kwargs) -> list[dict[str, dict[str, str]]]:
        """Generate A1, A2, B, C in order and return all complete response JSONs.
        A1 <- transcript; A2 <- selected frames; B <- A2 description + A1 topic;
        C <- B meaning + A1 topic + transcript. A1/A2 do not see each other's
        response. No annotation is used. JSON errors include stage/sample/raw text.
        Restores the model's previous training/eval mode even if a stage fails.
        """
        b = self._validate_batch(rgb, transcripts)
        responses = [{} for _ in range(b)]
        was_training = self.training
        self.eval()
        try:
            for name in PROMPT_ORDER:
                limit = max_new_tokens[name] if isinstance(max_new_tokens, Mapping) else max_new_tokens
                objects = self._generate_stage(name, self._inputs_for(name, transcripts, responses),
                                               rgb if name == 'Prompt A2' else None,
                                               limit, generation_kwargs)
                for item, obj in zip(responses, objects):
                    item[name] = obj
            return responses
        finally:
            self.train(was_training)
    def generate_text(self, rgb: Tensor, transcripts: Sequence[str], **kwargs) -> list[dict[str, str]]:
        """Return each stage's validated response serialized as JSON text.
        Unlike the original single-C API this returns four strings per sample.
        Use generate_json for structured fields; neither method calculates loss.
        """
        return [{name: json.dumps(obj[name], ensure_ascii=False) for name in PROMPT_ORDER}
        for obj in self.generate_json(rgb, transcripts, **kwargs)]
class AphasiaIntentPipeline(nn.Module):
    """Convenience composition of the same selector and shared Qwen wrapper.
    Accepts full-clip RGB; Qwen receives only the selector's selected_rgb, at A2.
    Existing callers may continue using the two components separately instead.
    """
    def __init__(self, selector: TemporalFrameSelector, llm: QwenIntentModel):
        super().__init__()
        if selector.num_frames != llm.num_frames:
            raise ValueError('Selector and Qwen must use the same num_frames')
        self.selector, self.llm = selector, llm
    def forward(self, landmarks: Tensor, rgb: Tensor, transcripts: Sequence[str],
        target_jsons: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
                *, frame_mask: Tensor | None = None, landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None, frame_times: Tensor | None = None,
        temperature: float | None = None, **prepare_kwargs) -> HierarchicalOutput:
        selection = self.selector(landmarks, frame_mask=frame_mask, landmark_mask=landmark_mask,
            visibility=visibility, rgb=rgb, frame_times=frame_times,
            temperature=temperature)
        prepared = self.llm.prepare_inputs(selection.selected_rgb, transcripts, target_jsons, **prepare_kwargs)
        result = self.llm(prepared=prepared)
        result.selection = selection
        return result
    @torch.no_grad()
    def generate_json(self, landmarks: Tensor, rgb: Tensor, transcripts: Sequence[str],
                      *, frame_mask: Tensor | None = None, landmark_mask: Tensor | None = None,
        visibility: Tensor | None = None, frame_times: Tensor | None = None,
                      **generation_kwargs) -> list[dict[str, dict[str, str]]]:
        was_training = self.training
        self.eval()
        try:
            selection = self.selector(landmarks, frame_mask=frame_mask, landmark_mask=landmark_mask,
            visibility=visibility, rgb=rgb, frame_times=frame_times)
            return self.llm.generate_json(selection.selected_rgb, transcripts, **generation_kwargs)
        finally:
            self.train(was_training)
