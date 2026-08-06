"""Hugging Face Qwen vision-language model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image

ImageInput = Union[str, Path, Image.Image]

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class VLLM:
    """Wrapper around a Hugging Face Qwen vision-language model."""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device_map: str | dict = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install it with: pip install transformers"
            ) from exc

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    @staticmethod
    def _normalize_image(image: ImageInput) -> ImageInput:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return image

    def generate_response(
        self,
        prompt: str,
        image: Optional[ImageInput] = None,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the Qwen model.

        Args:
            prompt: Text prompt for the model.
            image: Optional image path or PIL image. When ``None``, only the text
                prompt is used.
            max_new_tokens: Override the default generation length for this call.

        Returns:
            The model's generated text response.
        """
        if image is None:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        else:
            image_input = self._normalize_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                if isinstance(image_input, Image.Image):
                    pil_image = image_input
                else:
                    pil_image = Image.open(image_input).convert("RGB")
                inputs = self.processor(
                    text=[text],
                    images=[pil_image],
                    padding=True,
                    return_tensors="pt",
                )

        inputs = inputs.to(self.model.device)
        generation_length = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_length,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
