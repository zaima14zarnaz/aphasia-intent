#!/usr/bin/env python3
"""Infer communicative intent from frames and transcript in one Qwen call.

Keep prompt_t1.py beside this script to reuse TRANSCRIPT without running T1.
Frames named f_{i}.png or f{i}.png are loaded from indir/frames if present,
otherwise directly from indir. Frames are sorted numerically and interleaved
with labels specifying their chronological order in a single user message.
The transcript and requested JSON structure follow the frames.

The output retains MM3's top-level keys; evidence contains only transcript.
Selection fields are marked not applicable because no candidate list is
supplied. Frames and transcript are both used for the single inference call;
no previous-stage files are used. The response is validated before saving. Use --llm qwen2
for Qwen2.5-VL-7B-Instruct or --llm qwen3 for Qwen3-VL-8B-Instruct (default).

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Example:
    python prompt_full.py --indir /path/to/clip --out-dir /path/to/sample_output

--out-dir accepts a directory (writes full.txt) or an explicit .txt/.json file.
The response is also printed to stdout. Total elapsed run time, including
model loading and generation, is saved in seconds under the "full" key in
latency.json beside the output. Existing latency keys are preserved.
Override that file with --latency-file. Requires Python 3.10+.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import re
import time


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LLM_MODELS = {
    "qwen3": MODEL_ID,
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
}
NOT_APPLICABLE = "Not applicable (single-pass inference)"
OUTPUT_FORMAT = {
    "selected candidate": NOT_APPLICABLE,
    "selected candidate intent": NOT_APPLICABLE,
    "final communicative intent": (
        "One sentence stating what the speaker appears to be trying to communicate; "
        "if the purpose is unclear, say so"
    ),
    "evidence": {
        "transcript": "Relevant words and what they support",
    },
    "reason": "A brief explanation connecting the transcript and gestures to the inferred intent",
    "uncertainty": "Ambiguities, conflicting evidence, or unsupported aspects of the intent",
}
PROMPT = (
    "Describe the communicative intent of the person based on the transcript "
    "and the gestures made on the frames."
)
SYSTEM_PROMPT = (
    "Images are ordered chronologically, earliest to latest. Use the transcript "
    "and visible gestures together to infer what the speaker wants the listener "
    "to understand or do, rather than only describing hand actions. "
    "No candidate list is supplied: copy the two not-applicable selection values "
    "exactly. The evidence object must contain only transcript: cite words from "
    "the transcript and what they support, not visual observations or generated "
    "context. Use visible gestures in the combined interpretation and reason. "
    "Separate observations from possible meanings. "
    "Do not invent prior-stage outputs, observed object contact, or unsupported goals. "
    "Repetition or pauses alone do not establish instructional intent or emphasis. "
    "Acknowledge uncertainty and keep fields brief. Return only the requested "
    "valid JSON object, without Markdown or extra text."
)



def load_transcript() -> str:
    try:
        from prompt_t1 import TRANSCRIPT
    except ImportError as exc:
        raise ValueError(
            "Keep prompt_t1.py beside prompt_full.py and define TRANSCRIPT in it."
        ) from exc
    if not isinstance(TRANSCRIPT, str) or not TRANSCRIPT.strip():
        raise ValueError("TRANSCRIPT in prompt_t1.py must be a nonempty string.")
    return TRANSCRIPT.strip()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


FRAME_NAME = re.compile(r"f_?(\d+)\.png")


def select_frames(indir: Path) -> list[Path]:
    indir = indir.expanduser().resolve()
    if not indir.is_dir():
        raise ValueError(f"Input directory does not exist: {indir}")
    frame_dir = indir / "frames" if (indir / "frames").is_dir() else indir
    indexed = []
    seen = set()
    for path in frame_dir.iterdir():
        match = FRAME_NAME.fullmatch(path.name)
        if path.is_file() and match:
            index = int(match.group(1))
            if index in seen:
                raise ValueError(f"Duplicate frame index {index} in {frame_dir}")
            seen.add(index)
            indexed.append((index, path))
    if not indexed:
        raise ValueError(f"No frames named f_{{i}}.png or f{{i}}.png in {frame_dir}")
    return [path for _, path in sorted(indexed)]


def load_images(paths: list[Path]) -> list:
    from PIL import Image, ImageOps

    images = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        except OSError as exc:
            raise ValueError(f"Cannot read image {path}: {exc}") from exc
    return images


def load_fresh_model(model_id: str, device_map: str):
    """Load the selected vision-language checkpoint without prior chat state."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="auto", device_map=device_map, attn_implementation="sdpa",
    ).eval()
    return processor, model


def build_messages(paths: list[Path], transcript: str) -> list[dict]:
    content = []
    for position, path in enumerate(paths, start=1):
        index = int(FRAME_NAME.fullmatch(path.name).group(1))
        content.extend([
            {"type": "text", "text":
             f"Frame f{index} (chronological position {position} of {len(paths)}):"},
            {"type": "image"},
        ])
    # Keep the transcript, task, and JSON schema after all image blocks.
    content.append({
        "type": "text",
        "text": f"Transcript: {transcript}\n\n{PROMPT}\nOutput JSON:\n"
                + json.dumps(OUTPUT_FORMAT, indent=2),
    })
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def parse_response(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Model response contains an unclosed JSON code fence.")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != set(OUTPUT_FORMAT):
        raise ValueError("Model response must contain exactly the requested top-level fields.")
    for key in OUTPUT_FORMAT:
        if key != "evidence" and (not isinstance(data[key], str) or not data[key].strip()):
            raise ValueError(f"{key!r} must be a nonempty string.")
    for key in ("selected candidate", "selected candidate intent"):
        if data[key].strip() != NOT_APPLICABLE:
            raise ValueError(f"{key!r} must be {NOT_APPLICABLE!r}; no candidate list was supplied.")
    evidence = data["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(OUTPUT_FORMAT["evidence"]):
        raise ValueError("Evidence must contain only the transcript field.")
    if any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
        raise ValueError("Each evidence field must be a nonempty string.")
    return {
        key: ({source: evidence[source].strip() for source in OUTPUT_FORMAT["evidence"]}
              if key == "evidence" else data[key].strip())
        for key in OUTPUT_FORMAT
    }


def resolve_output(target: Path) -> Path:
    target = target.expanduser()
    if target.is_dir() or target.suffix.lower() not in (".txt", ".json"):
        return target / "full.txt"
    return target


def record_latency(path: Path, key: str, elapsed_seconds: float) -> Path:
    """Update one latency value while preserving the other JSON keys."""
    path = path.expanduser()
    data = {}
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Latency file must contain a JSON object: {path}")
        data[key] = round(float(elapsed_seconds), 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot update latency file {path}: {exc}") from exc
    return path


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indir", type=Path, required=True,
                        help="Directory containing frames/ or the frame PNG files themselves.")
    parser.add_argument("--out-dir", "--outdir", type=Path, required=True,
                        help="Output directory for full.txt, or explicit .txt/.json file.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="qwen3: Qwen3-VL-8B; qwen2: Qwen2.5-VL-7B.")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path overriding --llm.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--latency-file", type=Path,
                        help="Defaults to latency.json beside full.txt; updates the full key.")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]
    output_path = resolve_output(args.out_dir)
    latency_path = (args.latency_file or output_path.parent / "latency.json").expanduser()

    try:
        transcript = load_transcript()
        paths = select_frames(args.indir)
        input_paths = [*paths, Path(__file__).resolve().with_name("prompt_t1.py")]
        if output_path.resolve() in input_paths:
            raise ValueError("Output must differ from the frame inputs and transcript script.")
        if latency_path.resolve() in [*input_paths, output_path.resolve()]:
            raise ValueError("Latency file must differ from all input and response files.")
        images = load_images(paths)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))

    try:
        import torch
        import torchvision  # Qwen's processor requires this even for images.
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except (ImportError, RuntimeError) as exc:
        parser.exit(1, f"Missing or incompatible dependency: {exc}\n"
                    "Install compatible torch/torchvision builds, plus "
                    "transformers>=4.57.0, accelerate, and pillow.\n")

    print(f"Interleaved frame images: {len(paths)}", file=sys.stderr)
    for path in paths:
        print(f"  {path}", file=sys.stderr)
    print(f"Loading {args.model} ...", file=sys.stderr)
    processor, model = load_fresh_model(args.model, args.device_map)

    text = processor.apply_chat_template(
        build_messages(paths, transcript), tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    assert "past_key_values" not in inputs
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    # Release generation state before output validation, including invalid JSON.
    del inputs, generated_ids, new_ids, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        result = parse_response(response)
    except ValueError as exc:
        print(f"Raw model response:\n{response}", file=sys.stderr)
        parser.exit(1, f"{exc}\nNo output was saved.\n")
    response = json.dumps(result, ensure_ascii=False, indent=2)
    print(response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save full-prompt response to {output_path}: {exc}\n")
    print(f"Saved response to {output_path.resolve()}", file=sys.stderr)

    # The decoded CPU text above ensures generated tokens are ready.
    elapsed = time.perf_counter() - start_time
    try:
        latency_path = record_latency(latency_path, "full", elapsed)
        print(f"Latency (full): {elapsed:.3f} s; "
              f"recorded in {latency_path.resolve()}", file=sys.stderr)
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
