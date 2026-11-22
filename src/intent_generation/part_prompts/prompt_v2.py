#!/usr/bin/env python3
"""Describe hand and head movement using interleaved frame images.

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Example:
    python qwen_temporal_movement_interleaved.py --indir /path/to/indir

Loads f_{i}.png (also accepts f{i}.png) from indir/frames when that folder
exists, or directly from indir otherwise, sorted by numeric frame index.
The prompt is followed by alternating frame labels and actual image inputs
in one request. No contact sheet or V1 descriptions are read.
The response defaults to sample_output/v2_frames.txt. Total elapsed seconds are
recorded under the "v2" key in latency.json, preserving other keys.
Requires Python 3.10+.
"""

import argparse
import json
from pathlib import Path
import sys
import re
import time


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# --llm qwen3 (default) uses the VL checkpoint above; qwen2 uses its
# Qwen2.5-VL counterpart so image inputs still work.
LLM_MODELS = {
    "qwen3": MODEL_ID,
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
}
SAMPLE_OUTPUT_DIR = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/sample_output"
)
LATENCY_FILE = SAMPLE_OUTPUT_DIR / "latency.json"
LATENCY_KEY = "v2"
import json

import json

PROMPT = json.dumps({
    "task": (
        "Infer the continuous hand and head movement connecting the "
        "chronologically ordered frames. Treat the frames as sampled "
        "moments from one movement sequence."
    ),
    "rules": [
        "Track the hand trajectory, palm movement and shape changes, finger movement and shape changes, and head movement across the frames.",
        "Combine small changes into an overall trajectory, including, curves, reversals, and repetition when supported.",
        "Describe movement directions using image-left, image-right, upward, and downward.",
        "Return only the JSON object below, replacing each description with a brief observation or supported movement inference."
    ],
    "output_format": {
        "Hand movement direction": "Overall hand trajectory and direction changes.",
        "Palm movement": "Changes in palm orientation or rotation.",
        "Finger movement": "Opening, closing, bending, or straightening.",
        "Notable finger shapes": "Distinct visible finger configurations.",
        "Notable palm shapes": "Distinct palm configurations, such as flat or cupped.",
        "Head configuration": "General head position and orientation.",
        "Head movement": "Turns, tilts, nods, or stillness.",
    }
}, indent=2) + "\nImages:"

SYSTEM_PROMPT = (

)

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


def build_messages(paths: list[Path]) -> list[dict]:
    # Each image placeholder is filled with its matching PIL image by processor.
    # These are multimodal image inputs, not literal <f0> strings alone.
    content = [{"type": "text", "text": PROMPT}]
    for position, path in enumerate(paths, start=1):
        index = int(FRAME_NAME.fullmatch(path.name).group(1))
        content.append({"type": "text", "text":
                        f"\nFrame f{index} (chronological position {position} of {len(paths)}):"})
        content.append({"type": "image"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def resolve_output(args) -> Path:
    if args.outdir is not None:
        target = args.outdir.expanduser()
        # Retain --outdir support for either a .txt file or a directory.
        if target.is_dir() or target.suffix.lower() != ".txt":
            return target / "v2_frames.txt"
        return target
    return (args.output or args.sample_output_dir / "v2_frames.txt").expanduser()


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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--outdir", type=Path,
                              help="Output .txt file or directory for v2_frames.txt.")
    output_group.add_argument("--output", type=Path, help="Output text file.")
    parser.add_argument("--vis-source", choices=("frames",), default="frames",
                        help="Only frames is supported; this flag is optional.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--sample-output-dir", type=Path, default=SAMPLE_OUTPUT_DIR,
                        help="Directory for the default v2_frames.txt response.")
    parser.add_argument("--latency-file", type=Path, default=LATENCY_FILE)
    parser.add_argument("--latency-key", default=LATENCY_KEY)
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]
    output_path = resolve_output(args)
    latency_path = args.latency_file.expanduser()

    try:
        paths = select_frames(args.indir)
        if output_path.resolve() in paths:
            raise ValueError("Output must differ from every input frame.")
        if latency_path.resolve() in [*paths, output_path.resolve()]:
            raise ValueError("Latency file must differ from the images and response file.")
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
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()

    text = processor.apply_chat_template(
        build_messages(paths), tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    print(response)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save V2 response to {output_path}: {exc}\n")
    print(f"Saved response to {output_path.resolve()}", file=sys.stderr)

    # The decoded CPU text above ensures generated tokens are ready.
    elapsed = time.perf_counter() - start_time
    try:
        latency_path = record_latency(latency_path, args.latency_key, elapsed)
        print(f"Latency ({args.latency_key}): {elapsed:.3f} s; "
              f"recorded in {latency_path.resolve()}", file=sys.stderr)
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
