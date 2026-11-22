#!/usr/bin/env python3
"""Describe body configurations with Qwen3-VL-8B-Instruct.

Install (use a PyTorch build suitable for your CUDA driver):
    pip install -U torch 'transformers>=4.57.0' accelerate pillow

Examples:
    python prompt_v1.py --indir /path/to/inputs --outdir /path/to/outputs --vis-source frames
    python prompt_v1.py --outdir /path/to/dir --vis-source grid

Frames mode sends all f_{i}.png images, sorted by integer i, in ONE request.
Grid mode sends only indir/contact_sheet.png in ONE request.
The response is always printed and saved to <outdir>/response.txt
(or to --output if given). If --indir is omitted, --outdir is also used
as the input directory.
At the end of the run, total elapsed time (seconds) is recorded under
the "v1" key of sample_output/latency.json (--latency-file/--latency-key).
The first run downloads the model and processor from Hugging Face.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# --llm qwen3 (default) uses the VL checkpoint above; qwen2 uses its
# Qwen2.5-VL counterpart so image inputs still work.
LLM_MODELS = {
    "qwen3": MODEL_ID,
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
}
PROMPT = "Describe the body configuration in each of these frames."
FRAME_NAME = re.compile(r"f_(\d+)\.png")
LATENCY_FILE = Path("/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/sample_output/latency.json")
LATENCY_KEY = "v1"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_images(indir: Path, vis_source: str) -> list[Path]:
    """Find only the requested inputs; never include the grid in frame mode."""
    indir = indir.expanduser().resolve()
    if not indir.is_dir():
        raise ValueError(f"Input directory does not exist: {indir}")

    if vis_source == "grid":
        grid = indir / "contact_sheet.png"
        if not grid.is_file():
            raise ValueError(f"Contact sheet not found: {grid}")
        return [grid]
    if vis_source != "frames":
        raise ValueError(f"Unknown visual source: {vis_source}")

    frames = []
    frame_dir = indir / "frames"
    for path in frame_dir.iterdir():
        match = FRAME_NAME.fullmatch(path.name)
        if path.is_file() and match:
            frames.append((int(match.group(1)), path.name, path))
    if not frames:
        raise ValueError(f"No frames named f_{{i}}.png found in {indir}")
    return [path for _, _, path in sorted(frames)]


def record_latency(path: Path, key: str, elapsed_seconds: float) -> Path:
    """Set latency.json[key] = elapsed seconds, preserving other keys."""
    path = path.expanduser()
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot update latency file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Latency file must contain a JSON object: {path}")
    data[key] = round(float(elapsed_seconds), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    return path


def load_images(paths: list[Path]) -> list:
    """Load detached RGB images so source files can close immediately."""
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
    """Attach each filename to its image placeholder, then the exact prompt."""
    content = []
    for path in paths:
        content.append({"type": "text", "text": path.name})
        content.append({"type": "image"})
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", type=Path,
                        help="Directory containing f_{i}.png and/or contact_sheet.png. "
                             "Defaults to --outdir if omitted.")
    parser.add_argument("--outdir", type=Path,
                        help="Directory for response.txt (created if needed). "
                             "Also used as the input directory if --indir is omitted.")
    parser.add_argument("--vis-source", choices=("frames", "grid"), required=True)
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto",
                        help="Model placement, e.g. auto, cuda:0, or cpu (default: auto).")
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--output", type=Path,
                        help="Optional text file for the response (also printed to stdout). "
                             "Overrides <outdir>/response.txt.")
    parser.add_argument("--latency-file", type=Path, default=LATENCY_FILE,
                        help=f"JSON file to record total run latency in seconds (default: {LATENCY_FILE}).")
    parser.add_argument("--latency-key", default=LATENCY_KEY,
                        help=f"Key in the latency JSON to update (default: {LATENCY_KEY}).")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]
    if args.indir is None and args.outdir is None:
        parser.error("either --indir or --outdir is required")
    if args.indir is None:
        args.indir = args.outdir

    try:
        paths = select_images(args.indir, args.vis_source)
        images = load_images(paths)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        parser.exit(1, f"Missing or outdated dependency: {exc}\n"
                    "Install: pip install -U torch 'transformers>=4.57.0' accelerate pillow\n")

    print(f"Visual source: {args.vis_source}; images: {len(paths)}", file=sys.stderr)
    for path in paths:
        print(f"  {path}", file=sys.stderr)
    print(f"Loading {args.model} ...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()

    # Placeholders and PIL images share the same order. No qwen-vl-utils needed.
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

    # Decode new tokens only, excluding the user's prompt and image placeholders.
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    print(response)

    if args.output:
        output_path = args.output.expanduser()
    elif args.outdir:
        output_path = args.outdir.expanduser() 
    else:
        output_path = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
        print(f"Saved response to {output_path.resolve()}", file=sys.stderr)

    # Report total run latency (seconds) under the v1 key, preserving v2..v10.
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    elapsed = time.perf_counter() - start_time
    try:
        latency_path = record_latency(args.latency_file, args.latency_key, elapsed)
        print(f"Latency ({args.latency_key}): {elapsed:.3f} s; recorded in {latency_path.resolve()}", file=sys.stderr)
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
