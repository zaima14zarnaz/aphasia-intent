#!/usr/bin/env python3
"""Summarize saved V2 body-part movement descriptions with Qwen3-VL.

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Examples:
    python prompt_t3.py --indir /path/to/sample_output/v2_frames.txt
    python prompt_t3.py --indir /path/to/sample_output
    python prompt_t3.py --indir /path/to/sample_output --input-name v2_grid.txt
    python prompt_t3.py --indir /path/to/sample_output --outdir /path/to/t3.txt

--indir accepts a V2 text/JSON file or a directory containing exactly one
v2*.txt or v2*.json file. Use --input-name to select among multiple V2 files.
The input is passed as text, including JSON formatting when present. No
images, contact sheet, or V1 output are loaded. The default response file is
t3.txt beside the V2 input. Total elapsed seconds are stored under "t3" in
latency.json beside that input, preserving other keys. Requires Python 3.10+.
"""

import argparse
import json
from pathlib import Path
import sys
import time


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# --llm qwen3 (default) uses the VL checkpoint above; qwen2 uses its
# Qwen2.5-VL counterpart so image inputs still work.
LLM_MODELS = {
    "qwen3": MODEL_ID,
    "qwen2": "Qwen/Qwen2.5-VL-7B-Instruct",
}
PROMPT_TEMPLATE = (
    "Describe the overall movement based on the hand, palm, finger and head "
    "movement descriptions in the input in a single paragraph.\n"
    "Individual body part movement description:\n"
    "{v2_text}\n"
    "Overall Movement description:"
)
SYSTEM_PROMPT = (
    "Use only the supplied movement descriptions. Preserve their uncertainty. "
    "Do not add gesture meanings or intentions. Return one paragraph of "
    "plain text without a heading, list, or JSON."
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_v2(indir: Path, input_name: str | None = None) -> Path:
    source = indir.expanduser().resolve()
    if source.is_file():
        if input_name is not None:
            raise ValueError("--input-name requires --indir to be a directory.")
        return source
    if not source.is_dir():
        raise ValueError(f"Input path does not exist: {source}")
    if input_name is not None:
        selected = source / input_name
        if not selected.is_file():
            raise ValueError(f"V2 input file not found: {selected}")
        return selected.resolve()
    candidates = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.name.lower().startswith("v2")
        and p.suffix.lower() in (".txt", ".json")
    )
    if not candidates:
        raise ValueError(f"No v2*.txt or v2*.json file found in {source}.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"Multiple V2 files found: {names}. Select one with --input-name.")
    return candidates[0]


def read_v2(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read V2 input {path}: {exc}") from exc
    if not text:
        raise ValueError(f"V2 input is empty: {path}")
    return text


def build_messages(v2_text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{
            "type": "text", "text": PROMPT_TEMPLATE.format(v2_text=v2_text),
        }]},
    ]


def resolve_output(target: Path | None, input_path: Path) -> Path:
    if target is None:
        return input_path.parent / "t3.txt"
    target = target.expanduser()
    if target.is_dir() or target.suffix.lower() != ".txt":
        return target / "t3.txt"
    return target


def record_latency(path: Path, key: str, seconds: float) -> None:
    data = {}
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Latency file must contain a JSON object: {path}")
        data[key] = round(seconds, 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot update latency file {path}: {exc}") from exc


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--indir", type=Path, required=True,
                        help="V2 output file or directory containing saved V2 output.")
    parser.add_argument("--input-name", help="V2 filename to select inside --indir.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--outdir", type=Path,
                              help="Output .txt file or directory for t3.txt.")
    output_group.add_argument("--output", type=Path, help="Output .txt file.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--latency-file", type=Path,
                        help="Defaults to latency.json beside the V2 input.")
    parser.add_argument("--latency-key", default="v3")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]

    try:
        input_path = select_v2(args.indir, args.input_name)
        v2_text = read_v2(input_path)
        output_path = (args.output.expanduser() if args.output is not None
                       else resolve_output(args.outdir, input_path))
        latency_path = (args.latency_file or input_path.parent / "latency.json").expanduser()
        if len({input_path.resolve(), output_path.resolve(), latency_path.resolve()}) != 3:
            raise ValueError("V2 input, T3 output, and latency file must have different paths.")
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import torch
        import torchvision  # Required by Qwen's AutoProcessor initialization.
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except (ImportError, RuntimeError) as exc:
        parser.exit(1, f"Missing or incompatible dependency: {exc}\n"
                    "Install compatible torch/torchvision builds, plus "
                    "transformers>=4.57.0, accelerate, and pillow.\n")

    print(f"V2 input: {input_path}", file=sys.stderr)
    print(f"Loading {args.model} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()
    text = processor.apply_chat_template(
        build_messages(v2_text), tokenize=False, add_generation_prompt=True,
    )
    # Text-only inference: deliberately omit all image/video inputs.
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
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
    # Normalize paragraph whitespace without changing the generated wording.
    response = " ".join(response.split())
    if not response:
        parser.exit(1, "The model returned an empty response; no output was saved.\n")
    print(response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save T3 response to {output_path}: {exc}\n")
    print(f"Saved response to {output_path.resolve()}", file=sys.stderr)

    elapsed = time.perf_counter() - start_time
    try:
        record_latency(latency_path, args.latency_key, elapsed)
        print(f"Latency ({args.latency_key}): {elapsed:.3f} s; "
              f"recorded in {latency_path.resolve()}", file=sys.stderr)
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
