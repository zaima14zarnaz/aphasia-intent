#!/usr/bin/env python3
"""Extract speech context and keywords from a transcript using Qwen3-VL.

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Examples:
    python prompt_t1.py
    python prompt_t1.py --outdir /path/to/sample_output/t1.txt
    python prompt_t1.py --transcript 'The bread, spread the peanut butter.'
    python prompt_t1.py --transcript-file /path/to/transcript.txt

Uses the editable TRANSCRIPT variable by default. The response is printed and
saved as t1.txt in SAMPLE_OUTPUT_DIR, or at --outdir / --output. Run latency
is stored under "t1" in latency.json beside the response file. This is a
text-only stage; no images or previous-stage outputs are loaded.
Requires Python 3.10+.
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
SAMPLE_OUTPUT_DIR = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/sample_output"
)
# Edit this variable to process a different transcript, or use a CLI override.
TRANSCRIPT = (
    '“the bread” [pause] “spread” [gesture] “spread” [gesture] '
    '[pause] “the peanut butter.”'
)
PROMPT_TEMPLATE = (
    "Read the transcript. Summarize its topic and list important words or phrases. "
    "Use only the transcript. Ignore pause and frame markers.\n"
    "Transcript: {transcript}\n"
    "Context: <one short sentence>\n"
    "Keywords: <comma-separated list>"
)
SYSTEM_PROMPT = (
    "Use only the transcript. Ignore bracketed annotation markers. "
    "Do not infer gesture meanings or speaker intentions. "
    "Return exactly two lines: Context: followed by one short sentence, "
    "and Keywords: followed by a comma-separated list."
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def get_transcript(args) -> str:
    if args.transcript_file is not None:
        try:
            text = args.transcript_file.expanduser().read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read transcript file: {exc}") from exc
    else:
        text = args.transcript if args.transcript is not None else TRANSCRIPT
    text = text.strip()
    if not text:
        raise ValueError("Transcript must not be empty.")
    return text


def build_messages(transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{
            "type": "text", "text": PROMPT_TEMPLATE.format(transcript=transcript),
        }]},
    ]


def resolve_output(target: Path | None) -> Path:
    if target is None:
        return SAMPLE_OUTPUT_DIR / "t1.txt"
    target = target.expanduser()
    if target.is_dir() or target.suffix.lower() != ".txt":
        return target / "t1.txt"
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
    transcript_group = parser.add_mutually_exclusive_group()
    transcript_group.add_argument("--transcript", help="Override the TRANSCRIPT variable.")
    transcript_group.add_argument("--transcript-file", type=Path,
                                  help="Read the transcript from a UTF-8 text file.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--outdir", type=Path,
                              help="Output .txt file or directory for t1.txt.")
    output_group.add_argument("--output", type=Path, help="Output .txt file.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--latency-file", type=Path,
                        help="Defaults to latency.json beside the T1 output.")
    parser.add_argument("--latency-key", default="t1")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]

    try:
        transcript = get_transcript(args)
        output_path = (args.output.expanduser() if args.output is not None
                       else resolve_output(args.outdir))
        latency_path = (args.latency_file or output_path.parent / "latency.json").expanduser()
        if output_path.resolve() == latency_path.resolve():
            raise ValueError("T1 output and latency file must have different paths.")
        if args.transcript_file is not None:
            input_path = args.transcript_file.expanduser().resolve()
            if input_path in (output_path.resolve(), latency_path.resolve()):
                raise ValueError("Output and latency paths must differ from the transcript file.")
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

    print(f"Loading {args.model} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()
    text = processor.apply_chat_template(
        build_messages(transcript), tokenize=False, add_generation_prompt=True,
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
    if not response:
        parser.exit(1, "The model returned an empty response; no output was saved.\n")
    print(response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save T1 response to {output_path}: {exc}\n")
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
