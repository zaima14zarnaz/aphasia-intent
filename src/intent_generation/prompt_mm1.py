#!/usr/bin/env python3
"""Generate three candidate gesture meanings from transcript, gesture, and context.

Keep prompt_t1.py beside this script: its TRANSCRIPT variable is imported
without running T1 inference. Context and gesture files may contain plain text
or JSON; their full contents are supplied to Qwen as text.

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Example:
    python prompt_mm1.py --context-file sample_output/t1.txt \
        --gest-desc-file sample_output/t3.txt --outdir sample_output/mm1.json

--outdir accepts a .json/.txt file or a directory, where mm1.json is created.
Only validated JSON containing exactly three labeled candidate meanings is
saved. Run latency is recorded under "mm1" in latency.json beside the output.
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
CANDIDATE_KEYS = tuple(f"gesture meaning candidate {i}" for i in range(1, 4))
OUTPUT_FORMAT = {key: "..." for key in CANDIDATE_KEYS}
PROMPT_TEMPLATE = (
    "Generate three candidate meanings for the gesture based on the description "
    "of the gesture movements, the transcript, and the conversation context.\n"
    "Transcript: {transcript}\n"
    "Gesture description: {gesture_description}\n"
    "Context: {context}"
)
SYSTEM_PROMPT = (
    "Generate three distinct possible gesture meanings supported by the inputs. "
    "Present them as possibilities, not established facts. Do not choose a final "
    "meaning or describe the speaker's communicative intent. "
    "Return only a valid JSON object with exactly the requested keys and "
    "nonempty string values. Do not include Markdown fences or explanations."
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_transcript() -> str:
    try:
        from prompt_t1 import TRANSCRIPT
    except ImportError as exc:
        raise ValueError(
            "Keep prompt_t1.py beside prompt_mm1.py and define TRANSCRIPT in it."
        ) from exc
    if not isinstance(TRANSCRIPT, str) or not TRANSCRIPT.strip():
        raise ValueError("TRANSCRIPT in prompt_t1.py must be a nonempty string.")
    return TRANSCRIPT.strip()


def read_input(path: Path, label: str) -> str:
    try:
        text = path.expanduser().read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {label} file {path}: {exc}") from exc
    if not text:
        raise ValueError(f"{label} file is empty: {path}")
    return text


def build_messages(transcript: str, gesture_description: str, context: str) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(
        transcript=transcript, gesture_description=gesture_description, context=context,
    ) + "\nOutput format JSON:\n" + json.dumps(OUTPUT_FORMAT, indent=2)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def resolve_output(target: Path) -> Path:
    target = target.expanduser()
    if target.is_dir() or target.suffix.lower() not in (".json", ".txt"):
        return target / "mm1.json"
    return target


def parse_candidates(response: str) -> dict[str, str]:
    text = response.strip()
    # Accept a surrounding Markdown code fence, but save only clean JSON.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("The model returned an unclosed JSON code fence.")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != set(CANDIDATE_KEYS):
        raise ValueError("Expected one JSON object with exactly the three candidate keys.")
    if any(not isinstance(data[key], str) or not data[key].strip() for key in CANDIDATE_KEYS):
        raise ValueError("Each candidate meaning must be a nonempty string.")
    return {key: data[key].strip() for key in CANDIDATE_KEYS}


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
    parser.add_argument("--context-file", type=Path, required=True,
                        help="Text or JSON file containing the conversation context.")
    parser.add_argument("--gest-desc-file", type=Path, required=True,
                        help="Text or JSON file containing the gesture movement description.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output .json/.txt file or directory for mm1.json.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--latency-file", type=Path,
                        help="Defaults to latency.json beside the MM1 output.")
    parser.add_argument("--latency-key", default="mm1")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]

    try:
        transcript = load_transcript()
        context = read_input(args.context_file, "Context")
        gesture_description = read_input(args.gest_desc_file, "Gesture description")
        output_path = resolve_output(args.outdir)
        latency_path = (args.latency_file or output_path.parent / "latency.json").expanduser()
        input_paths = {
            args.context_file.expanduser().resolve(),
            args.gest_desc_file.expanduser().resolve(),
            Path(__file__).resolve().with_name("prompt_t1.py"),
        }
        if output_path.resolve() == latency_path.resolve():
            raise ValueError("MM1 output and latency file must have different paths.")
        if input_paths.intersection({output_path.resolve(), latency_path.resolve()}):
            raise ValueError("Output and latency paths must differ from the input files.")
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
        build_messages(transcript, gesture_description, context), tokenize=False, add_generation_prompt=True,
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
    try:
        candidates = parse_candidates(response)
    except ValueError as exc:
        print(f"Raw model response:\n{response}", file=sys.stderr)
        parser.exit(1, f"{exc}\nNo candidate output was saved.\n")
    response = json.dumps(candidates, ensure_ascii=False, indent=2)
    print(response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save MM1 response to {output_path}: {exc}\n")
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
