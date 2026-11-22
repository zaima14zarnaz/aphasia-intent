#!/usr/bin/env python3
"""Select one final communicative intent using all supplied evidence.

Keep prompt_t1.py beside this script to reuse its TRANSCRIPT variable without
running T1 inference. Supply context/keywords, gesture description, gesture
candidate meanings, and the JSON candidate-intent output from prompt_mm2.py.
No images are loaded. The selected candidate's original text is preserved;
the final statement clarifies its communicative purpose using the evidence.

Install compatible torch/torchvision builds for your CUDA runtime:
    pip install -U torch torchvision 'transformers>=4.57.0' accelerate pillow

Example:
    python prompt_mm3.py --context-file sample_output/t1.txt --gest-desc-file sample_output/t3.txt --candidate-gesture-meanings sample_output/mm1.json --candidate-intents sample_output/mm2.json --outdir sample_output/mm3.json

--outdir accepts a .json/.txt file or a directory, where mm3.json is created.
The saved JSON contains the selected candidate and original text, a clarified
communicative intent, evidence by source, a brief reason, and uncertainty.
Run latency is stored under "mm3" in latency.json beside the response.
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
OUTPUT_FORMAT = {
    "selected candidate": "Exact key of one candidate in the supplied list",
    "selected candidate intent": "Copy that candidate's intent text exactly",
    "final communicative intent": (
        "One sentence stating what the speaker is trying to communicate. "
        "Clarify the selected candidate using the evidence without adding a new goal. "
        "If the purpose is unclear, say so."
    ),
    "evidence": {
        "transcript": "Relevant words and what they support",
        "gesture description": (
            "Relevant described movements and what they support; do not claim to see the frames"
        ),
        "gesture candidate meanings": "The relevant possible meaning, kept tentative",
        "context and keywords": "Relevant context and keywords supporting the interpretation",
    },
    "reason": (
        "A brief explanation of why the selected candidate fits the combined evidence "
        "better than the alternatives; if there is only one candidate, assess its fit"
    ),
    "uncertainty": "Ambiguities, conflicting evidence, or unsupported aspects of the intent",
}
PROMPT_TEMPLATE = (
    "Select the candidate communicative intent most consistent with the transcript, "
    "gesture description, gesture candidate meanings, and context. "
    "Preserve the original candidate text, then state its supported communicative "
    "purpose clearly. Distinguish what the gesture depicts from what the speaker "
    "is trying to communicate. Give brief evidence by source and state uncertainty.\n"
    "Transcript: {transcript}\n"
    "Gesture description: {gesture_description}\n"
    "Gesture candidate meanings: {candidate_gesture_meanings}\n"
    "Context and keywords: {context}\n"
    "List of candidate intents: {candidate_intents}"
)
SYSTEM_PROMPT = (
    "Select exactly one candidate from the supplied list, considering all inputs. "
    "Copy its key and original intent exactly into the selection fields. "
    "The final intent may clarify that candidate's communicative purpose, but must "
    "not introduce an unsupported goal. If the candidate only describes a gesture "
    "action and the speaker's purpose is unclear, acknowledge that limitation. "
    "Ground each evidence field in its named input. You have descriptions, not images. "
    "Treat gesture meanings as hypotheses and do not treat transcript-derived context "
    "as independent confirmation. Do not assume pauses or repetition mean emphasis. "
    "Avoid certainty beyond the evidence. Keep each field brief. "
    "Return only the requested valid JSON object without Markdown."
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
            "Keep prompt_t1.py beside prompt_mm3.py and define TRANSCRIPT in it."
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


def build_messages(
    transcript: str, gesture_description: str, candidate_gesture_meanings: str,
    context: str, candidate_intents: str,
) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(
        transcript=transcript, gesture_description=gesture_description,
        candidate_gesture_meanings=candidate_gesture_meanings,
        context=context, candidate_intents=candidate_intents,
    ) + "\nOutput JSON:\n" + json.dumps(OUTPUT_FORMAT, indent=2)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def resolve_output(target: Path) -> Path:
    target = target.expanduser()
    if target.is_dir() or target.suffix.lower() not in (".json", ".txt"):
        return target / "mm3.json"
    return target


def parse_json_object(text: str, label: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError(f"{label} contains an unclosed JSON code fence.")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{label} must be a nonempty JSON object.")
    return data


def parse_intent_candidates(text: str) -> dict[str, str]:
    """Accept MM2's nested intent/clue objects, or labeled intent strings."""
    data = parse_json_object(text, "Candidate intents file")
    intents = {}
    for key, value in data.items():
        intent = value.get("intent") if isinstance(value, dict) else value
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Every candidate intent needs a nonempty label.")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError(f"Candidate {key!r} must contain a nonempty intent string.")
        intents[key] = intent.strip()
    return intents


def parse_selection(response: str, intents: dict[str, str]) -> dict:
    data = parse_json_object(response, "Model response")
    if set(data) != set(OUTPUT_FORMAT):
        raise ValueError("Model response must contain exactly the requested output fields.")
    string_keys = tuple(key for key in OUTPUT_FORMAT if key != "evidence")
    if any(not isinstance(data[key], str) or not data[key].strip() for key in string_keys):
        raise ValueError("Each selection, intent, reason, and uncertainty field must be a nonempty string.")
    key = data["selected candidate"].strip()
    if key not in intents:
        raise ValueError(f"Selected candidate {key!r} is not in the supplied list.")
    intent = data["selected candidate intent"].strip()
    if intent != intents[key]:
        raise ValueError("Selected candidate intent must match the original candidate text exactly.")
    evidence = data["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(OUTPUT_FORMAT["evidence"]):
        raise ValueError("Evidence must contain exactly the four requested source fields.")
    if any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
        raise ValueError("Each evidence field must be a nonempty string.")
    # Validate identity and structure. Grounding of the clarified purpose still
    # requires reviewing the evidence; JSON validation cannot establish it.
    return {
        "selected candidate": key,
        "selected candidate intent": intents[key],
        "final communicative intent": data["final communicative intent"].strip(),
        "evidence": {source: evidence[source].strip() for source in OUTPUT_FORMAT["evidence"]},
        "reason": data["reason"].strip(),
        "uncertainty": data["uncertainty"].strip(),
    }


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
    parser.add_argument("--candidate-gesture-meanings", type=Path, required=True,
                        help="File containing candidate gesture meanings, e.g. mm1.json.")
    parser.add_argument("--gest-desc-file", type=Path, required=True,
                        help="File containing the gesture movement description.")
    parser.add_argument("--candidate-intents", "--candidate-intents-file", type=Path, required=True,
                        help="JSON candidate-intent output from prompt_mm2.py.")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output .json/.txt file or directory for mm3.json.")
    parser.add_argument("--llm", choices=("qwen3", "qwen2"), default="qwen3",
                        help="Model family: qwen3 (Qwen3-VL-8B-Instruct) or qwen2 (Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--model", default=None,
                        help="Explicit checkpoint ID/path; overrides the --llm default.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument("--latency-file", type=Path,
                        help="Defaults to latency.json beside the MM3 output.")
    parser.add_argument("--latency-key", default="mm3")
    args = parser.parse_args()
    args.model = args.model or LLM_MODELS[args.llm]

    try:
        transcript = load_transcript()
        context = read_input(args.context_file, "Context")
        candidate_gesture_meanings = read_input(
            args.candidate_gesture_meanings, "Candidate gesture meanings",
        )
        gesture_description = read_input(args.gest_desc_file, "Gesture description")
        candidate_intents = read_input(args.candidate_intents, "Candidate intents")
        intents = parse_intent_candidates(candidate_intents)
        output_path = resolve_output(args.outdir)
        latency_path = (args.latency_file or output_path.parent / "latency.json").expanduser()
        input_paths = {
            args.context_file.expanduser().resolve(),
            args.candidate_gesture_meanings.expanduser().resolve(),
            args.candidate_intents.expanduser().resolve(),
            args.gest_desc_file.expanduser().resolve(),
            Path(__file__).resolve().with_name("prompt_t1.py"),
        }
        if output_path.resolve() == latency_path.resolve():
            raise ValueError("MM3 output and latency file must have different paths.")
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
        build_messages(transcript, gesture_description, candidate_gesture_meanings,
                       context, candidate_intents),
        tokenize=False, add_generation_prompt=True,
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
        selection = parse_selection(response, intents)
    except ValueError as exc:
        print(f"Raw model response:\n{response}", file=sys.stderr)
        parser.exit(1, f"{exc}\nNo final-intent output was saved.\n")
    response = json.dumps(selection, ensure_ascii=False, indent=2)
    print(response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not save MM3 response to {output_path}: {exc}\n")
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
