"""Validation generation metrics for intent/reasoning, with multiple references.

All requested metrics are enabled; missing dependencies/checkpoints raise errors.
BLEURT runs through the official package in a separate CPU Python process.
No training loss or optimizer operations are defined here. See METRICS_USAGE.md.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from tqdm.auto import tqdm

# Keep the worker importable in a TensorFlow-only environment without torch.
EXPECTED_FIELDS = ("communicative intent", "reasoning")
if TYPE_CHECKING:
    from intent_output_formatter import FormattedIntentOutput

DEFAULT_METRICS_CSV = Path("/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/runs/metrics.csv")
PAIR_METRICS = ("bertscore_f1", "rouge1_f1", "rouge2_f1", "rougeL_f1", "meteor", "bleurt")


@dataclass
class MetricsConfig:
    bertscore_model: str = "roberta-large"
    bertscore_num_layers: int | None = None
    device: str = "cpu"
    batch_size: int = 8
    bleurt_checkpoint: str = "checkpoints/BLEURT-20"
    bleurt_python: str = sys.executable
    bleurt_timeout: float = 1800.0
    # Temporary kill-switch used by train.py / train_motion.py.
    enable_bleurt: bool = True


@dataclass
class ValidationSample:
    media_id: str
    transcript: str
    prediction: FormattedIntentOutput
    references: dict[str, dict[str, str]] = field(default_factory=dict)


class ValidationAccumulator:
    """One prediction per (media ID, transcript), preserving every annotation row."""

    def __init__(self):
        self.samples: dict[tuple[str, str], ValidationSample] = {}
        self.reference_keys: dict[str, tuple[str, str]] = {}

    def update(self, *, media_ids: Sequence[str], transcripts: Sequence[str],
               predictions: Sequence[FormattedIntentOutput],
               references: Sequence[Mapping[str, str]], reference_ids: Sequence[str]) -> None:
        n = len(media_ids)
        if any(len(values) != n for values in (transcripts, predictions, references, reference_ids)):
            raise ValueError("Validation prediction/reference batch lengths differ")
        for media, transcript, predicted, reference, reference_id in zip(media_ids, transcripts, predictions, references, reference_ids):
            key = (media, transcript)
            if reference_id in self.reference_keys and self.reference_keys[reference_id] != key:
                raise ValueError(f"Reference ID assigned to different inputs: {reference_id}")
            target = {field: reference[field] for field in EXPECTED_FIELDS}
            if any(not isinstance(text, str) or not text.strip() for text in target.values()):
                raise ValueError("Metric references require nonempty intent and reasoning")
            if key not in self.samples:
                self.samples[key] = ValidationSample(media, transcript, predicted)
            sample = self.samples[key]
            if sample.prediction.raw_output != predicted.raw_output:
                raise ValueError("Multiple predictions for the same validation input; reuse the generation cache")
            if reference_id in sample.references and sample.references[reference_id] != target:
                raise ValueError(f"Reference contents changed within the epoch: {reference_id}")
            sample.references[reference_id] = target
            self.reference_keys[reference_id] = key


class OfficialBleurtBackend:
    """Official BLEURT scorer isolated from the Qwen process and its CUDA context.

    bleurt_python may point to a separate environment with TensorFlow/BLEURT.
    A probe runs during setup, before expensive training. Scores are never faked
    or silently omitted if this backend cannot load.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        checkpoint = Path(config.bleurt_checkpoint).expanduser().resolve()
        if not (checkpoint / "bleurt_config.json").is_file() or not (checkpoint / "saved_model.pb").is_file():
            raise FileNotFoundError(f"BLEURT checkpoint missing/incomplete: {checkpoint}. "
                                    "Download BLEURT-20 and pass --bleurt-checkpoint; see METRICS_USAGE.md")
        self.checkpoint = str(checkpoint)
        self.score(candidates=["Setup verification."], references=["Setup verification."])

    def score(self, *, candidates: list[str], references: list[str], batch_size: int | None = None) -> list[float]:
        payload = {"checkpoint": self.checkpoint, "candidates": candidates,
                   "references": references, "batch_size": batch_size or self.config.batch_size}
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        try:
            result = subprocess.run([self.config.bleurt_python, str(Path(__file__).resolve()), "--bleurt-worker"],
                                    input=json.dumps(payload), text=True, capture_output=True, env=env,
                                    timeout=self.config.bleurt_timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"BLEURT worker could not complete: {exc}") from exc
        if result.returncode:
            raise RuntimeError("Official BLEURT worker failed. Check the BLEURT Python environment and checkpoint.\n" + result.stderr[-12000:])
        try:
            values = json.loads(result.stdout)["scores"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Invalid response from BLEURT worker: " + result.stdout[-1000:]) from exc
        return _validated_scores(values, len(candidates), "BLEURT")


def _validated_scores(values, expected: int, name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} returned wrong-length or nonfinite scores")
    return result


class GenerationMetrics:
    """Calculate field-level metrics and required-field rates after validation.

    BERTScore/ROUGE/METEOR/BLEURT: report max over references and mean over
    references, then average equally over unique inputs. BLEU: native multi-
    reference corpus SacreBLEU, on its conventional 0-100 scale. Missing fields
    become empty candidates; BERTScore raw F1 is set to zero for empty candidates.
    BLEURT evaluates empties normally and its scores are not clipped to [0,1].
    Optional backend injection is for local tests; production initializes real
    scorers. Evidence is never scored.
    """

    def __init__(self, config: MetricsConfig | None = None, *, bert_scorer=None, bleurt_scorer=None):
        self.config = config or MetricsConfig()
        if self.config.batch_size <= 0 or not math.isfinite(self.config.bleurt_timeout) or self.config.bleurt_timeout <= 0:
            raise ValueError("Metric batch size and BLEURT timeout must be positive")
        try:
            from rouge_score.rouge_scorer import RougeScorer
            from sacrebleu.metrics import BLEU
            from nltk.tokenize import TreebankWordTokenizer
            from nltk.translate.meteor_score import single_meteor_score
            from nltk.corpus import wordnet
        except ImportError as exc:
            raise ImportError("Install bert-score rouge-score nltk sacrebleu; see METRICS_USAGE.md") from exc
        try:
            wordnet.ensure_loaded()
        except LookupError as exc:
            raise RuntimeError("METEOR requires WordNet: run python -m nltk.downloader wordnet omw-1.4") from exc
        self.rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.tokenizer = TreebankWordTokenizer()
        self.meteor = single_meteor_score
        self.bleu_class = BLEU
        self.sentence_bleu = BLEU(tokenize="13a", smooth_method="exp", effective_order=True)
        self.reference_metrics: dict[str, dict[str, float]] = {}
        if self.config.enable_bleurt:
            # Probe BLEURT before loading BERTScore, so setup errors surface early.
            self.bleurt = bleurt_scorer if bleurt_scorer is not None else OfficialBleurtBackend(self.config)
        else:
            self.bleurt = None
            print("BLEURT scoring temporarily disabled (enable_bleurt=False).", flush=True)
        if bert_scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError as exc:
                raise ImportError("BERTScore requires pip install bert-score") from exc
            bert_scorer = BERTScorer(model_type=self.config.bertscore_model,
                                    num_layers=self.config.bertscore_num_layers, lang="en",
                                    device=self.config.device, batch_size=self.config.batch_size,
                                    idf=False, rescale_with_baseline=False)
        self.bert = bert_scorer
        self.metadata = {"bertscore_model": self.config.bertscore_model,
                         "bertscore_num_layers": getattr(self.bert, "num_layers", self.config.bertscore_num_layers),
                         "bertscore_hash": getattr(self.bert, "hash", "injected"),
                         "bleurt_checkpoint": str(self.config.bleurt_checkpoint),
                         "bleurt_enabled": self.config.enable_bleurt,
                         "bleu_settings": "tokenizer=13a;smooth=exp;effective_order=true;scale=0-100",
                         "reference_aggregation": "pairwise_best_and_mean;bleu_native_multireference"}

    def _pair_scores(self, candidates: list[str], references: list[str]) -> dict[str, list[float | None]]:
        count = len(candidates)
        values: dict[str, list[float | None]] = {metric: [0.0] * count for metric in PAIR_METRICS}
        active = [i for i, text in enumerate(candidates) if text.strip()]
        if active:
            _, _, f1 = self.bert.score([candidates[i] for i in active], [references[i] for i in active],
                                      batch_size=self.config.batch_size)
            scores = _validated_scores(f1.detach().cpu().tolist(), len(active), "BERTScore")
            for index, score in zip(active, scores):
                values["bertscore_f1"][index] = score
        pair_iter = tqdm(
            enumerate(zip(candidates, references)),
            total=count,
            desc="Val metrics (ROUGE/METEOR)",
            leave=False,
            dynamic_ncols=True,
        )
        for i, (candidate, reference) in pair_iter:
            rouge = self.rouge.score(reference, candidate)
            for name in ("rouge1", "rouge2", "rougeL"):
                values[name + "_f1"][i] = float(rouge[name].fmeasure)
            values["meteor"][i] = float(self.meteor(self.tokenizer.tokenize(reference), self.tokenizer.tokenize(candidate)))
        if self.bleurt is None:
            values["bleurt"] = [None] * count
        else:
            # One worker call scores both fields, minimizing checkpoint reloads.
            values["bleurt"] = _validated_scores(
                self.bleurt.score(candidates=candidates, references=references,
                                  batch_size=self.config.batch_size), count, "BLEURT")
        return values

    def compute(self, accumulator: ValidationAccumulator) -> dict[str, float | int]:
        samples = list(accumulator.samples.values())
        if not samples:
            raise ValueError("Cannot compute metrics for an empty validation set")
        candidates, references, ranges = [], [], {}
        reference_order: dict[tuple[str, int], list[str]] = {}
        for field in EXPECTED_FIELDS:
            for i, sample in enumerate(samples):
                start = len(candidates)
                ids = []
                for reference_id, reference in sample.references.items():
                    candidates.append(sample.prediction.fields[field] or "")
                    references.append(reference[field])
                    ids.append(reference_id)
                if len(candidates) == start:
                    raise ValueError("Validation sample has no references")
                ranges[(field, i)] = (start, len(candidates))
                reference_order[(field, i)] = ids
        values = self._pair_scores(candidates, references)
        per_reference: dict[str, dict[str, float]] = {
            reference_id: {} for sample in samples for reference_id in sample.references}
        for field, prefix in zip(EXPECTED_FIELDS, ("intent", "reasoning")):
            for i, sample in enumerate(samples):
                start, _ = ranges[(field, i)]
                for offset, reference_id in enumerate(reference_order[(field, i)]):
                    index = start + offset
                    hypothesis = sample.prediction.fields[field] or ""
                    reference_text = sample.references[reference_id][field]
                    scored = per_reference[reference_id]
                    for metric in PAIR_METRICS:
                        scored[f"{prefix}_{metric}"] = values[metric][index]
                    scored[f"{prefix}_bleu"] = float(self.sentence_bleu.sentence_score(hypothesis, [reference_text]).score)
                    scored[f"{prefix}_exact_match"] = float(hypothesis == reference_text)
        self.reference_metrics = per_reference
        result: dict[str, float | int | None] = {
            "generation_unique_inputs": len(samples),
            "generation_unique_clips": len({sample.media_id for sample in samples}),
            "generation_annotation_references": sum(len(sample.references) for sample in samples),
            "required_field_completion_rate": statistics.mean(not s.prediction.missing_fields for s in samples),
            "missing_field_rate": sum(len(s.prediction.missing_fields) for s in samples) / (len(samples) * len(EXPECTED_FIELDS)),
            "generation_valid_json_rate": statistics.mean(s.prediction.valid_json for s in samples),
        }
        for field, prefix in zip(EXPECTED_FIELDS, ("intent", "reasoning")):
            result[prefix + "_missing_field_rate"] = statistics.mean(field in s.prediction.missing_fields for s in samples)
            for metric in PAIR_METRICS:
                if metric == "bleurt" and self.bleurt is None:
                    result[f"{prefix}_{metric}_best"] = None
                    result[f"{prefix}_{metric}_mean"] = None
                    continue
                best, mean = [], []
                for i in range(len(samples)):
                    start, stop = ranges[(field, i)]
                    scores = values[metric][start:stop]
                    best.append(max(scores))
                    mean.append(statistics.mean(scores))
                result[f"{prefix}_{metric}_best"] = statistics.mean(best)
                result[f"{prefix}_{metric}_mean"] = statistics.mean(mean)
            hypotheses = [sample.prediction.fields[field] or "" for sample in samples]
            grouped_refs = [[ref[field] for ref in sample.references.values()] for sample in samples]
            max_refs = max(map(len, grouped_refs))
            # SacreBLEU supports None entries for samples with fewer references.
            reference_documents = [[refs[j] if j < len(refs) else None for refs in grouped_refs] for j in range(max_refs)]
            bleu = self.bleu_class(tokenize="13a", smooth_method="exp", effective_order=True)
            result[prefix + "_bleu"] = float(bleu.corpus_score(hypotheses, reference_documents).score)
        return result


def print_validation_metrics(values: Mapping[str, Any], epoch: int) -> None:
    """Display every requested metric; BLEU uses 0-100, rates display percentages."""
    print(f"\nValidation generation metrics — epoch {epoch} ({values['generation_unique_inputs']} unique inputs)", flush=True)
    print(f"{'Metric':<23} {'Intent best':>12} {'Intent mean':>12} {'Reason best':>12} {'Reason mean':>12}", flush=True)

    def _fmt(score) -> str:
        return f"{score:>13.4f}" if score is not None else f"{'n/a':>13}"

    for metric in PAIR_METRICS:
        scores = [values[f"{field}_{metric}_{aggregation}"] for field in ("intent", "reasoning") for aggregation in ("best", "mean")]
        print(f"{metric:<23}" + "".join(_fmt(score) for score in scores), flush=True)
    print(f"BLEU (corpus, 0-100)     intent={values['intent_bleu']:.4f}  reasoning={values['reasoning_bleu']:.4f}", flush=True)
    for name in ("required_field_completion_rate", "missing_field_rate", "intent_missing_field_rate", "reasoning_missing_field_rate"):
        print(f"{name}: {100 * values[name]:.2f}%", flush=True)


def append_epoch_metrics(path: str | Path, *, epoch: int, run_id: str,
                         train_metrics: Mapping[str, Any],
                         val_metrics: Mapping[str, Any],
                         train_eval_metrics: Mapping[str, Any] | None = None,
                         metadata: Mapping[str, Any] | None = None) -> None:
    """Append one row per epoch; reject schema changes instead of corrupting CSV.

    run_id distinguishes experiments sharing the user's runs/metrics.csv path.
    POSIX file locking serializes concurrent appenders on the training server.
    train_eval is optional and omitted when not provided (train_eval pass skipped).
    """
    import fcntl
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    phases = (("train", train_metrics), ("val", val_metrics))
    if train_eval_metrics:
        phases = (("train", train_metrics), ("train_eval", train_eval_metrics), ("val", val_metrics))
    for phase, metrics in phases:
        row.update({f"{phase}_{key}": value for key, value in sorted(metrics.items())})
    row.update({key: value for key, value in sorted((metadata or {}).items())})
    fieldnames = list(row)
    with path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = next(csv.reader(handle), None)
            if existing is not None and existing != fieldnames:
                raise ValueError(f"CSV schema differs at {path}; choose a new --metrics-csv path or archive the old file")
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if existing is None:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bleurt_worker() -> None:
    """Subprocess entry: stdout is JSON only; package logs go to stderr."""
    import contextlib
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from bleurt import score
        scorer = score.BleurtScorer(request["checkpoint"])
        scores = scorer.score(references=request["references"], candidates=request["candidates"],
                              batch_size=request["batch_size"])
    json.dump({"scores": [float(value) for value in scores]}, sys.stdout)


if __name__ == "__main__":
    if sys.argv[1:] == ["--bleurt-worker"]:
        _bleurt_worker()
    else:
        raise SystemExit("Use GenerationMetrics from train.py; see METRICS_USAGE.md")
