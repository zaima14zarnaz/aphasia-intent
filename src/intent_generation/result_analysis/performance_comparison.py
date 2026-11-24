#!/usr/bin/env python3
"""Compare training strategies using SAVED metric values only.
Trained runs: read run_dir/metrics.jsonl, use only each record's val object,
select val.<intent/reasoning metric>_best for the six pairwise metrics per field,
and use val.intent_bleu / val.reasoning_bleu as logged. The *_best values already
are per-sample best-reference scores averaged over unique validation inputs.
For the main intent/reasoning figure, choose one epoch per supervised run
by the LOWEST saved val.response_loss,
matching train.py's best-checkpoint rule; ties choose the earlier epoch.
--selection-metric loss selects lowest total validation loss instead. Explicit
metric choices (including sum/scaled_mean) still select the highest score.
Zero-shot baselines: read first_val_responses.json (or its legacy filename)
separately from single-run-dir and hierarchical-run-dir. Use each file's stored
per-reference scores: group by video/gesture, take each metric's maximum over
references, then average equally over samples. Neither baseline substitutes for
the other. Plotted order: single zero shot, hierarchical zero shot, single
supervised, then optional Hierarchical prompt (RGB, supervised) and
Hierarchical prompt (pose, supervised) when those run dirs are provided.
Gesture-description / gesture-meaning figures compare the RGB zero-shot baseline
and available RGB/pose supervised runs (pairwise metrics panel + BLEU panel each).
The description figure uses one shared epoch for all supervised runs: the
minimum of each run's last trained epoch. The meaning figure selects each
supervised run by lowest overall val.response_loss. The main intent/reasoning
figure is unchanged and still selects by overall val.response_loss.
No metric computation or neural scorer/model loading occurs. No epoch response
JSONL or metrics.py is read. Missing saved metrics cause explicit errors; the
script never substitutes losses or *_mean values for plotted metrics, and never
recomputes scores. Loss is used only for epoch selection.
BLEU aggregation differs in these stored sources: baseline scores are mean
best-reference SENTENCE BLEU, while the training log stores CORPUS BLEU. The
plot and terminal disclose this; no best-reference corpus
BLEU is invented from the saved corpus values. Pairwise metrics are directly
aligned to the requested best-reference aggregation.
Example from intent_generation/:
  python result_analysis/performance_comparison.py \\
      --single-run-dir runs/single \\
      --hierarchical-run-dir runs/hierarchical \\
      --hierarchical-rgb-run-dir runs/hierarchical_v5 \\
      --hierarchical-pose-run-dir runs/hierarchical_pose
Relative run paths are resolved under this project root when needed, so both
runs/hierarchical and ../runs/hierarchical work from intent_generation/.
Exports PNG figures only. No CSV, JSON, PDF or SVG outputs are generated.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / 'runs'
SINGLE_RUN_DIR = RUNS_ROOT / 'single'
HIERARCHICAL_RUN_DIR = RUNS_ROOT / 'hierarchical'
OUTPUT = Path(__file__).resolve().parent / 'performance_comparison.png'
def resolve_from_project(path: Path) -> Path:
    """Resolve relative paths against cwd, then intent_generation/, then runs/."""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    parts = path.parts
    if len(parts) >= 2 and parts[0] == '..' and parts[1] == 'runs':
        candidates.append((PROJECT_ROOT / Path(*parts[1:])).resolve())
    if parts and parts[0] != 'runs':
        candidates.append((RUNS_ROOT / path.name).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
def first_val_file(run_dir: Path) -> Path:
    run_dir = resolve_from_project(run_dir)
    candidates = (run_dir / 'first_val_responses.json', run_dir / 'first_val_set_responses.json')
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Neither baseline file exists under {run_dir}: '
                            f'{candidates[0].name} or {candidates[1].name}')
def metrics_log_file(run_dir: Path, override: Path | None = None) -> Path:
    if override is not None:
        path = resolve_from_project(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    run_dir = resolve_from_project(run_dir)
    path = run_dir / 'metrics.jsonl'
    if path.is_file():
        return path
    raise FileNotFoundError(
        f'Missing trained metrics log: {path} '
        f'(run may still be in progress, or use a finished run dir under {RUNS_ROOT})'
    )
LABELS = {
    'intent_bertscore_f1': 'Intent BERTScore',
    'intent_rouge1_f1': 'Intent ROUGE-1',
    'intent_rouge2_f1': 'Intent ROUGE-2',
    'intent_rougeL_f1': 'Intent ROUGE-L',
    'intent_meteor': 'Intent METEOR',
    'intent_bleurt': 'Intent BLEURT',
    'intent_bleu': 'Intent BLEU',
    'reasoning_bertscore_f1': 'Reasoning BERTScore',
    'reasoning_rouge1_f1': 'Reasoning ROUGE-1',
    'reasoning_rouge2_f1': 'Reasoning ROUGE-2',
    'reasoning_rougeL_f1': 'Reasoning ROUGE-L',
    'reasoning_meteor': 'Reasoning METEOR',
    'reasoning_bleurt': 'Reasoning BLEURT',
    'reasoning_bleu': 'Reasoning BLEU',
}
GESTURE_LABELS = {
    'gesture_description_bertscore_f1': 'Gesture description BERTScore',
    'gesture_description_rouge1_f1': 'Gesture description ROUGE-1',
    'gesture_description_rouge2_f1': 'Gesture description ROUGE-2',
    'gesture_description_rougeL_f1': 'Gesture description ROUGE-L',
    'gesture_description_meteor': 'Gesture description METEOR',
    'gesture_description_bleurt': 'Gesture description BLEURT',
    'gesture_description_bleu': 'Gesture description BLEU',
    'gesture_meaning_bertscore_f1': 'Gesture meaning BERTScore',
    'gesture_meaning_rouge1_f1': 'Gesture meaning ROUGE-1',
    'gesture_meaning_rouge2_f1': 'Gesture meaning ROUGE-2',
    'gesture_meaning_rougeL_f1': 'Gesture meaning ROUGE-L',
    'gesture_meaning_meteor': 'Gesture meaning METEOR',
    'gesture_meaning_bleurt': 'Gesture meaning BLEURT',
    'gesture_meaning_bleu': 'Gesture meaning BLEU',
}
ALL_LABELS = {**LABELS, **GESTURE_LABELS}
BLEURT_KEYS = (
    'intent_bleurt', 'reasoning_bleurt',
    'gesture_description_bleurt', 'gesture_meaning_bleurt',
)
SAVED_COLUMNS = {
    key: key if key.endswith('_bleu') else key + '_best' for key in ALL_LABELS
}
CORE_REQUIRED_KEYS = tuple(key for key in LABELS if key not in BLEURT_KEYS)
GESTURE_REQUIRED_KEYS = tuple(key for key in GESTURE_LABELS if key not in BLEURT_KEYS)
LOSS_SELECTION_FIELDS = ('response_loss', 'loss')
DEFAULT_SELECTION_METRIC = 'response_loss'
CORE_SERIES_LABELS = (
    'Single prompt (zero shot)',
    'Hierarchical prompt (zero shot)',
    'Single prompt (supervised)',
)
RGB_SERIES_LABEL = 'Hierarchical prompt (RGB, supervised)'
POSE_SERIES_LABEL = 'Hierarchical prompt (pose, supervised)'
GESTURE_ZERO_SHOT_LABEL = 'Hierarchical (RGB, zero shot)'
COLORS = {
    CORE_SERIES_LABELS[0]: '#B8B6B3',
    CORE_SERIES_LABELS[1]: '#C3B2D0',
    CORE_SERIES_LABELS[2]: '#A8BECC',
    RGB_SERIES_LABEL: '#D8B5BE',
    POSE_SERIES_LABEL: '#B7C9A4',
    GESTURE_ZERO_SHOT_LABEL: '#C3B2D0',
}
BLEU_NOTE = ('BLEU uses different saved aggregations: zero-shot runs = mean best-reference sentence BLEU; '
             'trained runs = logged corpus BLEU. These BLEU values are not directly comparable.')
@dataclass
class EpochSummary:
    path: Path
    epoch: int
    metrics: dict[str, float | None]
    sample_count: int | None
    reference_count: int | None
    selection_score: float | None
    inference_time_avg_sec: float | None = None
    validation_losses: dict[str, float | None] = field(default_factory=dict)
def finite_score(value: Any, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f'{context}: missing/non-numeric saved score; no recomputation is permitted')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{context}: invalid saved score; no recomputation is permitted') from exc
    if not math.isfinite(number):
        raise ValueError(f'{context}: nonfinite saved score')
    return number
def optional_finite_score(value: Any, context: str) -> float | None:
    """Allow missing BLEURT (None / absent); still reject malformed non-null values."""
    if value is None:
        return None
    return finite_score(value, context)
def plottable_metric_keys(*metric_maps: Mapping[str, float | None],
    labels: Mapping[str, str] = LABELS) -> list[str]:
    """Drop BLEURT bars when any compared series lacks a finite BLEURT score."""
    keys = []
    for key in labels:
        if key in BLEURT_KEYS:
            if any(metrics.get(key) is None for metrics in metric_maps):
                continue
        keys.append(key)
    return keys
def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open(encoding='utf-8-sig') as handle:
            for line_number, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}, line {line_number}: {exc.msg}') from exc
                if not isinstance(item, dict):
                    raise ValueError(f'{path}, line {line_number}: expected a JSON object')
                rows.append(item)
    else:
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'{path}: expected a JSON array of response objects')
    if not rows:
        raise ValueError(f'No metric records in {path}')
    return rows
def score_epoch(metrics: Mapping[str, float | None], selection_metric: str = 'sum') -> float:
    usable = {key: value for key, value in metrics.items() if value is not None}
    if not usable:
        raise ValueError('No finite saved metrics available for epoch selection')
    if selection_metric == 'sum':
        return math.fsum(usable.values())
    if selection_metric == 'scaled_mean':
        return math.fsum(value / 100. if key.endswith('_bleu') else value for key, value in usable.items()) / len(usable)
    if selection_metric not in usable:
        raise ValueError(f'Selection metric {selection_metric!r} is unavailable (missing/null saved score)')
    return usable[selection_metric]
def optional_count(values: Mapping, key: str, path: Path, epoch: int) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f'{path}, epoch {epoch}: invalid {key} count')
    return int(value)
def _extract_metrics(val: Mapping, path: Path, epoch: int, *, require_gesture: bool) -> dict[str, float | None]:
    required = list(CORE_REQUIRED_KEYS)
    if require_gesture:
        required.extend(GESTURE_REQUIRED_KEYS)
    missing = [SAVED_COLUMNS[key] for key in required if SAVED_COLUMNS[key] not in val]
    if missing:
        raise ValueError(f'{path}, epoch {epoch}: missing saved val metrics {missing}. '
                         'Use the log containing *_best fields; scores will not be recomputed.')
    metrics: dict[str, float | None] = {
        key: finite_score(val[SAVED_COLUMNS[key]], f'{path}, epoch {epoch}, val.{SAVED_COLUMNS[key]}')
        for key in required
    }
    optional_keys = [key for key in BLEURT_KEYS if key in LABELS or require_gesture]
    if not require_gesture:
        # Still capture gesture scores when present for downstream plots.
        for key in GESTURE_REQUIRED_KEYS:
            source = SAVED_COLUMNS[key]
            if source in val:
                metrics[key] = optional_finite_score(val.get(source), f'{path}, epoch {epoch}, val.{source}')
        optional_keys = list(BLEURT_KEYS)
    for key in optional_keys:
        source = SAVED_COLUMNS[key]
        metrics[key] = optional_finite_score(val.get(source), f'{path}, epoch {epoch}, val.{source}')
    return metrics
def best_metrics_from_log(path: Path, *, selection_metric: str = DEFAULT_SELECTION_METRIC,
    allow_cohort_differences: bool = False, require_gesture: bool = False,
    gesture_loss_keys: Sequence[str] = (),
    ) -> tuple[EpochSummary, list[EpochSummary]]:
    """Choose lowest saved validation loss by default; plot that epoch's metrics."""
    summaries, seen = [], set()
    for index, row in enumerate(read_rows(path), 1):
        epoch = row.get('epoch')
        if type(epoch) is not int or epoch < 0:
            raise ValueError(f'{path}, record {index}: epoch must be a nonnegative integer')
        if epoch in seen:
            raise ValueError(f'{path}: duplicate epoch {epoch}; use one metric record per epoch')
        seen.add(epoch)
        if epoch == 0:
            continue  # Baseline comes from the explicitly requested first_val file.
        val = row.get('val')
        if not isinstance(val, Mapping):
            raise ValueError(f'{path}, epoch {epoch}: missing val metrics object')
        metrics = _extract_metrics(val, path, epoch, require_gesture=require_gesture)
        count = optional_count(val, 'generation_unique_inputs', path, epoch)
        references = optional_count(val, 'generation_annotation_references', path, epoch)
        if selection_metric in LOSS_SELECTION_FIELDS:
            selection_score = finite_score(val.get(selection_metric),
                                           f'{path}, epoch {epoch}, val.{selection_metric}')
        else:
            selection_score = score_epoch(
                {key: metrics[key] for key in LABELS if key in metrics}, selection_metric)
        inference_avg = optional_finite_score(
            val.get('inference_time_avg_sec'),
            f'{path}, epoch {epoch}, val.inference_time_avg_sec')
        summaries.append(EpochSummary(
            path, epoch, metrics, count, references, selection_score,
            inference_time_avg_sec=inference_avg,
            validation_losses={
                key: optional_finite_score(val.get(key), f'{path}, epoch {epoch}, val.{key}')
                for key in gesture_loss_keys
            }))
    if not summaries:
        raise ValueError(f'{path}: no trained validation epochs')
    summaries.sort(key=lambda summary: summary.epoch)
    known_counts = {item.sample_count for item in summaries if item.sample_count is not None}
    if len(known_counts) > 1 and not allow_cohort_differences:
        raise ValueError(f'{path}: validation input counts change across epochs. '
                         'Use matched cohorts, or --allow-cohort-differences for intentional unmatched comparisons.')
    print(f'\nSaved validation metrics: {path}', flush=True)
    for item in summaries:
        inference = (f'{item.inference_time_avg_sec:.4f}s'
        if item.inference_time_avg_sec is not None else 'n/a')
        print(f'  epoch={item.epoch:04d} samples={item.sample_count if item.sample_count is not None else "unknown"} '
              f'references={item.reference_count if item.reference_count is not None else "unknown"} '
              f'{selection_metric}={item.selection_score:.6f} avg_inference={inference}', flush=True)
    if selection_metric in LOSS_SELECTION_FIELDS:
        best = min(summaries, key=lambda item: (item.selection_score, item.epoch))
    else:
        best = max(summaries, key=lambda item: (item.selection_score, -item.epoch))
    return best, summaries
def best_gesture_epoch(epochs: Sequence[EpochSummary], loss_key: str) -> EpochSummary:
    """Select one whole epoch by its task loss, never by generation metrics."""
    eligible = [item for item in epochs if item.validation_losses.get(loss_key) is not None]
    if not eligible:
        source = epochs[0].path if epochs else '<empty log>'
        raise ValueError(
            f'{source}: no trained epoch has a finite, non-null val.{loss_key}. '
            'Use --description-loss-key / --meaning-loss-key to match the saved log; '
            'there is no fallback to overall loss or generation scores.')
    return min(eligible, key=lambda item: (item.validation_losses[loss_key], item.epoch))


def last_gesture_epoch(epochs: Sequence[EpochSummary]) -> EpochSummary:
    """Return the final trained epoch for a run."""
    if not epochs:
        raise ValueError('No trained epochs available for gesture description selection')
    return max(epochs, key=lambda item: item.epoch)


def epoch_by_number(epochs: Sequence[EpochSummary], epoch: int) -> EpochSummary:
    """Return the summary for an exact epoch index."""
    for item in epochs:
        if item.epoch == epoch:
            return item
    source = epochs[0].path if epochs else '<empty log>'
    available = sorted(item.epoch for item in epochs)
    raise ValueError(
        f'{source}: requested description epoch {epoch} is missing '
        f'(available epochs: {available})'
    )

def _video_name(value: str) -> str:
    name = value.replace('\\', '/').rsplit('/', 1)[-1].strip()
    return name[:-4] if name.lower().endswith('.mp4') else name
def sample_id(row: Mapping[str, Any]) -> str:
    candidates = []
    media = row.get('media_id')
    if isinstance(media, str):
        match = re.fullmatch(r'(.+)/gest_([0-9]+)', media.replace('\\', '/'))
        if match:
            candidates.append((_video_name(match[1]), int(match[2])))
    clip = row.get('clip_id')
    if isinstance(clip, str):
        match = re.fullmatch(r'(.+)_([0-9]+)_[^/\\]+?\.mp4', clip.replace('\\', '/').rsplit('/', 1)[-1], re.IGNORECASE)
        if match:
            candidates.append((_video_name(match[1]), int(match[2])))
    filename = row.get('filename') or row.get('fiename') or row.get('video_id')
    index = row.get('gest_idx', row.get('gesture_index'))
    if isinstance(filename, str) and index is not None:
        try:
            if isinstance(index, bool) or int(index) != float(index) or int(index) < 0:
                raise ValueError
            candidates.append((_video_name(filename), int(index)))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Invalid baseline gesture index {index!r}') from exc
    if not candidates or not candidates[0][0]:
        raise ValueError('Baseline sample needs clip_id/media_id, or filename/fiename + gest_idx')
    if len(set(candidates)) != 1:
        raise ValueError(f'Conflicting baseline sample identifiers: {candidates}')
    return f'{candidates[0][0]}.mp4/gest_{candidates[0][1]}'
def baseline_metrics(path: Path, *, selection_metric: str = DEFAULT_SELECTION_METRIC,
    include_gesture: bool = False) -> EpochSummary:
    grouped, predictions = {}, {}
    score_keys = dict(LABELS)
    if include_gesture:
        score_keys.update(GESTURE_LABELS)
    rows = read_rows(path)
    for index, row in enumerate(rows, 1):
        if row.get('split') not in (None, 'val', 'validation', 'untrained_val'):
            raise ValueError(f'{path}, record {index}: expected validation baseline, got split={row["split"]!r}')
        identity = sample_id(row)
        prediction = row.get('pred_response', row.get('output'))
        if isinstance(prediction, Mapping):
            signature = (prediction.get('communicative intent'), prediction.get('reasoning'))
            if identity in predictions and predictions[identity] != signature:
                raise ValueError(f'{path}: baseline predictions differ across annotation references for {identity}')
            predictions[identity] = signature
        metrics = row.get('metrics')
        if not isinstance(metrics, Mapping):
            raise ValueError(f'{path}, record {index}: no saved baseline metrics; no recomputation is permitted')
        sample = grouped.setdefault(identity, {})
        for key in score_keys:
            if key in BLEURT_KEYS:
                value = optional_finite_score(metrics.get(key), f'{path}, record {index}, {key}')
                if value is None:
                    sample[key] = None
                    continue
                previous = sample.get(key)
                sample[key] = value if previous is None else max(previous, value)
                continue
            value = finite_score(metrics.get(key), f'{path}, record {index}, {key}')
            sample[key] = max(sample.get(key, -math.inf), value)
    means: dict[str, float | None] = {}
    for key in score_keys:
        if key in BLEURT_KEYS and any(values.get(key) is None for values in grouped.values()):
            means[key] = None
            continue
        means[key] = math.fsum(values[key] for values in grouped.values()) / len(grouped)
    print(f'Zero-shot baseline: {path} (saved scores only'
          f'{"; includes gesture fields" if include_gesture else ""})', flush=True)
    # Zero-shot results have no trained epoch or aggregate validation loss to select.
    core_means = {key: means[key] for key in LABELS}
    selection_score = None if selection_metric in LOSS_SELECTION_FIELDS else score_epoch(core_means, selection_metric)
    return EpochSummary(path, 0, means, len(grouped), len(rows), selection_score)
STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'text.color': '#30343B',
    'axes.labelcolor': '#484D55',
    'xtick.color': '#484D55',
    'ytick.color': '#6A7078',
    'axes.linewidth': 0.7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}
def _bar_layout(n_series: int) -> tuple[float, float]:
    if n_series <= 0:
        raise ValueError('Need at least one series to plot')
    if n_series <= 3:
        return 0.18, 0.21
    if n_series == 4:
        return 0.15, 0.175
    return 0.12, 0.14
def grouped_bars(ax, labels: list[str], series: Sequence[tuple[str, Sequence[float]]],
    ylabel: str, title: str, *, bleu: bool = False, show_values: bool = True,
    title_pad: float = 15) -> None:
    """Draw centered bars per group for the provided series order."""
    if not series:
        raise ValueError('Need at least one series to plot')
    positions = np.array([1.15, 3.85]) if bleu else np.arange(len(labels), dtype=float)
    if bleu and len(positions) != len(labels):
        raise ValueError('The BLEU panel requires one label per BLEU metric')
    width, step = _bar_layout(len(series))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * step
    for offset, (name, values) in zip(offsets, series):
        if len(values) != len(labels):
            raise ValueError(f'{name}: metric values and labels must have equal length')
        bars = ax.bar(positions + offset, values, width, label=name,
        color=COLORS[name], edgecolor='white', linewidth=0.5, zorder=3)
        if show_values:
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=5.5, color='#3C4249')
    span = max(positions) - min(positions) if len(positions) > 1 else 1.0
    pad = max(0.7, len(series) * step / 2 + width)
    ax.set_xlim(min(positions) - pad, max(positions) + pad)
    ax.set_xticks(positions, labels)
    ax.tick_params(axis='x', length=0, pad=8, labelsize=10)
    ax.tick_params(axis='y', length=0, pad=7, labelsize=9)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=10)
    ax.set_title(title, loc='left', pad=title_pad, fontsize=12, fontweight='semibold')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E8EB', linewidth=0.65)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#B8BDC4')
    _ = span  # retained for layout clarity
def _legend_column_order(names: Sequence[str]) -> list[str]:
    """Matplotlib fills multi-column legends by column; keep rows left-to-right."""
    if len(names) <= 3:
        return list(names)
    # Prefer two columns: fill row-wise visually by reordering for column-major fill.
    rows = 2
    cols = math.ceil(len(names) / rows)
    ordered = []
    for row in range(rows):
        for col in range(cols):
            index = col * rows + row
            if index < len(names):
                ordered.append(names[index])
    return ordered
def build_figure(series: Sequence[tuple[str, Mapping[str, float | None]]],
                 *, show_values: bool = True, selection_metric: str = DEFAULT_SELECTION_METRIC,
    plot_keys: list[str] | None = None):
    """Return the multi-series intent/reasoning figure without modifying metric values."""
    if len(series) < 3:
        raise ValueError('Need the three core series to build the main figure')
    metric_maps = [metrics for _, metrics in series]
    active = plot_keys if plot_keys is not None else plottable_metric_keys(*metric_maps)
    intent_keys = [key for key in active if key.startswith('intent_') and not key.endswith('_bleu')]
    reasoning_keys = [key for key in active if key.startswith('reasoning_') and not key.endswith('_bleu')]
    bleu_keys = [key for key in active if key.endswith('_bleu') and key in LABELS]
    if not intent_keys or not reasoning_keys or not bleu_keys:
        raise ValueError('Need at least one intent, reasoning, and BLEU metric to build the figure')
    names = [name for name, _ in series]
    with plt.rc_context(STYLE):
        width = 9.4 if len(series) <= 4 else 10.6
        figure, axes = plt.subplots(3, 1, figsize=(width, 9.6),
        gridspec_kw={'height_ratios': [1, 1, 0.62]})
        figure.subplots_adjust(left=0.095, right=0.985, bottom=0.075,
        top=0.825, hspace=0.46)
        figure.suptitle('Validation performance across prompting and training strategies',
        x=0.080, y=0.978, ha='left', fontsize=12.5, fontweight='semibold')
        selection_note = (f'Lowest validation {selection_metric.replace("_", " ")} per supervised run'
        if selection_metric in LOSS_SELECTION_FIELDS
        else f'Highest {selection_metric} per supervised run')
        figure.text(0.080, 0.944, 'Saved best-reference averages | ' + selection_note,
        fontsize=9, color='#737982')
        legend_names = _legend_column_order(names)
        handles = [Patch(facecolor=COLORS[name], edgecolor='none', label=name) for name in legend_names]
        ncol = 2 if len(names) <= 4 else 3
        figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.075, 0.916),
        ncol=ncol, frameon=False, fontsize=8, handlelength=1.6,
        handleheight=0.85, columnspacing=2.0, borderaxespad=0)
        for ax, keys, title, prefix, ylabel, bleu in (
            (axes[0], intent_keys, '(a) Communicative intent', 'Intent ', 'Mean score', False),
            (axes[1], reasoning_keys, '(b) Reasoning', 'Reasoning ', 'Mean score', False),
            (axes[2], bleu_keys, '(c) BLEU (saved aggregation)', '', 'BLEU (0-100)', True),
        ):
            panel_series = [
                (name, [float(metrics[key]) for key in keys]) for name, metrics in series
            ]
            grouped_bars(ax, [LABELS[key].removeprefix(prefix) for key in keys],
                         panel_series, ylabel, title, bleu=bleu, show_values=show_values)
        regular_keys = intent_keys + reasoning_keys
        values = [metrics[key] for _, metrics in series for key in regular_keys]
        low, high = min(0.0, *values), max(0.0, *values)
        span = high - low or 1.0
        bottom = low - 0.12 * span if low < 0 else 0.0
        top = max(1.0, high + 0.12 * span)
        for ax in axes[:2]:
            ax.set_ylim(bottom, top)
            ax.yaxis.set_major_locator(MultipleLocator(0.2) if top - bottom <= 1.6 else MaxNLocator(5))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if low < 0:
                ax.axhline(0, color='#A9AFB6', linewidth=0.8, zorder=2)
        bleu_values = [metrics[key] for _, metrics in series for key in bleu_keys]
        low, high = min(0.0, *bleu_values), max(0.0, *bleu_values)
        span = high - low or 1.0
        axes[2].set_ylim(low - 0.15 * span if low < 0 else 0.0, high + 0.23 * span)
        axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        figure.text(0.080, 0.020, 'Pairwise: best-reference averages. BLEU: zero-shot sentence mean; supervised corpus score.',
    fontsize=9, color='#737982')
    return figure
def build_gesture_figure(field: str, series: Sequence[tuple[str, Mapping[str, float | None]]],
                         *, show_values: bool = True, selection_metric: str | None = None):
    """Two panels with a separate title/legend/subtitle band and footer."""
    if field not in ('gesture_description', 'gesture_meaning'):
        raise ValueError(f'Unknown gesture field: {field}')
    if not series:
        raise ValueError(f'Need at least one RGB/pose series for {field}')
    title_field = 'Gesture description' if field == 'gesture_description' else 'Gesture meaning'
    pairwise_keys = [
        f'{field}_bertscore_f1', f'{field}_rouge1_f1', f'{field}_rouge2_f1',
        f'{field}_rougeL_f1', f'{field}_meteor',
    ]
    bleu_key = f'{field}_bleu'
    for name, metrics in series:
        missing = [key for key in pairwise_keys + [bleu_key] if metrics.get(key) is None]
        if missing:
            raise ValueError(f'{name}: missing gesture metrics {missing}')
    prefix = 'Gesture description ' if field == 'gesture_description' else 'Gesture meaning '
    selection_metric = selection_metric or f'{field}_ce'
    if selection_metric == 'shared_last_epoch':
        selection_note = 'Shared epoch = min last trained epoch across RGB/pose runs'
    elif selection_metric == 'last_epoch':
        selection_note = 'Last trained epoch per supervised run'
    else:
        selection_note = f'Lowest validation {selection_metric.replace("_", " ")} per run'
    # The row sizes are expressed in inches, keeping a variable-height legend
    # from overlapping either the figure title or the first plotting panel.
    legend_height = 0.22 * len(series) + 0.10
    header_height = 0.39 + legend_height + 0.28
    height = 6.75 + header_height
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(9.4, height))
        outer = figure.add_gridspec(3, 1, left=0.105, right=0.975,
                                    top=0.975, bottom=0.035,
                                    height_ratios=[header_height, 5.85, 0.24], hspace=0.13)
        header = outer[0].subgridspec(3, 1, height_ratios=[0.39, legend_height, 0.28], hspace=0.0)
        title_ax = figure.add_subplot(header[0])
        legend_ax = figure.add_subplot(header[1])
        note_ax = figure.add_subplot(header[2])
        for ax in (title_ax, legend_ax, note_ax):
            ax.set_axis_off()
        title_ax.text(0.0, 1.0, title_field, transform=title_ax.transAxes,
                      ha='left', va='top', fontsize=15, fontweight='semibold')
        handles = [Patch(facecolor=COLORS[name], edgecolor='none', label=name) for name, _ in series]
        legend_ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.0),
                         ncol=1, frameon=False, fontsize=9,
                         handlelength=1.6, handleheight=0.85,
                         labelspacing=0.55, borderaxespad=0.0)
        note_ax.text(0.0, 0.70, 'Saved best-reference averages | ' + selection_note,
                     transform=note_ax.transAxes, ha='left', va='top', fontsize=8.5, color='#737982')

        panels = outer[1].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.42)
        axes = [figure.add_subplot(panels[0]), figure.add_subplot(panels[1])]
        pairwise_series = [(name, [float(metrics[key]) for key in pairwise_keys]) for name, metrics in series]
        grouped_bars(axes[0], [GESTURE_LABELS[key].removeprefix(prefix) for key in pairwise_keys],
                     pairwise_series, 'Mean score', '', show_values=show_values, title_pad=0)
        bleu_series = [(name, [float(metrics[bleu_key])]) for name, metrics in series]
        grouped_bars(axes[1], ['BLEU'], bleu_series, 'BLEU (0-100)', 'BLEU',
                     bleu=False, show_values=show_values, title_pad=12)
        # Retain the existing score values and limits; only presentation changes.
        values = [metrics[key] for _, metrics in series for key in pairwise_keys]
        low, high = min(0.0, *values), max(0.0, *values)
        span = high - low or 1.0
        axes[0].set_ylim(0.0, max(1.0, high + 0.12 * span))
        axes[0].yaxis.set_major_locator(MultipleLocator(0.2) if high <= 1.6 else MaxNLocator(5))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        bleu_values = [metrics[bleu_key] for _, metrics in series]
        low, high = min(0.0, *bleu_values), max(0.0, *bleu_values)
        span = high - low or 1.0
        axes[1].set_ylim(0.0, high + 0.23 * span)
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        footer_ax = figure.add_subplot(outer[2])
        footer_ax.set_axis_off()
        footer_ax.text(0.0, 0.0, 'Pairwise: best-reference averages. BLEU: logged corpus score.',
                       transform=footer_ax.transAxes, ha='left', va='bottom', fontsize=8.5, color='#737982')
    return figure
def save_figure(figure, output: Path, dpi: int = 300) -> list[Path]:
    """Save a PNG only, including when another extension was supplied."""
    output = output.expanduser().with_suffix('.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        figure.savefig(output, format='png', dpi=dpi, facecolor='white')
    return [output]

def baseline_file(run_dir: Path, override: Path | None = None) -> Path:
    """Resolve this run's own zero-shot results; never borrow another run's."""
    if override is None:
        return first_val_file(run_dir)
    path = resolve_from_project(override)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--single-run-dir', type=Path, default=SINGLE_RUN_DIR)
    parser.add_argument('--hierarchical-run-dir', type=Path, default=HIERARCHICAL_RUN_DIR,
        help='Hierarchical zero-shot baseline source (first_val_responses.json)')
    parser.add_argument('--hierarchical-rgb-run-dir', type=Path, default=None,
        help='Optional RGB hierarchical supervised run; adds the fourth bar when set')
    parser.add_argument('--hierarchical-pose-run-dir', type=Path, default=None,
        help='Optional pose hierarchical supervised run; adds the fifth bar when set')
    parser.add_argument('--single-metrics-jsonl', type=Path, help='Default: single-run-dir/metrics.jsonl')
    parser.add_argument('--hierarchical-rgb-metrics-jsonl', type=Path,
        help='Default: hierarchical-rgb-run-dir/metrics.jsonl')
    parser.add_argument('--hierarchical-pose-metrics-jsonl', type=Path,
        help='Default: hierarchical-pose-run-dir/metrics.jsonl')
    parser.add_argument('--single-zero-shot-json', '--single-no-training-json', dest='single_zero_shot_json',
        type=Path, help='Default: single-run-dir/first_val_responses.json')
    parser.add_argument('--hierarchical-zero-shot-json', '--hierarchical-no-training-json', '--no-training-json',
        dest='hierarchical_zero_shot_json', type=Path,
        help='Default: hierarchical-run-dir/first_val_responses.json; --no-training-json is a legacy alias')
    parser.add_argument('--selection-metric',
        choices=(*LOSS_SELECTION_FIELDS, 'sum', 'scaled_mean', *LABELS, *GESTURE_LABELS),
        default=DEFAULT_SELECTION_METRIC,
        help='Default: lowest val.response_loss (train.py checkpoint rule); loss selects lowest val.loss; explicit score metrics select highest')
    parser.add_argument('--description-loss-key', default='gesture_description_ce',
        help='Saved val loss key minimized separately for the description graph (default: gesture_description_ce)')
    parser.add_argument('--meaning-loss-key', default='gesture_meaning_ce',
        help='Saved val loss key minimized separately for the meaning graph (default: gesture_meaning_ce)')
    parser.add_argument('--allow-cohort-differences', action='store_true',
        help='Permit intentional comparisons with different validation input counts')
    parser.add_argument('--no-recompute', action='store_true',
        help='Compatibility flag: recomputation is always disabled')
    parser.add_argument('--output', type=Path, default=OUTPUT,
        help='PNG output path; any supplied extension is replaced with .png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-values', action='store_true')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    if args.hierarchical_rgb_metrics_jsonl is not None and args.hierarchical_rgb_run_dir is None:
        parser.error('--hierarchical-rgb-metrics-jsonl requires --hierarchical-rgb-run-dir')
    if args.hierarchical_pose_metrics_jsonl is not None and args.hierarchical_pose_run_dir is None:
        parser.error('--hierarchical-pose-metrics-jsonl requires --hierarchical-pose-run-dir')
    single_dir = resolve_from_project(args.single_run_dir)
    hierarchical_dir = resolve_from_project(args.hierarchical_run_dir)
    single_baseline_path = baseline_file(single_dir, args.single_zero_shot_json)
    hierarchical_baseline_path = baseline_file(hierarchical_dir, args.hierarchical_zero_shot_json)
    single_path = metrics_log_file(single_dir, args.single_metrics_jsonl)
    print('Reading saved metric values only. Metric recomputation and scorer/model loading are disabled.', flush=True)
    print(f'Resolved single run: {single_dir}', flush=True)
    print(f'Resolved hierarchical zero-shot run: {hierarchical_dir}', flush=True)
    single_zero_shot = baseline_metrics(single_baseline_path, selection_metric=args.selection_metric)
    hierarchical_zero_shot = baseline_metrics(hierarchical_baseline_path, selection_metric=args.selection_metric)
    single, single_epochs = best_metrics_from_log(
        single_path, selection_metric=args.selection_metric,
        allow_cohort_differences=args.allow_cohort_differences)
    series_summaries: list[tuple[str, EpochSummary, list[EpochSummary]]] = [
        (CORE_SERIES_LABELS[0], single_zero_shot, [single_zero_shot]),
        (CORE_SERIES_LABELS[1], hierarchical_zero_shot, [hierarchical_zero_shot]),
        (CORE_SERIES_LABELS[2], single, single_epochs),
    ]
    rgb = pose = None
    rgb_epochs = pose_epochs = None
    if args.hierarchical_rgb_run_dir is not None:
        rgb_dir = resolve_from_project(args.hierarchical_rgb_run_dir)
        print(f'Resolved hierarchical RGB run: {rgb_dir}', flush=True)
        rgb_path = metrics_log_file(rgb_dir, args.hierarchical_rgb_metrics_jsonl)
        rgb, rgb_epochs = best_metrics_from_log(
            rgb_path, selection_metric=args.selection_metric,
            allow_cohort_differences=args.allow_cohort_differences, require_gesture=True,
            gesture_loss_keys=(args.description_loss_key, args.meaning_loss_key, 'response_loss'))
        series_summaries.append((RGB_SERIES_LABEL, rgb, rgb_epochs))
    if args.hierarchical_pose_run_dir is not None:
        pose_dir = resolve_from_project(args.hierarchical_pose_run_dir)
        print(f'Resolved hierarchical pose run: {pose_dir}', flush=True)
        pose_path = metrics_log_file(pose_dir, args.hierarchical_pose_metrics_jsonl)
        pose, pose_epochs = best_metrics_from_log(
            pose_path, selection_metric=args.selection_metric,
            allow_cohort_differences=args.allow_cohort_differences, require_gesture=True,
            gesture_loss_keys=(args.description_loss_key, args.meaning_loss_key, 'response_loss'))
        series_summaries.append((POSE_SERIES_LABEL, pose, pose_epochs))
    selected = [item for _, item, _ in series_summaries]
    counts = {item.sample_count for item in selected if item.sample_count is not None}
    if len(counts) > 1 and not args.allow_cohort_differences:
        raise ValueError('Zero-shot and supervised validation input counts differ. Use matched data, '
                         'or --allow-cohort-differences for an intentional unmatched comparison.')
    print('\nPairwise metrics: saved val.*_best fields; BLEU: saved val.intent_bleu / val.reasoning_bleu.')
    for name, item, _ in series_summaries:
        if item.epoch == 0:
            continue
        inference = (f'{item.inference_time_avg_sec:.4f}s'
        if item.inference_time_avg_sec is not None else 'n/a')
        print(f'{name}: epoch={item.epoch:04d}, {args.selection_metric}={item.selection_score:.6f}, '
              f'avg val inference={inference}')
    print(BLEU_NOTE)
    plot_series = [(name, item.metrics) for name, item, _ in series_summaries]
    active_keys = plottable_metric_keys(*(metrics for _, metrics in plot_series))
    skipped_bleurt = [key for key in ('intent_bleurt', 'reasoning_bleurt') if key not in active_keys]
    if skipped_bleurt:
        print('BLEURT unavailable for at least one run; plotting remaining metrics only '
              f'(skipped: {", ".join(skipped_bleurt)}).', flush=True)
    series_names = [name for name, _, _ in series_summaries]
    print(f'{"metric":<28}' + ''.join(f'{name:>40}' for name in series_names))
    for key in LABELS:
        cells = []
        for _, item, _ in series_summaries:
            value = item.metrics.get(key)
            cells.append(f'{value:40.4f}' if value is not None else f'{"n/a":>40}')
        print(f'{key:<28}' + ''.join(cells))
    figure = build_figure(plot_series, show_values=not args.no_values,
        selection_metric=args.selection_metric, plot_keys=active_keys)
    def _fmt_inference(seconds: float | None) -> str:
        return f'{seconds:.2f}s' if seconds is not None else 'n/a'
    supervised_bits = [f'single: {single.epoch}']
    inference_bits = [f'single: {_fmt_inference(single.inference_time_avg_sec)}']
    if rgb is not None:
        supervised_bits.append(f'rgb: {rgb.epoch}')
        inference_bits.append(f'rgb: {_fmt_inference(rgb.inference_time_avg_sec)}')
    if pose is not None:
        supervised_bits.append(f'pose: {pose.epoch}')
        inference_bits.append(f'pose: {_fmt_inference(pose.inference_time_avg_sec)}')
    figure.text(
        0.975, 0.850,
        'Supervised epochs — ' + ' | '.join(supervised_bits) + '\n'
        'Avg val inference — ' + ' | '.join(inference_bits),
        ha='right', va='top', fontsize=8, color='#737982', linespacing=1.35,
    )
    output = args.output.expanduser().with_suffix('.png')
    paths = save_figure(figure, output, dpi=args.dpi)
    plt.close(figure)
    gesture_runs = []
    rgb_zero_shot = None
    if rgb is not None:
        rgb_dir = resolve_from_project(args.hierarchical_rgb_run_dir)
        rgb_zero_shot = baseline_metrics(
            baseline_file(rgb_dir), selection_metric=args.selection_metric, include_gesture=True)
        gesture_runs.append((RGB_SERIES_LABEL, rgb_epochs))
    if pose is not None:
        gesture_runs.append((POSE_SERIES_LABEL, pose_epochs))
    if gesture_runs:
        if rgb_zero_shot is None:
            print('Hierarchical RGB zero-shot baseline unavailable; '
                  'gesture figures omit that leading bar.', flush=True)
        last_epochs = {
            name: last_gesture_epoch(epochs).epoch for name, epochs in gesture_runs
        }
        shared_epoch = min(last_epochs.values())
        for field, loss_key in (
            ('gesture_description', args.description_loss_key),
            ('gesture_meaning', args.meaning_loss_key),
        ):
            use_shared_last_epoch = field == 'gesture_description'
            shared_series = []
            if rgb_zero_shot is not None:
                shared_series.append((GESTURE_ZERO_SHOT_LABEL, rgb_zero_shot.metrics))
            if use_shared_last_epoch:
                print(
                    f'\n{field.replace("_", " ").capitalize()}: shared epoch '
                    f'{shared_epoch:04d} (= min last epoch among {last_epochs})',
                    flush=True,
                )
            else:
                print(f'\n{field.replace("_", " ").capitalize()}: lowest val.response_loss',
                      flush=True)
            task_selected = []
            for name, epochs in gesture_runs:
                if use_shared_last_epoch:
                    item = epoch_by_number(epochs, shared_epoch)
                    task_selected.append(item)
                    shared_series.append((name, item.metrics))
                    field_ce = (item.validation_losses or {}).get(loss_key)
                    ce_note = (f', val.{loss_key}={field_ce:.6f}' if field_ce is not None else '')
                    print(f'  {name}: epoch={item.epoch:04d} '
                          f'(run last={last_epochs[name]:04d}){ce_note}', flush=True)
                else:
                    item = best_gesture_epoch(epochs, 'response_loss')
                    task_selected.append(item)
                    shared_series.append((name, item.metrics))
                    print(f'  {name}: epoch={item.epoch:04d}, '
                          f'val.response_loss={item.validation_losses["response_loss"]:.6f}',
                          flush=True)
            task_counts = {item.sample_count for item in task_selected if item.sample_count is not None}
            if rgb_zero_shot is not None and rgb_zero_shot.sample_count is not None:
                task_counts.add(rgb_zero_shot.sample_count)
            if len(task_counts) > 1 and not args.allow_cohort_differences:
                raise ValueError(f'{field}: selected validation input counts differ. '
                                 'Use matched data or --allow-cohort-differences.')
            for key in GESTURE_LABELS:
                if not key.startswith(field + '_'):
                    continue
                cells = []
                for _, metrics in shared_series:
                    value = metrics.get(key)
                    cells.append(f'{value:40.4f}' if value is not None else f'{"n/a":>40}')
                print(f'{key:<36}' + ''.join(cells))
            gesture_figure = build_gesture_figure(
                field, shared_series, show_values=not args.no_values,
                selection_metric=('shared_last_epoch' if use_shared_last_epoch else 'response_loss'))
            gesture_output = output.with_name(f'{output.stem}_{field}.png')
            paths.extend(save_figure(gesture_figure, gesture_output, dpi=args.dpi))
            plt.close(gesture_figure)
    else:
        print('Skipping gesture description/meaning figures '
              '(pass --hierarchical-rgb-run-dir and/or --hierarchical-pose-run-dir).', flush=True)
    for path in paths:
        print(f'Saved {path}')
if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'Comparison failed: {exc}') from exc
