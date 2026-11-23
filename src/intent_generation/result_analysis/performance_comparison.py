#!/usr/bin/env python3
"""Compare training strategies using SAVED metric values only.
Trained runs: read run_dir/metrics.jsonl, use only each record's val object,
select val.<intent/reasoning metric>_best for the six pairwise metrics per field,
and use val.intent_bleu / val.reasoning_bleu as logged. The *_best values already
are per-sample best-reference scores averaged over unique validation inputs.
Choose ONE whole epoch per run by the highest sum of the 14 plotted values;
ties choose the earlier epoch. --selection-metric can override the raw sum.
Zero-shot baselines: read first_val_responses.json (or its legacy filename)
separately from single-run-dir and hierarchical-run-dir. Use each file's stored
per-reference scores: group by video/gesture, take each metric's maximum over
references, then average equally over samples. Neither baseline substitutes for
the other. Plotted order: single zero shot, hierarchical zero shot, single
supervised, hierarchical supervised.
No metric computation or neural scorer/model loading occurs. No epoch response
JSONL or metrics.py is read. Missing saved metrics cause explicit errors; the
script never falls back to loss values, *_mean fields, or recomputation.
BLEU aggregation differs in these stored sources: baseline scores are mean
best-reference SENTENCE BLEU, while the training log stores CORPUS BLEU. The
plot, terminal and selection report disclose this; no best-reference corpus
BLEU is invented from the saved corpus values. Pairwise metrics are directly
aligned to the requested best-reference aggregation.
Example from intent_generation/:
  python result_analysis/performance_comparison.py \\
      --single-run-dir runs/single \\
      --hierarchical-run-dir runs/hierarchical
Relative run paths are resolved under this project root when needed, so both
runs/hierarchical and ../runs/hierarchical work from intent_generation/.
Exports PNG/PDF/SVG, a per-epoch metric CSV and a JSON selection report.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
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
SAVED_COLUMNS = {key: key if key.endswith('_bleu') else key + '_best' for key in LABELS}
SERIES_LABELS = (
    'Single prompt (zero shot)',
    'Hierarchical prompt (zero shot)',
    'Single prompt (supervised)',
    'Hierarchical prompt (supervised)',
)
COLORS = dict(zip(SERIES_LABELS, ('#B8B6B3', '#C3B2D0', '#A8BECC', '#D8B5BE')))
BLEU_NOTE = ('BLEU uses different saved aggregations: zero-shot runs = mean best-reference sentence BLEU; '
             'trained runs = logged corpus BLEU. These BLEU values are not directly comparable.')
@dataclass
class EpochSummary:
    path: Path
    epoch: int
    metrics: dict[str, float]
    sample_count: int | None
    reference_count: int | None
    selection_score: float
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
def score_epoch(metrics: Mapping[str, float], selection_metric: str = 'sum') -> float:
    if selection_metric == 'sum':
        return math.fsum(metrics.values())
    if selection_metric == 'scaled_mean':
        return math.fsum(value / 100. if key.endswith('_bleu') else value for key, value in metrics.items()) / len(metrics)
    return metrics[selection_metric]
def optional_count(values: Mapping, key: str, path: Path, epoch: int) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f'{path}, epoch {epoch}: invalid {key} count')
    return int(value)
def best_metrics_from_log(path: Path, *, selection_metric: str = 'sum',
    allow_cohort_differences: bool = False) -> tuple[EpochSummary, list[EpochSummary]]:
    """Use already aggregated val.*_best scores; never average epoch means again."""
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
        missing = [source for source in SAVED_COLUMNS.values() if source not in val]
        if missing:
            raise ValueError(f'{path}, epoch {epoch}: missing saved val metrics {missing}. '
                             'Use the log containing *_best fields; scores will not be recomputed.')
        metrics = {key: finite_score(val[source], f'{path}, epoch {epoch}, val.{source}')
        for key, source in SAVED_COLUMNS.items()}
        count = optional_count(val, 'generation_unique_inputs', path, epoch)
        references = optional_count(val, 'generation_annotation_references', path, epoch)
        summaries.append(EpochSummary(path, epoch, metrics, count, references, score_epoch(metrics, selection_metric)))
    if not summaries:
        raise ValueError(f'{path}: no trained validation epochs')
    summaries.sort(key=lambda summary: summary.epoch)
    known_counts = {item.sample_count for item in summaries if item.sample_count is not None}
    if len(known_counts) > 1 and not allow_cohort_differences:
        raise ValueError(f'{path}: validation input counts change across epochs. '
                         'Use matched cohorts, or --allow-cohort-differences for intentional unmatched comparisons.')
    print(f'\nSaved validation metrics: {path}', flush=True)
    for item in summaries:
        print(f'  epoch={item.epoch:04d} samples={item.sample_count if item.sample_count is not None else "unknown"} '
              f'references={item.reference_count if item.reference_count is not None else "unknown"} '
              f'{selection_metric}={item.selection_score:.6f}', flush=True)
    best = max(summaries, key=lambda item: (item.selection_score, -item.epoch))
    return best, summaries
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
def baseline_metrics(path: Path, *, selection_metric: str = 'sum') -> EpochSummary:
    grouped, predictions = {}, {}
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
        for key in LABELS:
            value = finite_score(metrics.get(key), f'{path}, record {index}, {key}')
            sample[key] = max(sample.get(key, -math.inf), value)
    means = {key: math.fsum(values[key] for values in grouped.values()) / len(grouped) for key in LABELS}
    print(f'Zero-shot baseline: {path} (saved scores only)', flush=True)
    return EpochSummary(path, 0, means, len(grouped), len(rows), score_epoch(means, selection_metric))
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


def grouped_bars(ax, labels: list[str], single_zero_shot: list[float],
                 hierarchical_zero_shot: list[float], single_supervised: list[float],
                 hierarchical_supervised: list[float], ylabel: str, title: str,
                 *, bleu: bool = False, show_values: bool = True) -> None:
    """Draw four centered bars per group in the documented series order."""
    positions = np.array([1.15, 3.85]) if bleu else np.arange(len(labels), dtype=float)
    if len(positions) != len(labels):
        raise ValueError('The BLEU panel requires intent and reasoning BLEU')
    width, step = 0.15, 0.175
    series = (single_zero_shot, hierarchical_zero_shot, single_supervised, hierarchical_supervised)
    offsets = (np.arange(len(SERIES_LABELS)) - (len(SERIES_LABELS) - 1) / 2) * step
    for offset, values, name in zip(offsets, series, SERIES_LABELS):
        if len(values) != len(labels):
            raise ValueError(f'{name}: metric values and labels must have equal length')
        bars = ax.bar(positions + offset, values, width, label=name,
                      color=COLORS[name], edgecolor='white', linewidth=0.5, zorder=3)
        if show_values:
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=5.5, color='#3C4249')
    ax.set_xlim(-0.6, 5.6)
    ax.set_xticks(positions, labels)
    ax.tick_params(axis='x', length=0, pad=8, labelsize=10)
    ax.tick_params(axis='y', length=0, pad=7, labelsize=9)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=10)
    ax.set_title(title, loc='left', pad=15, fontsize=12, fontweight='semibold')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E6E8EB', linewidth=0.65)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#B8BDC4')


def build_figure(single_zero_shot: dict[str, float], hierarchical_zero_shot: dict[str, float],
                 single_supervised: dict[str, float], hierarchical_supervised: dict[str, float],
                 *, show_values: bool = True):
    """Return the four-series figure without modifying input metric values."""
    intent_keys = [key for key in LABELS if key.startswith('intent_') and not key.endswith('_bleu')]
    reasoning_keys = [key for key in LABELS if key.startswith('reasoning_') and not key.endswith('_bleu')]
    bleu_keys = [key for key in LABELS if key.endswith('_bleu')]
    series = (single_zero_shot, hierarchical_zero_shot, single_supervised, hierarchical_supervised)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(3, 1, figsize=(9.4, 9.6),
                                    gridspec_kw={'height_ratios': [1, 1, 0.62]})
        figure.subplots_adjust(left=0.095, right=0.985, bottom=0.075,
                               top=0.825, hspace=0.46)
        figure.suptitle('Validation performance across prompting and training strategies',
                        x=0.080, y=0.978, ha='left', fontsize=12.5, fontweight='semibold')
        figure.text(0.080, 0.944, 'Saved best-reference averages | One selected epoch per supervised run',
                    fontsize=10, color='#737982')
        handles = [Patch(facecolor=COLORS[name], edgecolor='none', label=name) for name in SERIES_LABELS]
        # Matplotlib fills a multi-column legend by column. Reorder handles to
        # keep each row aligned with the left-to-right order of the bars.
        legend_handles = [handles[i] for i in (0, 2, 1, 3)]
        figure.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.075, 0.916),
                      ncol=2, frameon=False, fontsize=8, handlelength=1.6,
                      handleheight=0.85, columnspacing=2.2, borderaxespad=0)
        for ax, keys, title, prefix, ylabel, bleu in (
            (axes[0], intent_keys, '(a) Communicative intent', 'Intent ', 'Mean score', False),
            (axes[1], reasoning_keys, '(b) Reasoning', 'Reasoning ', 'Mean score', False),
            (axes[2], bleu_keys, '(c) BLEU (saved aggregation)', '', 'BLEU (0-100)', True),
        ):
            grouped_bars(ax, [LABELS[key].removeprefix(prefix) for key in keys],
                         *([data[key] for key in keys] for data in series), ylabel, title,
                         bleu=bleu, show_values=show_values)
        # Use the same scale for the first two panels, including negative BLEURT.
        regular_keys = intent_keys + reasoning_keys
        values = [data[key] for data in series for key in regular_keys]
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
        bleu_values = [data[key] for data in series for key in bleu_keys]
        low, high = min(0.0, *bleu_values), max(0.0, *bleu_values)
        span = high - low or 1.0
        axes[2].set_ylim(low - 0.15 * span if low < 0 else 0.0, high + 0.23 * span)
        axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        figure.text(0.080, 0.020, 'Pairwise: best-reference averages. BLEU: zero-shot sentence mean; supervised corpus score.',
                    fontsize=9, color='#737982')
    return figure


def save_figure(figure, output: Path, dpi: int = 300) -> list[Path]:
    """Save the requested format plus PDF/SVG copies for clean resizing."""
    output = output.expanduser()
    if not output.suffix:
        output = output.with_suffix('.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = list(dict.fromkeys((output, output.with_suffix('.pdf'), output.with_suffix('.svg'))))
    with plt.rc_context(STYLE):
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor='white')
    return paths


def save_summaries(output: Path, single_zero_shot: EpochSummary, hierarchical_zero_shot: EpochSummary,
                   single: EpochSummary, single_epochs: list[EpochSummary],
                   hierarchical: EpochSummary, hierarchical_epochs: list[EpochSummary],
                   *, selection_metric: str):
    csv_path = output.with_name(output.stem + '_epoch_metrics.csv')
    report_path = output.with_name(output.stem + '_selection.json')
    columns = ['series', 'epoch', 'selected', 'sample_count', 'reference_count',
               'selection_metric', 'selection_score', 'source_file', *LABELS]
    selected_series = (single_zero_shot, hierarchical_zero_shot, single, hierarchical)
    all_epochs = ([single_zero_shot], [hierarchical_zero_shot], single_epochs, hierarchical_epochs)
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, summaries, selected in zip(SERIES_LABELS, all_epochs, selected_series):
            for item in summaries:
                writer.writerow({'series': name, 'epoch': item.epoch, 'selected': item is selected,
                                 'sample_count': item.sample_count, 'reference_count': item.reference_count,
                                 'selection_metric': selection_metric, 'selection_score': item.selection_score,
                                 'source_file': str(item.path), **item.metrics})
    report = {
        'metric_recomputation': False,
        'model_loading': False,
        'trained_metric_sources': {key: 'val.' + source for key, source in SAVED_COLUMNS.items()},
        'pairwise_aggregation': 'saved mean of per-input maxima across annotation references',
        'baseline_aggregation': 'max over saved reference scores per sample, then equal-weight sample mean',
        'bleu_note': BLEU_NOTE,
        'cohort_check': 'sample counts only; aggregate training logs contain no sample identities',
        'selection_metric': selection_metric,
        'tie_break': 'earliest epoch',
        'series': {},
    }
    for name, item in zip(SERIES_LABELS, selected_series):
        report['series'][name] = {'epoch': item.epoch, 'source_file': str(item.path),
                                 'sample_count': item.sample_count, 'reference_count': item.reference_count,
                                 'selection_score': item.selection_score, 'metrics': item.metrics}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return [csv_path, report_path]


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
    parser.add_argument('--hierarchical-run-dir', type=Path, default=HIERARCHICAL_RUN_DIR)
    parser.add_argument('--single-metrics-jsonl', type=Path, help='Default: single-run-dir/metrics.jsonl')
    parser.add_argument('--hierarchical-metrics-jsonl', type=Path, help='Default: hierarchical-run-dir/metrics.jsonl')
    parser.add_argument('--single-zero-shot-json', '--single-no-training-json', dest='single_zero_shot_json',
                        type=Path, help='Default: single-run-dir/first_val_responses.json')
    parser.add_argument('--hierarchical-zero-shot-json', '--hierarchical-no-training-json', '--no-training-json',
                        dest='hierarchical_zero_shot_json', type=Path,
                        help='Default: hierarchical-run-dir/first_val_responses.json; --no-training-json is a legacy alias')
    parser.add_argument('--selection-metric', choices=('sum', 'scaled_mean', *LABELS), default='sum',
                        help='Select one epoch by this aggregate; default raw sum preserves the existing rule')
    parser.add_argument('--allow-cohort-differences', action='store_true',
                        help='Permit intentional comparisons with different validation input counts')
    parser.add_argument('--no-recompute', action='store_true',
                        help='Compatibility flag: recomputation is always disabled')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--no-values', action='store_true')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    single_dir = resolve_from_project(args.single_run_dir)
    hierarchical_dir = resolve_from_project(args.hierarchical_run_dir)
    single_baseline_path = baseline_file(single_dir, args.single_zero_shot_json)
    hierarchical_baseline_path = baseline_file(hierarchical_dir, args.hierarchical_zero_shot_json)
    single_path = metrics_log_file(single_dir, args.single_metrics_jsonl)
    hierarchical_path = metrics_log_file(hierarchical_dir, args.hierarchical_metrics_jsonl)
    print('Reading saved metric values only. Metric recomputation and scorer/model loading are disabled.', flush=True)
    print(f'Resolved single run: {single_dir}', flush=True)
    print(f'Resolved hierarchical run: {hierarchical_dir}', flush=True)
    single_zero_shot = baseline_metrics(single_baseline_path, selection_metric=args.selection_metric)
    hierarchical_zero_shot = baseline_metrics(hierarchical_baseline_path, selection_metric=args.selection_metric)
    single, single_epochs = best_metrics_from_log(single_path, selection_metric=args.selection_metric,
                                                  allow_cohort_differences=args.allow_cohort_differences)
    hierarchical, hierarchical_epochs = best_metrics_from_log(
        hierarchical_path, selection_metric=args.selection_metric,
        allow_cohort_differences=args.allow_cohort_differences)
    series = (single_zero_shot, hierarchical_zero_shot, single, hierarchical)
    counts = {item.sample_count for item in series if item.sample_count is not None}
    if len(counts) > 1 and not args.allow_cohort_differences:
        raise ValueError('Zero-shot and supervised validation input counts differ. Use matched data, '
                         'or --allow-cohort-differences for an intentional unmatched comparison.')
    print('\nPairwise metrics: saved val.*_best fields; BLEU: saved val.intent_bleu / val.reasoning_bleu.')
    for name, item in zip(SERIES_LABELS[2:], (single, hierarchical)):
        print(f'{name}: epoch={item.epoch:04d}, {args.selection_metric}={item.selection_score:.6f}')
    print(BLEU_NOTE)
    print(f'{"metric":<28}' + ''.join(f'{name:>35}' for name in SERIES_LABELS))
    for key in LABELS:
        print(f'{key:<28}' + ''.join(f'{item.metrics[key]:35.4f}' for item in series))
    figure = build_figure(*(item.metrics for item in series), show_values=not args.no_values)
    figure.text(0.975, 0.850, f'Supervised epochs — single: {single.epoch} | hierarchical: {hierarchical.epoch}',
                ha='right', fontsize=8, color='#737982')
    output = args.output.expanduser()
    if not output.suffix:
        output = output.with_suffix('.png')
    paths = save_figure(figure, output, dpi=args.dpi)
    plt.close(figure)
    paths.extend(save_summaries(output, single_zero_shot, hierarchical_zero_shot,
                                single, single_epochs, hierarchical, hierarchical_epochs,
                                selection_metric=args.selection_metric))
    for path in paths:
        print(f'Saved {path}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'Comparison failed: {exc}') from exc
