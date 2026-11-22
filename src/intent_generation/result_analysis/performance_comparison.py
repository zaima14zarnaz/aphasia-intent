#!/usr/bin/env python3
"""Compare no training, single-prompt training, and hierarchical-prompt training.

Only the 14 intent/reasoning metric keys below are read. Additional gesture
metrics, statuses, and exact-match rates are ignored. Each plotted value is the
arithmetic mean of the per-reference values in the validation response JSON.
Intent and reasoning have separate panels; sentence BLEU stays on a 0-100 scale.

Run from the existing plotting script's location, or specify --run-dir:
    python plot_performance_comparison.py
    python plot_performance_comparison.py --run-dir /path/to/runs/20261118_135805

Exports the figure as PNG plus editable PDF/SVG copies. All metric aggregation,
input paths and training-strategy colors are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1] / 'runs' / '20261118_135805'
HIERARCHICAL_JSON = Path(
    '/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/'
    'runs/20261121_170332/final_val_responses.json'
)
OUTPUT = Path(__file__).resolve().parent / 'performance_comparison.png'

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

SERIES_LABELS = ('No training', 'Single Prompt Training', 'Hierarchical Prompt Training')
COLORS = {
    'No training': '#B8B6B3',                   # Gray.
    'Single Prompt Training': '#A8BECC',        # Muted, softly warm light blue.
    'Hierarchical Prompt Training': '#D8B5BE',  # Muted warm light pink.
}


def mean_metrics(path: Path) -> dict[str, float]:
    """Whitelist intent/reasoning fields; never average gesture metrics/statuses."""
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Expected a nonempty list of validation rows in {path}')
    collected = {key: [] for key in LABELS}
    for index, row in enumerate(rows):
        metrics = row.get('metrics') if isinstance(row, dict) else None
        if not isinstance(metrics, dict):
            raise ValueError(f'Missing metrics object in {path}, row {index + 1}')
        for key in LABELS:
            value = metrics.get(key)
            if value is None or isinstance(value, bool):
                raise ValueError(f'Missing/non-numeric {key!r} in {path}, row {index + 1}')
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Non-numeric {key!r} in {path}, row {index + 1}') from exc
            if not math.isfinite(number):
                raise ValueError(f'Nonfinite {key!r} in {path}, row {index + 1}')
            collected[key].append(number)
    return {key: math.fsum(values) / len(values) for key, values in collected.items()}


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


def grouped_bars(ax, labels: list[str], first: list[float], final: list[float],
                 hierarchical: list[float], ylabel: str, title: str,
                 *, bleu: bool = False, show_values: bool = True) -> None:
    # The BLEU panel has the same bar width as the other panels. Its two groups
    # are centered in the same horizontal span instead of becoming oversized.
    positions = np.array([1.15, 3.85]) if bleu else np.arange(len(labels), dtype=float)
    if len(positions) != len(labels):
        raise ValueError('The BLEU panel requires intent and reasoning BLEU')
    # Keep the inter-bar gap fixed at 0.025 while thinning the bars.
    width, gap = 0.15, 0.025
    step = width + gap
    for offset, values, name in zip((-step, 0.0, step), (first, final, hierarchical), SERIES_LABELS):
        if len(values) != len(labels):
            raise ValueError(f'{name}: metric values and labels must have equal length')
        bars = ax.bar(positions + offset, values, width, label=name,
                      color=COLORS[name], edgecolor='white', linewidth=0.5, zorder=3)
        if show_values:
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=5.5, color='#3C4249')
    ax.set_xlim(-0.55, 5.55)
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


def build_figure(first: dict[str, float], final: dict[str, float],
                 hierarchical: dict[str, float], *, show_values: bool = True):
    """Return a publication-style figure without modifying any input values."""
    intent_keys = [key for key in LABELS if key.startswith('intent_') and not key.endswith('_bleu')]
    reasoning_keys = [key for key in LABELS if key.startswith('reasoning_') and not key.endswith('_bleu')]
    bleu_keys = [key for key in LABELS if key.endswith('_bleu')]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(3, 1, figsize=(9.4, 9.6),
                                    gridspec_kw={'height_ratios': [1, 1, 0.62]})
        figure.subplots_adjust(left=0.095, right=0.985, bottom=0.075,
                               top=0.845, hspace=0.46)
        figure.suptitle('Validation performance across training strategies',
                        x=0.080, y=0.978, ha='left', fontsize=16, fontweight='semibold')
        figure.text(0.080, 0.944, 'Intent and reasoning | Mean scores over annotation references',
                    fontsize=10, color='#737982')
        handles = [Patch(facecolor=COLORS[name], edgecolor='none', label=name) for name in SERIES_LABELS]
        figure.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.075, 0.916),
                      ncol=3, frameon=False, fontsize=10, handlelength=1.6,
                      handleheight=0.85, columnspacing=2.2, borderaxespad=0)
        for ax, keys, title, prefix, ylabel, bleu in (
            (axes[0], intent_keys, '(a) Communicative intent', 'Intent ', 'Mean score', False),
            (axes[1], reasoning_keys, '(b) Reasoning', 'Reasoning ', 'Mean score', False),
            (axes[2], bleu_keys, '(c) Sentence BLEU', '', 'Mean sentence BLEU', True),
        ):
            grouped_bars(ax, [LABELS[key].removeprefix(prefix) for key in keys],
                         [first[key] for key in keys], [final[key] for key in keys],
                         [hierarchical[key] for key in keys], ylabel, title,
                         bleu=bleu, show_values=show_values)
        # Match the first two scales for direct comparison. BLEURT may be
        # negative, so expand below zero when needed instead of clipping it.
        regular_keys = intent_keys + reasoning_keys
        values = [data[key] for data in (first, final, hierarchical) for key in regular_keys]
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
        bleu_values = [data[key] for data in (first, final, hierarchical) for key in bleu_keys]
        low, high = min(0.0, *bleu_values), max(0.0, *bleu_values)
        span = high - low or 1.0
        axes[2].set_ylim(low - 0.15 * span if low < 0 else 0.0, high + 0.23 * span)
        axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        figure.text(0.080, 0.020, 'Higher is better for all metrics. BLEU is shown on its original 0-100 scale.',
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


def response_file(run_dir: Path, phase: str) -> Path:
    """Keep the existing filenames; also accept the newer shorter aliases."""
    candidates = (run_dir / f'{phase}_val_set_responses.json', run_dir / f'{phase}_val_responses.json')
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Neither validation file exists: {candidates[0]} or {candidates[1]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR, help='Run containing the existing no-training and single-prompt responses')
    parser.add_argument('--hierarchical-json', type=Path, default=HIERARCHICAL_JSON, help='Hierarchical final validation responses')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Output figure path, e.g. PNG, PDF, or SVG')
    parser.add_argument('--dpi', type=int, default=300, help='Raster resolution; default 300 dpi')
    parser.add_argument('--no-values', action='store_true', help='Omit numeric labels for a cleaner small figure')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')

    first = mean_metrics(response_file(args.run_dir.expanduser(), 'first'))
    final = mean_metrics(response_file(args.run_dir.expanduser(), 'final'))
    hierarchical = mean_metrics(args.hierarchical_json.expanduser())

    print(f'{"metric":<28} {SERIES_LABELS[0]:>14} {SERIES_LABELS[1]:>24} {SERIES_LABELS[2]:>30}')
    for key in LABELS:
        print(f'{key:<28} {first[key]:14.4f} {final[key]:24.4f} {hierarchical[key]:30.4f}')

    figure = build_figure(first, final, hierarchical, show_values=not args.no_values)
    paths = save_figure(figure, args.output, dpi=args.dpi)
    plt.close(figure)
    for path in paths:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
