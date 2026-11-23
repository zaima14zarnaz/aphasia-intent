#!/usr/bin/env python3
"""Preprocess, group-split, train, and evaluate aphasia intent generation.

Supports the existing single-prompt model and model/p_A1_A2_B_C.py.
Existing C intent/reasoning loss, missing-field penalty, frame-selection loss,
splits, optimizer and generation metrics are retained. Optional supervised terms:
  A2 'gesture movement description' <- CSV 'movement'
  B  'gesture meaning'              <- CSV 'meaning'
The description extractor also accepts an A1 field for older response packets.
Only examples with a nonempty generated field AND nonempty ground truth receive
that auxiliary loss. Missing annotation values are never replaced by pseudo-GT.
A1 conversation topic and C gesture contribution have no requested annotation
loss. Generated evidence, never annotation evidence, feeds later prompts.

Response JSON files include per-reference auxiliary CE/status and aggregate
reporting is added to metrics.csv. Validation scores BERTScore F1, ROUGE-1/2/L
F1, METEOR, BLEURT and BLEU for all four fields and prints them to the terminal.
Pairwise metrics retain best/mean over references; BLEU remains multireference
corpus BLEU. References without optional GT are excluded for that field, while
missing predictions with available GT are scored as empty strings. Exact optional CSV column names can be
changed with --gesture-description-column / --gesture-meaning-column or the
existing --column-map. Old manifests are enriched from their exact CSV source
record, retaining the reference/annotator pairing. No dataloader changes needed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from dataloader import create_dataloader, split_media_ids
from intent_output_formatter import (EXPECTED_FIELDS, annotation_targets,
                                     format_token_output, intent_output_formatter)
from losses import CombinedTrainingLoss
from metrics import (DEFAULT_METRICS_CSV, GenerationMetrics, MetricsConfig,
                     ValidationAccumulator, append_epoch_metrics, print_validation_metrics)
from pre_process import DATASET_DIR, RGB_DIR, LANDMARKS_DIR, preprocess

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_ORDER = ('Prompt A1', 'Prompt A2', 'Prompt B', 'Prompt C')
PROMPT_FIELDS = {
    'Prompt A1': ('conversation topic',),
    'Prompt A2': ('gesture movement description',),
    'Prompt B': ('gesture meaning',),
    'Prompt C': ('communicative intent', 'reasoning', 'gesture contribution'),
}
AUX_FIELDS = {
    'gesture movement description': ('movement_description', 'gesture_description'),
    'gesture meaning': ('gesture_meaning', 'gesture_meaning'),
}
PAIR_METRIC_NAMES = ('bertscore_f1', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'meteor', 'bleurt')


@dataclass
class OptionalMetricSample:
    media_id: str
    transcript: str
    prediction: str | None
    references: dict[str, str]


class FourFieldValidationAccumulator(ValidationAccumulator):
    """Keep existing C accumulation and independent optional-field references.

    References remain grouped by (media ID, transcript), never by annotator row.
    Missing GT is excluded for that field only. A missing generated field with
    available GT remains a scored empty candidate, matching the existing engine.
    """

    def __init__(self):
        super().__init__()
        self.optional_samples = {field: {} for field in AUX_FIELDS}
        self.optional_reference_values = {field: {} for field in AUX_FIELDS}

    def update_optional(self, *, media_ids, transcripts, packets, truths, reference_ids):
        n = len(media_ids)
        if any(len(values) != n for values in (transcripts, packets, reference_ids, *truths.values())):
            raise ValueError('Optional metric prediction/reference batch lengths differ')
        for i, (media, transcript, packet, reference_id) in enumerate(zip(media_ids, transcripts, packets, reference_ids)):
            key = (media, transcript)
            if self.reference_keys.get(reference_id) != key:
                raise ValueError('Update base validation references before optional metrics')
            for field in AUX_FIELDS:
                prediction = packet.fields.get(field)
                truth = truths[field][i]
                truth = truth if isinstance(truth, str) and truth.strip() else None
                seen = self.optional_reference_values[field]
                if reference_id in seen and seen[reference_id] != truth:
                    raise ValueError(f'Optional reference changed within the epoch: {reference_id}, {field}')
                seen[reference_id] = truth
                samples = self.optional_samples[field]
                if key not in samples:
                    samples[key] = OptionalMetricSample(media, transcript, prediction, {})
                sample = samples[key]
                if sample.prediction != prediction:
                    raise ValueError(f'Multiple predictions for the same validation input: {field}')
                if truth is not None:
                    sample.references[reference_id] = truth


def compute_all_generation_metrics(engine, accumulator: FourFieldValidationAccumulator):
    """Use the existing metric backends/settings for all four response fields.

    Existing intent/reasoning aggregate scores are returned unchanged by
    engine.compute(). Extra pairwise scores are computed in one backend call for
    both gesture fields. Best/mean is over references, then equally over unique
    inputs; BLEU uses the same native multireference corpus settings and 0-100
    scale. Per-reference JSON entries also include single-reference sentence BLEU.

    Old metrics.py versions without reference_metrics are supported by scoring
    the C reference pairs in the same extra call to reconstruct those JSON entries.
    No extra model generation, losses, or optimizer updates are introduced.
    """
    result = dict(engine.compute(accumulator))
    existing = getattr(engine, 'reference_metrics', None)
    have_base_rows = (isinstance(existing, Mapping)
                      and all(isinstance(existing.get(ref), Mapping) for ref in accumulator.reference_keys))
    by_reference = {ref: copy.deepcopy(dict(existing[ref])) if have_base_rows else {}
                    for ref in accumulator.reference_keys}
    candidates, references, requests, ranges = [], [], [], {}
    active_samples = {}
    for field, (_, prefix) in AUX_FIELDS.items():
        samples = [sample for sample in accumulator.optional_samples[field].values() if sample.references]
        active_samples[field] = samples
        for index, sample in enumerate(samples):
            start = len(candidates)
            for ref_id, reference in sample.references.items():
                candidates.append(sample.prediction or '')
                references.append(reference)
                requests.append((prefix, ref_id))
            ranges[(field, index)] = (start, len(candidates))
        for ref_id, truth in accumulator.optional_reference_values[field].items():
            status = ('missing_ground_truth' if truth is None else
                      'missing_response_scored_as_empty'
                      if not accumulator.optional_samples[field][accumulator.reference_keys[ref_id]].prediction else 'scored')
            by_reference[ref_id][prefix + '_metric_status'] = status
            for metric in (*PAIR_METRIC_NAMES, 'bleu'):
                by_reference[ref_id][prefix + '_' + metric] = None
    if not have_base_rows:
        for field, prefix in zip(EXPECTED_FIELDS, ('intent', 'reasoning')):
            for sample in accumulator.samples.values():
                for ref_id, reference in sample.references.items():
                    candidates.append(sample.prediction.fields[field] or '')
                    references.append(reference[field])
                    requests.append((prefix, ref_id))
    scores = engine._pair_scores(candidates, references) if candidates else {metric: [] for metric in PAIR_METRIC_NAMES}
    for metric in PAIR_METRIC_NAMES:
        values = scores[metric]
        if len(values) != len(candidates) or any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f'{metric} returned wrong-length or nonfinite gesture scores')
    bleu = engine.bleu_class(tokenize='13a', smooth_method='exp', effective_order=True)
    for index, (prefix, ref_id) in enumerate(requests):
        row = by_reference[ref_id]
        for metric in PAIR_METRIC_NAMES:
            row[prefix + '_' + metric] = float(scores[metric][index])
        row[prefix + '_bleu'] = float(bleu.sentence_score(candidates[index], [references[index]]).score)
    for field, (_, prefix) in AUX_FIELDS.items():
        samples = active_samples[field]
        count = sum(len(sample.references) for sample in samples)
        result[prefix + '_generation_unique_inputs'] = len(samples)
        result[prefix + '_generation_unique_clips'] = len({sample.media_id for sample in samples})
        result[prefix + '_annotation_references'] = count
        result[prefix + '_reference_coverage'] = count / len(accumulator.reference_keys)
        result[prefix + '_missing_field_rate'] = (statistics.mean(not sample.prediction for sample in samples)
                                                  if samples else None)
        for metric in PAIR_METRIC_NAMES:
            best, mean = [], []
            for index in range(len(samples)):
                start, stop = ranges[(field, index)]
                values = [float(value) for value in scores[metric][start:stop]]
                best.append(max(values))
                mean.append(statistics.mean(values))
            result[f'{prefix}_{metric}_best'] = statistics.mean(best) if best else None
            result[f'{prefix}_{metric}_mean'] = statistics.mean(mean) if mean else None
        if samples:
            hypotheses = [sample.prediction or '' for sample in samples]
            grouped_refs = [list(sample.references.values()) for sample in samples]
            documents = [[refs[j] if j < len(refs) else None for refs in grouped_refs]
                         for j in range(max(map(len, grouped_refs)))]
            result[prefix + '_bleu'] = float(bleu.corpus_score(hypotheses, documents).score)
        else:
            result[prefix + '_bleu'] = None
    return result, by_reference


def print_all_validation_metrics(values, epoch: int) -> None:
    """Keep the original C table, then print both gesture fields identically."""
    print_validation_metrics(values, epoch)
    print('\nGesture generation metrics (best/mean over available references)', flush=True)
    print(f'{"Metric":<23} {"Desc best":>12} {"Desc mean":>12} {"Meaning best":>12} {"Meaning mean":>12}', flush=True)
    def show(value):
        return f'{value:.4f}' if value is not None else 'n/a'
    for metric in PAIR_METRIC_NAMES:
        items = [values[f'{prefix}_{metric}_{aggregation}']
                 for prefix in ('gesture_description', 'gesture_meaning') for aggregation in ('best', 'mean')]
        print(f'{metric:<23}' + ''.join(f'{show(value):>13}' for value in items), flush=True)
    print('BLEU (corpus, 0-100)     '
          f'description={show(values["gesture_description_bleu"])}  meaning={show(values["gesture_meaning_bleu"])}', flush=True)
    for prefix in ('gesture_description', 'gesture_meaning'):
        missing = values[prefix + '_missing_field_rate']
        missing_text = f'{100 * missing:.2f}%' if missing is not None else 'n/a'
        print(f'{prefix}: {values[prefix + "_generation_unique_inputs"]} annotated inputs, '
              f'{values[prefix + "_annotation_references"]} references; missing prediction={missing_text}', flush=True)


def load_model_module(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Model module not found: {path}')
    name = 'aphasia_intent_architecture'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def resolve_from_root(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def timestamped_run_dir() -> Path:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate, suffix = PROJECT_ROOT / 'runs' / stamp, 2
    while candidate.exists():
        candidate = PROJECT_ROOT / 'runs' / f'{stamp}_{suffix}'
        suffix += 1
    return candidate


def video_filename(video_id: str) -> str:
    name = video_id.strip()
    if not name:
        raise ValueError('Validation sample is missing a video id')
    return name if name.lower().endswith('.mp4') else f'{name}.mp4'


def gesture_index(media_id: str) -> int:
    match = re.fullmatch(r'gest_([0-9]+)', media_id.rsplit('/', 1)[-1])
    if not match:
        raise ValueError(f'Cannot parse gesture index from {media_id!r}')
    return int(match.group(1))


def attach_sample_metrics(records: list[dict], by_reference: dict[str, dict]) -> None:
    for record in records:
        reference_id = record.pop('reference_id')
        if reference_id not in by_reference:
            raise KeyError(f'No metrics for validation reference {reference_id}')
        record['metrics'] = by_reference[reference_id]


def column_mapping(args) -> dict[str, str]:
    mapping = (json.loads(resolve_from_root(args.column_map).read_text(encoding='utf-8'))
               if args.column_map else {})
    if not isinstance(mapping, dict):
        raise ValueError('column-map must be a JSON object')
    if any(not isinstance(value, str) or not value.strip() for value in mapping.values()):
        raise ValueError('column-map values must be nonempty CSV header strings')
    mapping = dict(mapping)
    for key, option in (('movement_description', 'gesture_description_column'),
                        ('gesture_meaning', 'gesture_meaning_column')):
        override = getattr(args, option, None)
        if override is not None:
            if not override.strip():
                raise ValueError(f'--{option.replace("_", "-")} must not be empty')
            mapping[key] = override
    return mapping


def make_manifest(args) -> Path:
    if args.manifest:
        path = resolve_from_root(args.manifest)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    # Keep C preprocessing unchanged so an absent/blank optional annotation does
    # not reject a usable C reference. Optional GT is added by exact CSV record.
    return preprocess(resolve_from_root(args.dataset_dir), resolve_from_root(args.rgb_dir),
                      resolve_from_root(args.landmarks_dir), args.output_dir / 'preprocessed',
                      column_map=column_mapping(args), skip_invalid=args.skip_invalid,
                      target_fields=('communicative_intent', 'reasoning'))


def enrich_optional_annotations(manifest: dict, args, manifest_dir: Path) -> dict:
    """Preserve coherent references; never join annotators by media id alone.

    source.row_number is a logical CSV record number (header=1), so quoted
    multiline cells work. Embedded canonical GT is usable without the source
    CSV. Every sample gets the same optional keys, including absent values,
    because the existing collator uses the first item's target keys.
    """
    mapping = column_mapping(args)
    columns = {
        'movement_description': mapping.get('movement_description', 'movement'),
        'gesture_meaning': mapping.get('gesture_meaning', 'meaning'),
    }
    cached_csvs, issues, counts = {}, [], defaultdict(int)
    for sample in manifest['samples']:
        source = sample.get('source') or {}
        current = sample.setdefault('targets', {})
        needs = [key for key in columns if not isinstance(current.get(key), str) or not current[key].strip()]
        row = {}
        if needs and source.get('csv_file'):
            csv_path = Path(source['csv_file']).expanduser()
            if not csv_path.is_absolute():
                csv_path = manifest_dir / csv_path
            csv_path = csv_path.resolve()
            if csv_path not in cached_csvs:
                if csv_path.is_file():
                    with csv_path.open(newline='', encoding='utf-8-sig') as handle:
                        reader = csv.DictReader(handle)
                        headers = reader.fieldnames or []
                        if len(set(headers)) != len(headers):
                            raise ValueError(f'Duplicate CSV headers in {csv_path}')
                        cached_csvs[csv_path] = (headers, dict(enumerate(reader, start=2)))
                else:
                    cached_csvs[csv_path] = ([], {})
                    issues.append({'csv_file': str(csv_path), 'reason': 'source_csv_unavailable'})
            headers, rows = cached_csvs[csv_path]
            number = source.get('row_number')
            if type(number) is not int or number < 2:
                raise ValueError(f'Invalid CSV source row for {sample["reference_id"]}: {number!r}')
            if rows and number not in rows:
                raise ValueError(f'CSV source record {number} is absent in {csv_path}')
            row = rows.get(number, {})
            if None in row:
                raise ValueError(f'Incorrect CSV cell count in {csv_path}, record {number}')
            # Detect stale/reordered source records before attaching new targets.
            clip_column = mapping.get('clip_id')
            if clip_column is None:
                clip_column = next((h for h in headers if h.strip().lower() in ('clip_id', 'clip', 'clip_filename')), None)
            expected_clip = sample.get('original_clip_id') or sample.get('clip_id')
            if row and clip_column and expected_clip and str(row.get(clip_column, '')).strip() != expected_clip.strip():
                raise ValueError(f'Source CSV clip changed for {sample["reference_id"]}; regenerate the manifest')
        for key, header in columns.items():
            existing = current.get(key)
            value = existing.strip() if isinstance(existing, str) and existing.strip() else row.get(header, '')
            current[key] = value.strip() if isinstance(value, str) else ''
            counts[key + '_available' if current[key] else key + '_missing'] += 1
    manifest['target_fields'] = list(dict.fromkeys([*manifest['target_fields'], *columns]))
    report = {'columns': columns, 'counts': dict(counts), 'issues': issues}
    print(f'Optional GT columns: {columns}; availability: {dict(counts)}', flush=True)
    return report


def prepare_splits(manifest_path: Path, output_dir: Path, args) -> tuple[Path, dict]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not {'communicative_intent', 'reasoning'}.issubset(manifest['target_fields']):
        raise ValueError('Manifest lacks annotation intent/reasoning targets')
    report = enrich_optional_annotations(manifest, args, manifest_path.parent)
    write_json(output_dir / 'optional_annotation_report.json', report)
    eligible, excluded = set(), []
    for media in sorted({s['media_id'] for s in manifest['samples']}):
        clip = manifest['clips'][media]
        with np.load(manifest_path.parent / clip['landmarks_cache'], allow_pickle=False) as cache:
            valid_frames = (cache['valid'] & (cache['visibility'] > args.visibility_threshold)).any(-1)
        if int(valid_frames.sum()) < args.num_frames:
            excluded.append({'media_id': media, 'valid_frames': int(valid_frames.sum())})
        else:
            eligible.add(media)
    write_json(output_dir / 'excluded_short_clips.json', excluded)
    if excluded and args.short_clip_policy == 'error':
        raise ValueError(f'{len(excluded)} clips have fewer than {args.num_frames} valid frames. '
                         'See excluded_short_clips.json; choose --short-clip-policy skip to exclude them.')
    manifest['samples'] = [s for s in manifest['samples'] if s['media_id'] in eligible]
    manifest['clips'] = {k: v for k, v in manifest['clips'].items() if k in eligible}
    if not manifest['samples']:
        raise ValueError('No eligible annotation references')
    for clip in manifest['clips'].values():
        clip['landmarks_cache'] = str((manifest_path.parent / clip['landmarks_cache']).resolve())
    filtered = output_dir / 'training_manifest.json'
    write_json(filtered, manifest)
    participant_present = [bool(s.get('participant_id')) for s in manifest['samples']]
    grouping = args.split_group
    if grouping == 'auto':
        if all(participant_present):
            grouping = 'participant_id'
        elif not any(participant_present):
            grouping = 'video_id'
        else:
            raise ValueError('Participant IDs are partially missing. Complete them or explicitly choose --split-group video_id')
    groups = {s['participant_id'] if grouping == 'participant_id' else manifest['clips'][s['media_id']]['video_id']
              for s in manifest['samples']}
    if '' in groups or len(groups) < 2:
        raise ValueError('Need at least two nonempty, distinct split groups')
    n_val = max(1, min(len(groups) - 1, round(len(groups) * args.val_fraction)))
    splits = split_media_ids(filtered, val_fraction=n_val / len(groups), test_fraction=0,
                             seed=args.seed, group_by=grouping)
    if not splits['train'] or not splits['val'] or set(splits['train']) & set(splits['val']):
        raise ValueError('Invalid grouped train/validation split')
    split_sets = {k: set(v) for k, v in splits.items()}
    counts = {k: sum(s['media_id'] in split_sets[k] for s in manifest['samples']) for k in ('train', 'val')}
    summary = {'group_by': grouping, 'seed': args.seed, 'requested_val_fraction': args.val_fraction,
               'train_groups': len(groups) - n_val, 'val_groups': n_val,
               'references': counts, 'media_ids': {k: splits[k] for k in ('train', 'val')},
               'excluded_media': len(excluded), 'actual_val_reference_fraction': counts['val'] / sum(counts.values())}
    write_json(output_dir / 'splits.json', summary)
    print(f'Split by {grouping}: {len(groups)-n_val}/{n_val} groups; references train={counts["train"]}, val={counts["val"]}', flush=True)
    return filtered, splits


@dataclass
class ResponsePacket:
    raw: Any
    c_response: Any
    prompts: dict[str, dict]
    fields: dict[str, str]
    diagnostics: dict[str, str]


def parse_object(value) -> tuple[dict, str]:
    """Tolerant extraction; duplicate keys are removed rather than trained on."""
    if isinstance(value, Mapping):
        return dict(value), 'ok'
    if not isinstance(value, str):
        return {}, 'missing_response'
    text = value.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    duplicates = []
    def object_hook(pairs):
        obj, seen, repeated = {}, set(), set()
        for key, item in pairs:
            if key in seen:
                duplicates.append(key)
                repeated.add(key)
            seen.add(key)
            obj[key] = item
        for key in repeated:
            obj.pop(key, None)
        return obj
    try:
        obj = json.loads(text, object_pairs_hook=object_hook)
    except (ValueError, TypeError):
        return {}, 'invalid_json'
    if not isinstance(obj, dict):
        return {}, 'invalid_json'
    return obj, 'duplicate_keys:' + ','.join(sorted(set(duplicates))) if duplicates else 'ok'


def extract_response_packet(response) -> ResponsePacket:
    outer, outer_status = parse_object(response)
    nested = any(name in outer for name in PROMPT_ORDER)
    prompts, diagnostics = {}, {}
    if nested:
        for name in PROMPT_ORDER:
            prompts[name], diagnostics[name] = parse_object(outer.get(name))
        c_response = outer.get('Prompt C', {})
    else:
        prompts['Prompt C'], diagnostics['Prompt C'] = outer, outer_status
        c_response = response
    fields = {}
    # A2 is the architecture's description stage. Also read A1 for older packets.
    for field, sources in {
        'gesture movement description': ('Prompt A2', 'Prompt A1', 'Prompt C'),
        'gesture meaning': ('Prompt B', 'Prompt C'),
    }.items():
        for name in sources:
            value = prompts.get(name, {}).get(field)
            if isinstance(value, str) and value.strip():
                fields[field] = value
                break
    return ResponsePacket(response, c_response, prompts, fields, diagnostics)


def hierarchy_model(vlm) -> bool:
    return isinstance(getattr(vlm, 'prompts', None), Mapping) and hasattr(vlm, '_prepare_stage')


def stage_inputs(name: str, transcripts, packets: list[ResponsePacket]) -> list[dict[str, str]]:
    if name == 'Prompt A1':
        return [{'TRANSCRIPT': text} for text in transcripts]
    if name == 'Prompt A2':
        return [{} for _ in transcripts]
    results = []
    for text, packet in zip(transcripts, packets):
        topic = packet.prompts.get('Prompt A1', {}).get('conversation topic', '')
        topic = topic if isinstance(topic, str) else ''
        if name == 'Prompt B':
            results.append({'GESTURE_MOVEMENT_DESCRIPTION': packet.fields.get('gesture movement description', ''),
                            'CONVERSATION_TOPIC': topic})
        elif name == 'Prompt C':
            results.append({'GESTURE_MEANING': packet.fields.get('gesture meaning', ''),
                            'CONVERSATION_TOPIC': topic, 'TRANSCRIPT': text})
        else:
            raise ValueError(f'Unknown stage {name}')
    return results


@torch.no_grad()
def generate_packets(vlm, rgb, transcripts, args) -> list[ResponsePacket]:
    """Retain partial stage responses so optional fields can actually be skipped.

    The model's public generate_json enforces complete JSON at every stage. This
    trainer instead retains malformed/missing responses with explicit diagnostics
    and passes an empty string for unavailable upstream evidence. Annotation text
    is never inserted into these generation inputs. Raw JSON remains in reports.
    """
    if not hierarchy_model(vlm):
        texts = vlm.generate_text(rgb, transcripts, max_new_tokens=args.max_new_tokens)
        if len(texts) != len(transcripts):
            raise ValueError('Generation returned an unexpected batch size')
        return [extract_response_packet(text) for text in texts]
    packets = [extract_response_packet({name: {} for name in PROMPT_ORDER}) for _ in transcripts]
    raw = [{} for _ in transcripts]
    was_training = vlm.training
    vlm.eval()
    try:
        for name in PROMPT_ORDER:
            batch = vlm._prepare_stage(name, stage_inputs(name, transcripts, packets),
                                       rgb if name == 'Prompt A2' else None)
            check_sequence_length(batch, args, name)
            ids = vlm.model.generate(**batch.model_inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, use_cache=True,
                                     pad_token_id=vlm.processor.tokenizer.pad_token_id)
            continuation = ids[:, batch.model_inputs['input_ids'].shape[1]:]
            texts = vlm.processor.tokenizer.batch_decode(continuation, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
            if len(texts) != len(transcripts):
                raise ValueError(f'{name}: generation returned an unexpected batch size')
            for i, text in enumerate(texts):
                raw[i][name] = text
                packets[i] = extract_response_packet(raw[i])
        return packets
    finally:
        vlm.train(was_training)


def check_sequence_length(prepared, args, name='Prompt C') -> None:
    if prepared.model_inputs['input_ids'].shape[1] > args.max_sequence_length:
        raise ValueError(f'{name} exceeds --max-sequence-length={args.max_sequence_length}; '
                         'reduce image size or inspect long transcripts/targets (no silent truncation)')


def optional_targets(batch: dict, n: int) -> dict[str, list[str | None]]:
    result = {}
    for field, (canonical, _) in AUX_FIELDS.items():
        values = batch['targets'].get(canonical, [None] * n)
        if len(values) != n:
            raise ValueError(f'GT batch size mismatch for {canonical}')
        result[field] = [value if isinstance(value, str) and value.strip() else None for value in values]
    return result


def eligible_auxiliary(packets, truths, field) -> tuple[list[bool], list[str]]:
    eligible, statuses = [], []
    for packet, truth in zip(packets, truths):
        status = ('missing_response' if field not in packet.fields else
                  'missing_ground_truth' if truth is None else 'computed')
        eligible.append(status == 'computed')
        statuses.append(status)
    return eligible, statuses


def prepare_supervision(vlm, rgb, transcripts, c_targets, packets, truths, args):
    """Build GT target tokens; unsupervised schema fields carry no loss.

    For the strict multi-prompt wrapper, generated values fill unannotated schema
    fields solely to serialize a complete response. Only GT field masks below
    are ever supervised. These generated values are not pseudo-labels.
    """
    if not hierarchy_model(vlm):
        required = tuple(getattr(vlm, 'output_fields', EXPECTED_FIELDS))
        targets = []
        for i, c_target in enumerate(c_targets):
            obj = {}
            for field in required:
                value = c_target.get(field)
                if value is None:
                    value = truths.get(field, [None] * len(c_targets))[i]
                if value is None:
                    value = packets[i].prompts.get('Prompt C', {}).get(field, '')
                obj[field] = value if isinstance(value, str) else ''
            targets.append(obj)
        prepared = vlm.prepare_inputs(rgb, transcripts, targets)
        check_sequence_length(prepared, args)
        return prepared
    targets = []
    for item, packet in zip(c_targets, packets):
        contribution = packet.prompts.get('Prompt C', {}).get('gesture contribution', '')
        targets.append({**item, 'gesture contribution': contribution if isinstance(contribution, str) else ''})
    prepared = vlm._prepare_stage('Prompt C', stage_inputs('Prompt C', transcripts, packets), targets=targets)
    check_sequence_length(prepared, args)
    return prepared


def field_ce(logits, labels, mask, eligible):
    """Differentiable causal CE, mean over eligible examples; absent -> None.

    Each example's loss is its mean GT-token NLL. JSON key/syntax tokens assigned
    to the field by the model are included, matching existing C field masks.
    Shift masks with the labels; never decode/compare strings to create a loss.
    """
    if labels is None or mask is None:
        raise ValueError('Field loss requires GT labels and its token mask')
    use = mask[:, 1:].bool() & (labels[:, 1:] != -100)
    use = use & torch.tensor(eligible, device=labels.device, dtype=torch.bool)[:, None]
    counts = use.sum(-1)
    if any(eligible[i] and counts[i] == 0 for i in range(len(eligible))):
        raise ValueError('An eligible annotation has no supervised field tokens')
    if not use.any():
        return None, [None] * len(eligible)
    losses = F.cross_entropy(logits[:, :-1][use].float(), labels[:, 1:][use], reduction='none')
    token_losses = losses.new_zeros(use.shape).masked_scatter(use, losses)
    per_sample = token_losses.sum(-1) / counts.clamp_min(1)
    active = counts > 0
    values = [float(per_sample[i].detach()) if eligible[i] else None for i in range(len(eligible))]
    return per_sample[active].mean(), values


def compute_auxiliary_losses(vlm, rgb, transcripts, packets, truths, args, *, c_prepared, c_outputs):
    totals, values, statuses = {}, {}, {}
    for field, (_, prefix) in AUX_FIELDS.items():
        eligible, status = eligible_auxiliary(packets, truths[field], field)
        statuses[field] = status
        values[field] = [None] * len(packets)
        if not any(eligible):
            totals[field] = None
            continue
        if hierarchy_model(vlm):
            name = 'Prompt A2' if field == 'gesture movement description' else 'Prompt B'
            targets = [{field: truths[field][i] if eligible[i] else packets[i].fields.get(field, '')}
                       for i in range(len(packets))]
            prepared = vlm._prepare_stage(name, stage_inputs(name, transcripts, packets),
                                           rgb if name == 'Prompt A2' else None, targets)
            check_sequence_length(prepared, args, name)
            outputs = vlm(**prepared.model_inputs)
        else:
            prepared, outputs = c_prepared, c_outputs
        masks = prepared.field_masks or {}
        if field not in masks:
            # A string-only legacy packet cannot create a trainable auxiliary
            # loss when the model exposes no corresponding teacher-forced logits.
            statuses[field] = ['missing_token_mask' if flag else reason for flag, reason in zip(eligible, status)]
            totals[field] = None
            continue
        totals[field], values[field] = field_ce(outputs.logits, prepared.labels, masks[field], eligible)
    return totals, values, statuses


def sample_c_losses(formatted, generated, missing_weight: float) -> list[dict[str, float]]:
    """Detached per-reference reporting of existing C terms, not extra training."""
    logits, labels = formatted.logits.detach()[:, :-1], formatted.labels[:, 1:]
    masks = {field: formatted.field_masks[field][:, 1:] & (labels != -100) for field in EXPECTED_FIELDS}
    supervised = torch.stack(list(masks.values())).any(0)
    losses = F.cross_entropy(logits[supervised].float(), labels[supervised], reduction='none')
    token_losses = losses.new_zeros(labels.shape).masked_scatter(supervised, losses)
    result = []
    for i, row in enumerate(generated):
        per_field = {field: (token_losses[i] * mask[i]).sum() / mask[i].sum().clamp_min(1)
                     for field, mask in masks.items()}
        ce = token_losses[i].sum() / supervised[i].sum().clamp_min(1)
        penalty = missing_weight * (len(row.missing_fields) + sum(per_field[field] for field in row.missing_fields))
        result.append({'intent_ce': float(per_field[EXPECTED_FIELDS[0]]),
                       'reasoning_ce': float(per_field[EXPECTED_FIELDS[1]]),
                       'c_response_loss': float(ce + penalty)})
    return result


def select_frames(selector, batch: dict, device):
    return selector(batch['landmarks'].to(device), frame_mask=batch['frame_mask'].to(device),
                    landmark_mask=batch['landmark_mask'].to(device), visibility=batch['visibility'].to(device),
                    rgb=batch['rgb'].to(device), frame_times=batch['frame_indices'].to(device))


def run_epoch(*, loader, selector, vlm, loss_calculator, args, device, dtype,
              epoch: int, split: str, optimizer=None, scaler=None, scheduler=None,
              predictions_path: Path | None = None,
              metric_accumulator: ValidationAccumulator | None = None,
              response_records: list | None = None) -> dict:
    training = optimizer is not None
    if training and metric_accumulator is not None:
        raise ValueError('Generation metrics are collected only in validation')
    selector.train(training)
    vlm.train(training)
    sums, aux_sums, aux_counts, examples, optimizer_steps = defaultdict(float), defaultdict(float), defaultdict(int), 0, 0
    generation_cache = {} if metric_accumulator is not None else None
    if training:
        optimizer.zero_grad(set_to_none=True)
    handle = predictions_path.open('w', encoding='utf-8') if predictions_path else None
    parameters = [p for module in (selector, vlm) for p in module.parameters() if p.requires_grad]
    try:
        for step, batch in enumerate(loader):
            with torch.set_grad_enabled(training):
                chosen = select_frames(selector, batch, device)
                targets = annotation_targets(batch['targets'])
                n = len(targets)
                truths = optional_targets(batch, n)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                    if generation_cache is None:
                        packets = generate_packets(vlm, chosen.selected_rgb.detach(), batch['transcript'], args)
                    else:
                        keys = list(zip(batch['media_id'], batch['transcript']))
                        missing = {}
                        for index, key in enumerate(keys):
                            if key not in generation_cache:
                                missing.setdefault(key, index)
                        if missing:
                            indices = list(missing.values())
                            fresh = generate_packets(vlm, chosen.selected_rgb[indices].detach(),
                                                     [batch['transcript'][i] for i in indices], args)
                            generation_cache.update(zip(missing, fresh))
                        packets = [generation_cache[key] for key in keys]
                    generated = [intent_output_formatter(packet.c_response) for packet in packets]
                    prepared = prepare_supervision(vlm, chosen.selected_rgb, batch['transcript'], targets, packets, truths, args)
                    outputs = vlm(**prepared.model_inputs)
                    formatted = format_token_output(outputs.logits, prepared.labels, prepared.field_masks)
                    window_start = (step // args.grad_accum_steps) * args.grad_accum_steps
                    window_size = min(args.grad_accum_steps, len(loader) - window_start) if training else 1
                    loss_terms = loss_calculator(response=formatted, generated=generated,
                                                 original_landmarks=batch['landmarks'].to(device), selection=chosen,
                                                 backward_divisor=window_size)
                    auxiliary, aux_values, aux_status = compute_auxiliary_losses(
                        vlm, chosen.selected_rgb, batch['transcript'], packets, truths, args,
                        c_prepared=prepared, c_outputs=outputs)
                    auxiliary_total = loss_terms['loss'].new_zeros(())
                    for field, (_, prefix) in AUX_FIELDS.items():
                        weight = getattr(args, prefix + '_loss_weight', 1.0)
                        if auxiliary[field] is not None:
                            auxiliary_total = auxiliary_total + weight * auxiliary[field]
                    loss_terms['c_response_loss'] = loss_terms['response_loss']
                    loss_terms['auxiliary_response_loss'] = auxiliary_total
                    loss_terms['response_loss'] = loss_terms['response_loss'] + auxiliary_total
                    loss_terms['loss'] = loss_terms['loss'] + auxiliary_total
                    loss_terms['backward_loss'] = loss_terms['loss'] / window_size
                if not torch.isfinite(loss_terms['loss']):
                    raise FloatingPointError(f'Nonfinite loss at {split}, batch {step + 1}')
                if metric_accumulator is not None:
                    metric_accumulator.update(media_ids=batch['media_id'], transcripts=batch['transcript'],
                                              predictions=generated, references=targets, reference_ids=batch['reference_id'])
                    if isinstance(metric_accumulator, FourFieldValidationAccumulator):
                        metric_accumulator.update_optional(media_ids=batch['media_id'], transcripts=batch['transcript'],
                                                           packets=packets, truths=truths,
                                                           reference_ids=batch['reference_id'])
                per_sample = sample_c_losses(formatted, generated, args.missing_field_weight)
                if training:
                    scaler.scale(loss_terms['backward_loss']).backward()
                    if (step + 1) % args.grad_accum_steps == 0 or step + 1 == len(loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= previous_scale:
                            scheduler.step()
                            optimizer_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                metrics = {key: value for key, value in loss_terms.items() if key != 'backward_loss'}
                for key, value in metrics.items():
                    sums[key] += float(value.detach()) * n
                sums['valid_json_rate'] += sum(row.valid_json for row in generated)
                sums['complete_fields_rate'] += sum(not row.missing_fields for row in generated)
                for field, (_, prefix) in AUX_FIELDS.items():
                    for value in aux_values[field]:
                        if value is not None:
                            aux_sums[prefix] += value
                            aux_counts[prefix] += 1
                    for status in ('missing_response', 'missing_ground_truth', 'missing_token_mask'):
                        aux_counts[prefix + '_' + status] += aux_status[field].count(status)
                for i, row in enumerate(generated):
                    for field, label in zip(EXPECTED_FIELDS, ('intent_exact_match', 'reasoning_exact_match')):
                        sums[label] += int(row.fields[field] == targets[i][field])
                    losses = per_sample[i]
                    pred, gt = dict(row.fields), dict(targets[i])
                    losses['response_loss'] = losses['c_response_loss']
                    losses['auxiliary_response_loss'] = 0.0
                    statuses = {}
                    for field, (_, prefix) in AUX_FIELDS.items():
                        pred[field] = packets[i].fields.get(field)
                        gt[field] = truths[field][i]
                        value = aux_values[field][i]
                        losses[prefix + '_ce'] = value
                        statuses[field] = aux_status[field][i]
                        if value is not None:
                            weighted = getattr(args, prefix + '_loss_weight', 1.0) * value
                            losses['auxiliary_response_loss'] += weighted
                            losses['response_loss'] += weighted
                    common_record = {
                        'pred_response': pred, 'gt_response': gt, 'losses': losses,
                        'auxiliary_loss_status': statuses,
                        'prompt_responses': packets[i].prompts,
                        'prompt_response_diagnostics': packets[i].diagnostics,
                        'raw_response': packets[i].raw,
                    }
                    if handle:
                        record = {'epoch': epoch, 'split': split, 'reference_id': batch['reference_id'][i],
                                  'clip_id': batch['clip_id'][i], 'annotator_id': batch['annotator_id'][i],
                                  'selected_frame_indices': chosen.selected_times[i].detach().cpu().tolist(),
                                  'annotation': gt, 'output': pred,
                                  'missing_fields': row.missing_fields, 'valid_json': row.valid_json,
                                  'raw_response': packets[i].raw, **common_record}
                        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    if response_records is not None:
                        response_records.append({'reference_id': batch['reference_id'][i],
                                                 'fiename': video_filename(batch['video_id'][i]),
                                                 'gest_idx': gesture_index(batch['media_id'][i]), **common_record})
                examples += n
                if (step + 1) % args.log_every == 0 or step + 1 == len(loader):
                    print(f'epoch={epoch} {split} batch={step+1}/{len(loader)} '
                          f'response_loss={float(loss_terms["response_loss"].detach()):.4f} '
                          f'auxiliary_loss={float(auxiliary_total.detach()):.4f} '
                          f'missing={float(loss_terms["missing_count"].detach()):.2f}', flush=True)
                del outputs, prepared, formatted, chosen, loss_terms, metrics, auxiliary, auxiliary_total
    finally:
        if handle:
            handle.close()
    if not examples:
        raise ValueError(f'Empty {split} loader')
    result = {**{key: value / examples for key, value in sums.items()},
              'examples': examples, 'optimizer_steps': optimizer_steps}
    # Always emit the same keys, even when a field is absent for an entire epoch,
    # so metrics.append_epoch_metrics retains its fixed CSV schema.
    for _, (_, prefix) in AUX_FIELDS.items():
        count = aux_counts[prefix]
        result[prefix + '_ce'] = aux_sums[prefix] / count if count else None
        result[prefix + '_supervised_examples'] = count
        result[prefix + '_coverage'] = count / examples
        for status in ('missing_response', 'missing_ground_truth', 'missing_token_mask'):
            result[prefix + '_' + status] = aux_counts[prefix + '_' + status]
    return result


def save_checkpoint(path: Path, selector, vlm, optimizer, scheduler, scaler,
                    epoch: int, metrics: dict, args) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vlm.model.save_pretrained(path / 'qwen', safe_serialization=True)
    vlm.processor.save_pretrained(path / 'qwen')
    torch.save({'selector': selector.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'metrics': metrics, 'arguments': vars(args),
                'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path / 'training_state.pt')
    write_json(path / 'config.json', {'epoch': epoch, 'metrics': metrics,
                                      'output_fields': PROMPT_FIELDS if hierarchy_model(vlm) else EXPECTED_FIELDS,
                                      'supervised_optional_fields': list(AUX_FIELDS), 'arguments': vars(args)})


def save_validation_responses(output_dir: Path, phase: str, records) -> Path:
    path = output_dir / f'{phase}_val_responses.json'
    write_json(path, records)
    # Preserve filenames expected by scripts using the original trainer.
    write_json(output_dir / f'{phase}_val_set_responses.json', records)
    return path


def train(args, *, injected_model=None, injected_processor=None, injected_metric_engine=None) -> dict:
    if args.output_dir is None:
        args.output_dir = timestamped_run_dir()
    args.output_dir = resolve_from_root(args.output_dir)
    print(f'Run directory: {args.output_dir}', flush=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a new/empty --output-dir: {args.output_dir}')
    metrics_csv = (resolve_from_root(args.metrics_csv) if args.metrics_csv is not None
                   else args.output_dir / DEFAULT_METRICS_CSV.name)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.write_text('', encoding='utf-8')
    print(f'Cleared {metrics_csv}', flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    precision = args.precision
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type == 'cuda' else 'fp32'
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    if device.type != 'cuda' and dtype != torch.float32:
        raise ValueError('Use --precision fp32 for CPU training')
    metric_engine = injected_metric_engine
    if metric_engine is None:
        print('Initializing validation metrics (BERTScore and official BLEURT)...', flush=True)
        metric_engine = GenerationMetrics(MetricsConfig(
            bertscore_model=args.bertscore_model, bertscore_num_layers=args.bertscore_num_layers,
            device=args.metrics_device, batch_size=args.metrics_batch_size,
            bleurt_checkpoint=str(resolve_from_root(args.bleurt_checkpoint)),
            bleurt_python=args.bleurt_python, bleurt_timeout=args.bleurt_timeout))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'run_arguments.json', vars(args))
    write_json(args.output_dir / 'metrics_config.json', metric_engine.metadata)
    manifest_path = make_manifest(args)
    manifest_path, splits = prepare_splits(manifest_path, args.output_dir, args)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, num_frames=None,
                  image_size=(args.image_size, args.image_size), pin_memory=device.type == 'cuda')
    train_loader = create_dataloader(manifest_path, media_ids=splits['train'], shuffle=True, **common)
    train_eval_loader = create_dataloader(manifest_path, media_ids=splits['train'], shuffle=False, **common)
    val_loader = create_dataloader(manifest_path, media_ids=splits['val'], shuffle=False, **common)
    architecture = load_model_module(resolve_from_root(args.model_file))
    local_model = resolve_from_root(Path(args.model_name))
    model_name = str(local_model) if local_model.exists() else args.model_name
    selector = architecture.TemporalFrameSelector(num_joints=args.num_joints, num_frames=args.num_frames,
                                                  temperature=args.temperature,
                                                  visibility_threshold=args.visibility_threshold).to(device)
    multi_prompt = hasattr(architecture, 'PROMPT_ORDER') and hasattr(architecture.QwenIntentModel, '_prepare_stage')
    options = {} if multi_prompt else {'output_fields': EXPECTED_FIELDS}
    if getattr(args, 'prompts_file', None) is not None:
        if not multi_prompt:
            raise ValueError('--prompts-file requires the multi-prompt model')
        options['prompts_path'] = resolve_from_root(args.prompts_file)
    vlm = architecture.QwenIntentModel(model_name=model_name, training_mode=args.training_mode,
                                        device=device, dtype=dtype, gradient_checkpointing=not args.no_gradient_checkpointing,
                                        num_frames=args.num_frames, model=injected_model, processor=injected_processor,
                                        local_files_only=args.local_files_only, **options)
    groups = [{'params': list(selector.parameters()), 'lr': args.selector_lr}]
    trainable_qwen = [p for p in vlm.parameters() if p.requires_grad]
    if trainable_qwen:
        groups.append({'params': trainable_qwen, 'lr': args.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup = int(steps * args.warmup_ratio)
    def lr_multiplier(step):
        return step / max(1, warmup) if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and dtype == torch.float16)
    loss_calculator = CombinedTrainingLoss(missing_field_weight=args.missing_field_weight,
                                           selection_loss_weight=args.selection_loss_weight,
                                           selection_alpha=args.selection_alpha, selection_beta=args.selection_beta)
    best, history = math.inf, []
    history_path = args.output_dir / 'metrics.jsonl'
    print('Collecting validation responses from the untrained model...', flush=True)
    first_accumulator, first_responses = FourFieldValidationAccumulator(), []
    run_epoch(loader=val_loader, split='untrained_val', metric_accumulator=first_accumulator,
              response_records=first_responses, selector=selector, vlm=vlm,
              loss_calculator=loss_calculator, args=args, device=device, dtype=dtype, epoch=0)
    print('Scoring untrained validation responses...', flush=True)
    first_metrics, first_reference_metrics = compute_all_generation_metrics(metric_engine, first_accumulator)
    print_all_validation_metrics(first_metrics, 0)
    attach_sample_metrics(first_responses, first_reference_metrics)
    first_path = save_validation_responses(args.output_dir, 'first', first_responses)
    print(f'Saved {len(first_responses)} untrained validation responses to {first_path}', flush=True)
    for epoch in range(1, args.epochs + 1):
        shared = dict(selector=selector, vlm=vlm, loss_calculator=loss_calculator,
                      args=args, device=device, dtype=dtype, epoch=epoch)
        train_metrics = run_epoch(loader=train_loader, split='train', optimizer=optimizer, scaler=scaler,
                                  scheduler=scheduler, predictions_path=args.output_dir / f'epoch_{epoch:04d}_train.jsonl', **shared)
        train_eval = run_epoch(loader=train_eval_loader, split='train_eval',
                               predictions_path=args.output_dir / f'epoch_{epoch:04d}_train_eval.jsonl', **shared)
        val_accumulator = FourFieldValidationAccumulator()
        final_responses = [] if epoch == args.epochs else None
        val_metrics = run_epoch(loader=val_loader, split='val', metric_accumulator=val_accumulator,
                                predictions_path=args.output_dir / f'epoch_{epoch:04d}_val.jsonl',
                                response_records=final_responses, **shared)
        generation_metrics, reference_metrics = compute_all_generation_metrics(metric_engine, val_accumulator)
        print_all_validation_metrics(generation_metrics, epoch)
        if final_responses is not None:
            attach_sample_metrics(final_responses, reference_metrics)
            responses_path = save_validation_responses(args.output_dir, 'final', final_responses)
            print(f'Saved {len(final_responses)} validation responses to {responses_path}', flush=True)
        val_metrics.update(generation_metrics)
        append_epoch_metrics(metrics_csv, epoch=epoch, run_id=str(args.output_dir), train_metrics=train_metrics,
                              train_eval_metrics=train_eval, val_metrics=val_metrics, metadata=metric_engine.metadata)
        print(f'Epoch metrics saved to {metrics_csv}', flush=True)
        record = {'epoch': epoch, 'train': train_metrics, 'train_eval': train_eval, 'val': val_metrics}
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        history.append(record)
        save_checkpoint(args.output_dir / 'last', selector, vlm, optimizer, scheduler, scaler, epoch, record, args)
        if val_metrics['response_loss'] < best:
            best = val_metrics['response_loss']
            save_checkpoint(args.output_dir / 'best', selector, vlm, optimizer, scheduler, scaler, epoch, record, args)
        print(f'Finished epoch {epoch}: val response_loss={val_metrics["response_loss"]:.4f}; best={best:.4f}', flush=True)
    return {'best_val_response_loss': best, 'history': history}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path(DATASET_DIR))
    parser.add_argument('--rgb-dir', type=Path, default=Path(RGB_DIR))
    parser.add_argument('--landmarks-dir', type=Path, default=Path(LANDMARKS_DIR))
    parser.add_argument('--column-map', type=Path)
    parser.add_argument('--gesture-description-column', help='Exact CSV header; default movement (or column-map movement_description)')
    parser.add_argument('--gesture-meaning-column', help='Exact CSV header; default meaning (or column-map gesture_meaning)')
    parser.add_argument('--manifest', type=Path, help='Reuse an existing manifest instead of preprocessing')
    parser.add_argument('--output-dir', type=Path, default=None, help='Default: runs/{timestamp} under the project root')
    parser.add_argument('--model-file', type=Path, default=Path('model/p_A1_A2_B_C.py'))
    parser.add_argument('--prompts-file', type=Path, help='Optional A1_A2_B_C.json for the multi-prompt model')
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--training-mode', choices=('lora',), default='lora')
    parser.add_argument('--device')
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp16', 'fp32'), default='auto')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--selector-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--warmup-ratio', type=float, default=.03)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--split-group', choices=('auto', 'participant_id', 'video_id'), default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-joints', type=int, default=33)
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--visibility-threshold', type=float, default=0.)
    parser.add_argument('--selection-loss-weight', type=float, default=1.)
    parser.add_argument('--selection-alpha', type=float, default=1.)
    parser.add_argument('--selection-beta', type=float, default=1.)
    parser.add_argument('--missing-field-weight', type=float, default=1.)
    parser.add_argument('--gesture-description-loss-weight', type=float, default=1.)
    parser.add_argument('--gesture-meaning-loss-weight', type=float, default=1.)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--max-sequence-length', type=int, default=8192)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--short-clip-policy', choices=('error', 'skip'), default='error')
    parser.add_argument('--skip-invalid', action='store_true')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--metrics-csv', type=Path, default=None,
                        help=f'Default: {{output-dir}}/{DEFAULT_METRICS_CSV.name}')
    parser.add_argument('--bertscore-model', default='roberta-large')
    parser.add_argument('--bertscore-num-layers', type=int)
    parser.add_argument('--metrics-device', default='cpu', help='BERTScore device; BLEURT uses a separate CPU worker')
    parser.add_argument('--metrics-batch-size', type=int, default=8)
    parser.add_argument('--bleurt-checkpoint', type=Path, default=Path('checkpoints/BLEURT-20'))
    parser.add_argument('--bleurt-python', default=sys.executable)
    parser.add_argument('--bleurt-timeout', type=float, default=1800.)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    positive = ('epochs', 'batch_size', 'grad_accum_steps', 'image_size', 'num_frames', 'num_joints',
                'max_new_tokens', 'max_sequence_length', 'log_every', 'learning_rate', 'selector_lr',
                'grad_clip', 'temperature', 'metrics_batch_size', 'bleurt_timeout')
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        parser.error('Sizes, learning rates, temperature and gradient clip must be positive')
    if not 0 < args.val_fraction < 1 or not 0 <= args.warmup_ratio < 1 or args.num_workers < 0:
        parser.error('Invalid fractions or worker count')
    nonnegative = (args.weight_decay, args.selection_loss_weight, args.selection_alpha, args.selection_beta,
                   args.missing_field_weight, args.gesture_description_loss_weight, args.gesture_meaning_loss_weight)
    if not 0 <= args.visibility_threshold < 1 or any(not math.isfinite(v) or v < 0 for v in nonnegative):
        parser.error('Loss weights/decay must be nonnegative and visibility threshold in [0,1)')
    try:
        train(args)
    except (ValueError, OSError, FloatingPointError) as exc:
        raise SystemExit(f'Training stopped: {exc}') from exc


if __name__ == '__main__':
    main()
