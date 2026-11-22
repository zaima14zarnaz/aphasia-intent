#!/usr/bin/env python3
"""Prepare long-format annotation CSVs and aligned PNG/MediaPipe sequences.

One CSV row is one annotation reference. By default all four targets are kept
together; target_fields restricts which annotation outputs are required and stored.
This script does not merge, adjudicate, or automatically accept/reject meanings.
Requires Python >=3.10 and NumPy. See README.md for columns and usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

DATASET_DIR = "/data/quantization/zaima/aphasiabank/neural2/control/annotations/intent/annotations"
RGB_DIR = "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments"
LANDMARKS_DIR = "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments_landmarks/mediapipe"
TARGET_FIELDS = ("movement_description", "gesture_meaning", "communicative_intent", "reasoning")
ALIASES = {
    "clip_id": ("clip_id", "clip", "clip_filename"),
    "transcript": ("transcript", "transcription", "clip_transcript", "aligned_transcript"),
    "movement_description": ("movement_description", "gesture_movement_description", "gesture_description", "movement", "description"),
    "gesture_meaning": ("gesture_meaning", "meaning"),
    "communicative_intent": ("communicative_intent", "intent"),
    "reasoning": ("reasoning", "rationale", "explanation"),
    "annotator_id": ("annotator_id", "annotator", "annotator_email", "email", "worker_id"),
    "participant_id": ("participant_id", "speaker_id", "subject_id"),
}


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_columns(headers: list[str], mapping: dict[str, str],
                    required: tuple[str, ...] | None = None) -> dict[str, str]:
    """Reject ambiguous headers instead of silently choosing a target column.

    required selects the canonical keys to resolve. annotator_id and
    participant_id stay optional. The default resolves every alias key.
    """
    normalized: dict[str, list[str]] = {}
    for header in headers:
        normalized.setdefault(normalize_header(header), []).append(header)
    keys = tuple(ALIASES) if required is None else required
    unknown = [key for key in keys if key not in ALIASES]
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}. Keys: {list(ALIASES)}")
    optional = {"annotator_id", "participant_id"}
    result = {}
    for key in keys:
        aliases = ALIASES[key]
        if key in mapping:
            if mapping[key] not in headers:
                raise ValueError(f"Mapped column {mapping[key]!r} for {key} is absent. Headers: {headers}")
            result[key] = mapping[key]
            continue
        matches = [h for alias in aliases for h in normalized.get(alias, [])]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous columns for {key}: {matches}; use --column-map")
        if matches:
            result[key] = matches[0]
        elif key not in optional:
            raise ValueError(f"Missing column for {key}; use --column-map. Headers: {headers}")
    return result


def parse_clip_id(value: str) -> tuple[str, str, str, str]:
    """Parse from the right, allowing underscores in the source video name."""
    match = re.fullmatch(r"(.+)_([0-9]+)_([^/\\]+?)\s*\.\s*mp4", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid clip_id {value!r}; expected video_name_13_REF.mp4")
    video, index, category = match.groups()
    video = video.strip()
    category = category.strip()
    if video in ("", ".", "..") or "/" in video or "\\" in video or not category:
        raise ValueError(f"Unsafe or empty video/category in {value!r}")
    return video, index, category, f"{video}_{index}_{category}.mp4"


def load_aligned_media(rgb_segment: Path, landmark_path: Path) -> tuple[list[str], list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return matching PNGs, source indices, XYZ, visibility, coordinate validity.

    Null frames/joints have zero coordinates, zero visibility, and a False mask.
    Missing PNG files or malformed landmarks are errors, never positional joins.
    """
    with landmark_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {landmark_path}")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {landmark_path}")
    if any(not isinstance(frame, dict) for frame in frames):
        raise ValueError(f"Each frames entry must be an object: {landmark_path}")
    indices = [f.get("frame_index") for f in frames]
    if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"frame_index must be unique nonnegative integers: {landmark_path}")
    frames = sorted(frames, key=lambda f: f["frame_index"])
    if "num_frames" in data and data["num_frames"] != len(frames):
        raise ValueError(f"num_frames does not match frames in {landmark_path}")
    shape = data.get("landmark_shape")
    if shape is not None and (not isinstance(shape, list) or len(shape) != 3 or
                              any(type(v) is not int or v <= 0 for v in shape) or
                              shape[0] != len(frames)):
        raise ValueError(f"Invalid landmark_shape in {landmark_path}")
    first = next((f["landmarks"] for f in frames if f.get("landmarks") is not None), None)
    joints = shape[1] if shape else (len(first) if isinstance(first, list) else 0)
    if joints <= 0:
        raise ValueError(f"Cannot determine joint count: {landmark_path}")
    fmt = data.get("landmark_format", ["x", "y", "z", "visibility"])
    if not isinstance(fmt, list) or any(not isinstance(c, str) for c in fmt) or len(set(fmt)) != len(fmt) or not {"x", "y", "z"}.issubset(fmt):
        raise ValueError(f"landmark_format must contain x, y, z: {landmark_path}")
    if shape and shape[2] != len(fmt):
        raise ValueError(f"landmark_shape and landmark_format disagree: {landmark_path}")
    xyz_columns = [fmt.index(c) for c in ("x", "y", "z")]
    vis_column = fmt.index("visibility") if "visibility" in fmt else None
    xyz = np.zeros((len(frames), joints, 3), dtype=np.float32)
    visibility = np.zeros((len(frames), joints), dtype=np.float32)
    valid = np.zeros((len(frames), joints), dtype=np.bool_)
    paths, frame_indices, seen = [], [], set()
    for t, frame in enumerate(frames):
        filename = frame.get("frame_file")
        if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename or Path(filename).suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG basename in frame_file: {filename!r}")
        if filename in seen:
            raise ValueError(f"Duplicate frame_file: {filename}")
        seen.add(filename)
        path = rgb_segment / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path.resolve()))
        frame_indices.append(frame["frame_index"])
        landmarks = frame.get("landmarks")
        if landmarks is None:
            continue
        if not isinstance(landmarks, list) or len(landmarks) != joints:
            raise ValueError(f"Wrong joint count at {landmark_path}, frame {t}")
        for j, point in enumerate(landmarks):
            if point is None:
                continue
            point = np.asarray(point, dtype=np.float32)
            if point.shape != (len(fmt),) or not np.isfinite(point).all():
                raise ValueError(f"Invalid landmark at {landmark_path}, frame {t}, joint {j}")
            xyz[t, j] = point[xyz_columns]
            confidence = float(point[vis_column]) if vis_column is not None else 1.0
            if not 0 <= confidence <= 1:
                raise ValueError(f"Visibility outside [0, 1] at frame {t}, joint {j}")
            visibility[t, j] = confidence
            valid[t, j] = True
    # PNGs absent from the JSON cannot be aligned reliably; report rather than drop.
    extra = {p.name for p in rgb_segment.iterdir() if p.is_file() and p.suffix.lower() == ".png"} - seen
    if extra:
        raise ValueError(f"{len(extra)} PNG(s) lack landmark entries in {rgb_segment}: {sorted(extra)[:5]}")
    return paths, frame_indices, xyz, visibility, valid


def preprocess(dataset_dir: Path, rgb_dir: Path, landmarks_dir: Path, output_dir: Path,
               column_map: dict[str, str] | None = None, skip_invalid: bool = False,
               target_fields: tuple[str, ...] | list[str] | None = None) -> Path:
    dataset_dir, rgb_dir, landmarks_dir, output_dir = (
        p.expanduser().resolve() for p in (dataset_dir, rgb_dir, landmarks_dir, output_dir))
    csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files under {dataset_dir}")
    mapping = column_map or {}
    if set(mapping) - set(ALIASES) or any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError(f"column-map must map canonical keys to header strings. Keys: {list(ALIASES)}")
    selected = tuple(TARGET_FIELDS if target_fields is None else target_fields)
    if (not selected or len(set(selected)) != len(selected)
            or any(field not in TARGET_FIELDS for field in selected)):
        raise ValueError(f"target_fields must be a nonempty subset of {list(TARGET_FIELDS)}")
    required = ("clip_id", "transcript", *selected, "annotator_id", "participant_id")
    # A new output directory avoids stale manifests/caches after a failed rebuild.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    cache = output_dir / "landmarks"
    cache.mkdir(parents=True, exist_ok=True)
    clips, samples, rejected, column_report, failed_media = {}, [], [], {}, {}
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Missing header: {csv_path}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Duplicate CSV headers: {csv_path}")
            columns = resolve_columns(reader.fieldnames, mapping, required)
            column_report[str(csv_path)] = columns
            # row_number is a logical CSV record number; quoted newlines are supported.
            for row_number, row in enumerate(reader, start=2):
                if not any(str(v).strip() for v in row.values() if v is not None):
                    continue
                try:
                    if None in row or any(row.get(c) is None for c in columns.values()):
                        raise ValueError("CSV row has an incorrect number of cells")
                    values = {k: row[v].strip() for k, v in columns.items()}
                    if any(not values[field] for field in ("clip_id", *selected)):
                        raise ValueError(f"clip_id and {', '.join(selected)} must be nonempty")
                    video, index, category, canonical = parse_clip_id(values["clip_id"])
                    media_id = f"{video}.mp4/gest_{index}"
                    if media_id in failed_media:
                        raise ValueError(failed_media[media_id])
                    if media_id not in clips:
                        try:
                            paths, indices, xyz, vis, valid = load_aligned_media(
                                rgb_dir / media_id, landmarks_dir / media_id / "landmarks.json")
                        except (ValueError, TypeError, KeyError, OSError) as exc:
                            failed_media[media_id] = str(exc)
                            raise
                        digest = hashlib.sha256(media_id.encode()).hexdigest()
                        relative_cache = f"landmarks/{digest}.npz"
                        np.savez_compressed(output_dir / relative_cache, xyz=xyz, visibility=vis, valid=valid)
                        clips[media_id] = {
                            "video_id": video, "gesture_index": int(index),
                            "frame_paths": paths, "frame_indices": indices,
                            "landmarks_cache": relative_cache, "num_frames": len(paths),
                            "num_joints": xyz.shape[1],
                        }
                    # Same media may have different transcript text per annotation.
                    # Preserve the complete row instead of mixing annotators' fields.
                    source = {"csv_file": str(csv_path), "row_number": row_number}
                    reference_id = hashlib.sha256(f"{csv_path.relative_to(dataset_dir)}:{row_number}".encode()).hexdigest()
                    samples.append({
                        "reference_id": reference_id, "clip_id": canonical,
                        "original_clip_id": values["clip_id"], "media_id": media_id,
                        "gesture_category": category,
                        "annotator_id": values.get("annotator_id") or csv_path.stem,
                        "annotator_id_source": "column" if values.get("annotator_id") else "csv_filename",
                        "participant_id": values.get("participant_id", ""),
                        "transcript": values["transcript"],
                        "targets": {field: values[field] for field in selected},
                        "source": source,
                    })
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    error = {"csv_file": str(csv_path), "row_number": row_number, "error": str(exc)}
                    if not skip_invalid:
                        raise ValueError(f"{csv_path}, record {row_number}: {exc}") from exc
                    rejected.append(error)
    summary = {
        "csv_files": len(csv_paths), "unique_media": len(clips),
        "annotation_references": len(samples), "rejected_rows": len(rejected),
        "columns": column_report, "rejections": rejected,
    }
    (output_dir / "preprocess_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not samples:
        raise ValueError(f"No valid annotation references; see {output_dir / 'preprocess_report.json'}")
    manifest = {"schema_version": 1, "target_fields": list(selected), "clips": clips, "samples": samples}
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(samples)} annotation references across {len(clips)} media sequences. Rejected: {len(rejected)}")
    print(f"Manifest: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--rgb-dir", type=Path, default=Path(RGB_DIR))
    parser.add_argument("--landmarks-dir", type=Path, default=Path(LANDMARKS_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path("intent_preprocessed"))
    parser.add_argument("--column-map", type=Path, help="JSON mapping canonical field names to exact CSV headers")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid rows and record every rejection")
    args = parser.parse_args()
    mapping = json.loads(args.column_map.read_text(encoding="utf-8")) if args.column_map else None
    if mapping is not None and not isinstance(mapping, dict):
        parser.error("--column-map must contain a JSON object")
    try:
        preprocess(args.dataset_dir, args.rgb_dir, args.landmarks_dir, args.output_dir, mapping, args.skip_invalid)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
