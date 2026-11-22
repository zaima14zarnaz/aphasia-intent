"""PyTorch dataset for multiple-reference aphasia intent generation.

No tokenization, model-specific chat template, or loss is imposed here.
Each item contains one coherent annotation reference and its aligned media.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class IntentDataset(Dataset):
    """Load RGB PNGs, XYZ landmarks, transcript, and four natural-language targets.

    num_frames=None loads all frames. Otherwise, take up to num_frames uniformly
    spaced frames, using identical indices for RGB and landmarks. Short clips
    are padded by collate_intent, with frame_mask distinguishing padding.

    image_size is (height, width). Default RGB output is float32 in [0,1].
    An optional transform receives the entire [T,3,H,W] tensor and must preserve
    temporal order, length, and geometry unless it also updates landmarks.
    Use photometric transforms only here; geometric augmentation needs a joint
    image/landmark transform. Model-specific normalization can be applied there.
    """

    def __init__(self, manifest_path: str | Path, num_frames: int | None = 16,
                 image_size: tuple[int, int] = (224, 224),
                 transform: Callable | None = None, media_ids: Iterable[str] | None = None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema")
        if num_frames is not None and (type(num_frames) is not int or num_frames <= 0):
            raise ValueError("num_frames must be a positive integer or None")
        if len(image_size) != 2 or any(type(n) is not int or n <= 0 for n in image_size):
            raise ValueError("image_size must be (positive height, positive width)")
        selected = None if media_ids is None else set(media_ids)
        if selected is not None and selected - set(manifest["clips"]):
            raise ValueError(f"Unknown media_ids: {sorted(selected - set(manifest['clips']))}")
        self.samples = [s for s in manifest["samples"] if selected is None or s["media_id"] in selected]
        self.clips = manifest["clips"]
        self.target_fields = manifest["target_fields"]
        self.num_frames, self.image_size, self.transform = num_frames, image_size, transform
        if not self.samples:
            raise ValueError("Dataset selection has no annotations")

    def __len__(self) -> int:
        # Three annotation rows for one clip create three separate items.
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        clip = self.clips[sample["media_id"]]
        length = clip["num_frames"]
        count = length if self.num_frames is None else min(length, self.num_frames)
        positions = np.linspace(0, length - 1, count).round().astype(np.int64)
        frame_paths = [clip["frame_paths"][int(i)] for i in positions]
        height, width = self.image_size
        images = []
        for path in frame_paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                pixels = np.array(image, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(pixels).permute(2, 0, 1))
        rgb = torch.stack(images)
        if self.transform is not None:
            rgb = self.transform(rgb)
            if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or rgb.shape[:2] != (count, 3):
                raise ValueError("transform must return a [T,3,H,W] tensor preserving T")
        with np.load(self.manifest_path.parent / clip["landmarks_cache"], allow_pickle=False) as cache:
            landmarks = torch.from_numpy(cache["xyz"][positions].copy())
            visibility = torch.from_numpy(cache["visibility"][positions].copy())
            landmark_mask = torch.from_numpy(cache["valid"][positions].copy())
        if landmarks.shape != (count, clip["num_joints"], 3):
            raise ValueError(f"Unexpected cache shape for {sample['media_id']}")
        return {
            "rgb": rgb, "landmarks": landmarks, "visibility": visibility,
            "landmark_mask": landmark_mask,
            "frame_indices": torch.tensor([clip["frame_indices"][int(i)] for i in positions], dtype=torch.long),
            "frame_paths": frame_paths,
            "transcript": sample["transcript"], "targets": dict(sample["targets"]),
            "reference_id": sample["reference_id"], "clip_id": sample["clip_id"],
            "media_id": sample["media_id"], "video_id": clip["video_id"],
            "participant_id": sample["participant_id"], "annotator_id": sample["annotator_id"],
            "source": sample["source"],
        }


def collate_intent(items: list[dict]) -> dict:
    """Pad time only; text remains Python strings for the model's tokenizer.

    frame_mask True = actual selected frame. landmark_mask True = coordinate
    present in JSON, independent of visibility. Padding indices are -1.
    """
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    lengths = torch.tensor([item["rgb"].shape[0] for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    first = items[0]
    rgb = first["rgb"].new_zeros((batch_size, max_length, *first["rgb"].shape[1:]))
    landmarks = first["landmarks"].new_zeros((batch_size, max_length, *first["landmarks"].shape[1:]))
    visibility = first["visibility"].new_zeros((batch_size, max_length, *first["visibility"].shape[1:]))
    landmark_mask = torch.zeros(visibility.shape, dtype=torch.bool)
    frame_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    frame_indices = torch.full((batch_size, max_length), -1, dtype=torch.long)
    for b, item in enumerate(items):
        t = int(lengths[b])
        if item["rgb"].shape[1:] != first["rgb"].shape[1:] or item["landmarks"].shape[1:] != first["landmarks"].shape[1:]:
            raise ValueError("All items in a batch must have the same image size and joint count")
        rgb[b, :t] = item["rgb"]
        landmarks[b, :t] = item["landmarks"]
        visibility[b, :t] = item["visibility"]
        landmark_mask[b, :t] = item["landmark_mask"]
        frame_mask[b, :t] = True
        frame_indices[b, :t] = item["frame_indices"]
    batch = {"rgb": rgb, "landmarks": landmarks, "visibility": visibility,
             "landmark_mask": landmark_mask, "frame_mask": frame_mask,
             "frame_indices": frame_indices, "lengths": lengths,
             "transcript": [item["transcript"] for item in items],
             "targets": {field: [item["targets"][field] for item in items] for field in first["targets"]}}
    for field in ("reference_id", "clip_id", "media_id", "video_id", "participant_id", "annotator_id", "frame_paths", "source"):
        batch[field] = [item[field] for item in items]
    return batch


def split_media_ids(manifest_path: str | Path, val_fraction: float = 0.15,
                    test_fraction: float = 0.15, seed: int = 42,
                    group_by: str = "participant_id") -> dict[str, list[str]]:
    """Group-disjoint splits before expanding annotation references.

    Requires explicit participant IDs by default. video_id is an opt-in fallback
    that prevents source-video leakage, but cannot guarantee unseen participants.
    Fractions apply to group counts, not reference/clip counts; rounding may
    produce an empty validation/test split for very small datasets.
    """
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Fractions must be nonnegative and sum to less than 1")
    if group_by not in ("participant_id", "video_id"):
        raise ValueError("group_by must be participant_id or video_id")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    groups, media_groups = defaultdict(set), {}
    for sample in manifest["samples"]:
        media = sample["media_id"]
        group = sample.get("participant_id") if group_by == "participant_id" else manifest["clips"][media]["video_id"]
        if not group:
            raise ValueError("Participant IDs missing. Supply them in the CSV, or explicitly use group_by='video_id'.")
        if media in media_groups and media_groups[media] != group:
            raise ValueError(f"Conflicting group IDs for {media}")
        media_groups[media] = group
        groups[group].add(media)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test, n_val = round(len(keys) * test_fraction), round(len(keys) * val_fraction)
    if n_test + n_val >= len(keys):
        raise ValueError("Too few groups for these rounded split sizes; reduce fractions")
    partitions = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
    return {split: sorted(media for key in keys_in_split for media in groups[key]) for split, keys_in_split in partitions.items()}


def create_dataloader(manifest_path: str | Path, batch_size: int = 4,
                      shuffle: bool = True, num_workers: int = 0,
                      pin_memory: bool = False, **dataset_kwargs) -> DataLoader:
    """Shuffling visits every annotation reference once per epoch."""
    dataset = IntentDataset(manifest_path, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_intent,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0)
