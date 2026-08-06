"""
MediaPipe body pose landmark extraction for gesture video frames.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
import urllib.request

import numpy as np

# Suppress MediaPipe / glog / absl C++ chatter on import and inference.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

if TYPE_CHECKING:
    import torch

try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

try:
    import mediapipe as mp
except ImportError as exc:
    mp = None  # type: ignore[assignment]
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None

NUM_POSE_LANDMARKS = 33

_MODEL_DIR = Path(__file__).resolve().parent / "mediapipe_models"
_MODEL_SPECS: dict[int, tuple[str, str]] = {
    0: (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    1: (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    2: (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    ),
}


def _require_mediapipe() -> Any:
    if mp is None:
        raise ImportError(
            "mediapipe is required for pose extraction. "
            "Install it with: pip install mediapipe"
        ) from _MEDIAPIPE_IMPORT_ERROR
    return mp


@contextmanager
def _suppress_native_logs():
    """
    Silence native MediaPipe/glog lines that bypass Python logging.

    Examples:
        Feedback manager requires a model with a single signature inference...
        Successfully initialized EGL...
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def _frame_ndim(frames: np.ndarray | torch.Tensor) -> int:
    if isinstance(frames, np.ndarray):
        return frames.ndim
    return int(frames.ndim)


def _resolve_pose_model_path(model_complexity: int) -> Path:
    if model_complexity not in _MODEL_SPECS:
        raise ValueError(
            f"model_complexity must be 0, 1, or 2, got {model_complexity}"
        )

    filename, url = _MODEL_SPECS[model_complexity]
    model_path = _MODEL_DIR / filename
    if model_path.is_file():
        return model_path

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, model_path)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to download MediaPipe pose model to {model_path}. "
            f"Download it manually from:\n{url}"
        ) from exc

    return model_path


def _normalize_frames(frames: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert frames to a contiguous uint8 RGB batch with shape [T, H, W, 3].
    """
    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise TypeError(
                "frames must be a numpy.ndarray or torch.Tensor"
            ) from exc

        if not isinstance(frames, torch_module.Tensor):
            raise TypeError("frames must be a numpy.ndarray or torch.Tensor")

        arr = frames.detach().cpu().numpy()

    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]

    if arr.ndim == 5:
        raise ValueError(
            "Expected a single video with shape [T, H, W, C] or [T, C, H, W]. "
            f"For batched videos [B, T, C, H, W], pass them to "
            f"extract_pose_landmarks directly. Got {arr.shape}"
        )

    if arr.ndim != 4:
        raise ValueError(
            "Expected frames shaped [T, H, W, C], [T, C, H, W], "
            f"[H, W, C], or [C, H, W], got {arr.shape}"
        )

    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)

    return np.ascontiguousarray(arr)


def _write_landmarks_from_pose(
    landmarks: np.ndarray,
    frame_idx: int,
    pose_landmarks: list,
) -> None:
    for landmark_idx, landmark in enumerate(pose_landmarks[:NUM_POSE_LANDMARKS]):
        landmarks[frame_idx, landmark_idx, 0] = landmark.x
        landmarks[frame_idx, landmark_idx, 1] = landmark.y
        landmarks[frame_idx, landmark_idx, 2] = landmark.z
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", np.nan)
        landmarks[frame_idx, landmark_idx, 3] = visibility


def _extract_pose_landmarks_legacy(
    frame_batch: np.ndarray,
    *,
    model_complexity: int,
    smooth_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> np.ndarray:
    mediapipe = _require_mediapipe()
    landmarks = np.full(
        (frame_batch.shape[0], NUM_POSE_LANDMARKS, 4),
        np.nan,
        dtype=np.float32,
    )

    mp_pose = mediapipe.solutions.pose

    with _suppress_native_logs(), mp_pose.Pose(
        static_image_mode=frame_batch.shape[0] == 1,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        for frame_idx, frame in enumerate(frame_batch):
            results = pose.process(frame)
            if results.pose_landmarks is None:
                continue

            _write_landmarks_from_pose(
                landmarks,
                frame_idx,
                results.pose_landmarks.landmark,
            )

    return landmarks


def _extract_pose_landmarks_tasks(
    frame_batch: np.ndarray,
    *,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> np.ndarray:
    mediapipe = _require_mediapipe()
    model_path = _resolve_pose_model_path(model_complexity)

    landmarks = np.full(
        (frame_batch.shape[0], NUM_POSE_LANDMARKS, 4),
        np.nan,
        dtype=np.float32,
    )

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    with _suppress_native_logs(), mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:
        for frame_idx, frame in enumerate(frame_batch):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = frame_idx * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                continue

            _write_landmarks_from_pose(
                landmarks,
                frame_idx,
                result.pose_landmarks[0],
            )

    return landmarks


def _extract_pose_landmarks_single_video(
    frames: np.ndarray | torch.Tensor,
    *,
    model_complexity: int,
    smooth_landmarks: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> np.ndarray:
    frame_batch = _normalize_frames(frames)
    mediapipe = _require_mediapipe()

    if hasattr(mediapipe, "solutions") and hasattr(mediapipe.solutions, "pose"):
        return _extract_pose_landmarks_legacy(
            frame_batch,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    if not hasattr(mediapipe, "tasks"):
        raise AttributeError(
            "Installed mediapipe package does not expose 'solutions' or 'tasks'. "
            "Install a supported version with: pip install mediapipe"
        )

    return _extract_pose_landmarks_tasks(
        frame_batch,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_pose_landmarks(
    frames: np.ndarray | torch.Tensor,
    *,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> np.ndarray:
    """
    Extract MediaPipe body pose landmarks from RGB frame sequences.

    Uses MediaPipe Pose Landmarker (Tasks API) when available, with a legacy
    ``mp.solutions.pose`` fallback on older mediapipe versions. Returns 33 body
    landmarks only (no face mesh).

    Args:
        frames: RGB frames as uint8 or float arrays/tensors.
            Supported shapes:
                [T, H, W, 3], [T, 3, H, W], [H, W, 3], [3, H, W],
                [B, T, H, W, 3], or [B, T, 3, H, W].
        model_complexity: Pose model variant in {0, 1, 2}
            (lite, full, heavy) for the Tasks API.
        smooth_landmarks: Used only by the legacy solutions API.
        min_detection_confidence: Minimum confidence for pose detection.
        min_tracking_confidence: Minimum confidence for pose tracking.

    Returns:
        ndarray with shape [T, 33, 4] for a single video, or
        [B, T, 33, 4] for a batched input. Each landmark stores
        (x, y, z, visibility) in normalized image coordinates.
        Frames with no detected pose are filled with NaN.
    """
    if _frame_ndim(frames) == 5:
        batch_landmarks = [
            _extract_pose_landmarks_single_video(
                frames[batch_idx],
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            for batch_idx in range(frames.shape[0])
        ]
        return np.stack(batch_landmarks, axis=0)

    return _extract_pose_landmarks_single_video(
        frames,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
