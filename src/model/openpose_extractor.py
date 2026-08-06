"""
OpenPose BODY_25 (+ optional hand) landmark extraction for gesture video frames.

Uses the compiled OpenPose binary at ``external/openpose/build/.../openpose.bin``.
Each frame sequence is written to a temporary image directory, processed in one
OpenPose invocation, and parsed from JSON keypoint files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

DEFAULT_OPENPOSE_ROOT = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/external/openpose"
)
DEFAULT_IMAGE_PATH = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/model/single_frame_2.png"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/model/openpose_single_frame_output"
)

NUM_POSE_LANDMARKS = 25
NUM_HAND_LANDMARKS = 21


def _frame_ndim(frames: np.ndarray | torch.Tensor) -> int:
    if isinstance(frames, np.ndarray):
        return frames.ndim
    return int(frames.ndim)


def _normalize_frames(frames: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert frames to a contiguous uint8 RGB batch with shape [T, H, W, 3]."""
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


def _reshape_keypoints(flat_keypoints: list[float], num_keypoints: int) -> np.ndarray:
    """Convert flattened OpenPose x,y,confidence list into [K, 3]."""
    if not flat_keypoints:
        return np.zeros((num_keypoints, 3), dtype=np.float32)

    arr = np.asarray(flat_keypoints, dtype=np.float32)
    expected = num_keypoints * 3
    if arr.size != expected:
        raise ValueError(
            f"Expected {expected} values for {num_keypoints} keypoints, got {arr.size}"
        )

    return arr.reshape(num_keypoints, 3)


def _select_person(people: list[dict[str, Any]]) -> int:
    """Select the person with the highest sum of BODY_25 keypoint confidences."""
    if not people:
        return -1

    best_idx = 0
    best_score = -1.0

    for idx, person in enumerate(people):
        pose = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32)
        if pose.size >= 3:
            conf_sum = float(pose.reshape(-1, 3)[:, 2].sum())
        else:
            conf_sum = 0.0

        if conf_sum > best_score:
            best_score = conf_sum
            best_idx = idx

    return best_idx


def _prepare_env(openpose_root: Path) -> dict[str, str]:
    """Prepare runtime environment for the compiled OpenPose binary."""
    env = os.environ.copy()

    conda_prefix = Path(env.get("CONDA_PREFIX", ""))
    if conda_prefix:
        cuda_root = conda_prefix / "cuda-root"

        env["CUDA_ROOT"] = str(cuda_root)
        env["CUDA_HOME"] = str(cuda_root)
        env["CUDA_PATH"] = str(cuda_root)
        env["CUDA_TOOLKIT_ROOT_DIR"] = str(cuda_root)

        extra_ld_paths = [
            str(openpose_root / "build" / "src" / "openpose"),
            str(openpose_root / "build" / "caffe" / "lib"),
            str(openpose_root / "build" / "3rdparty" / "caffe" / "lib"),
            str(conda_prefix / "lib"),
            str(cuda_root / "lib64"),
            str(conda_prefix / "targets" / "x86_64-linux" / "lib"),
        ]

        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            extra_ld_paths + ([current_ld] if current_ld else [])
        )

        extra_path = [
            str(cuda_root / "bin"),
            str(conda_prefix / "bin"),
            str(conda_prefix / "nvvm" / "bin"),
        ]

        current_path = env.get("PATH", "")
        env["PATH"] = ":".join(extra_path + ([current_path] if current_path else []))

    return env


def _resolve_openpose_paths(openpose_root: Path) -> tuple[Path, Path]:
    openpose_root = openpose_root.expanduser().resolve()
    openpose_bin = openpose_root / "build" / "examples" / "openpose" / "openpose.bin"
    model_dir = openpose_root / "models"

    if not openpose_bin.is_file():
        raise FileNotFoundError(f"OpenPose binary not found: {openpose_bin}")

    required_models = [
        model_dir / "pose" / "body_25" / "pose_iter_584000.caffemodel",
    ]
    missing_models = [str(path) for path in required_models if not path.is_file()]
    if missing_models:
        raise FileNotFoundError(
            "Missing OpenPose model weights:\n"
            + "\n".join(missing_models)
            + "\n\nDownload/copy them before running this module."
        )

    return openpose_bin, model_dir


def _format_openpose_failure(stdout: str, returncode: int) -> str:
    message = f"OpenPose failed with exit code {returncode}.\n{stdout}"
    if "out of memory" in stdout.lower() or "cudaSuccess (2" in stdout:
        message += (
            "\n\nCUDA out of memory. Try one of:\n"
            "  --num_gpu_start 1     use a different GPU (GPU 0 may be busy)\n"
            "  --cpu                 run on CPU (slower, no GPU memory needed)\n"
            "  --net_resolution -1x256   lower inference resolution\n"
            "Also free GPU memory by stopping other jobs (e.g. train.py)."
        )
    return message


def run_openpose_on_image_dir(
    image_dir: Path,
    openpose_root: Path,
    output_dir: Path,
    *,
    enable_hand: bool = False,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
) -> list[Path]:
    """
    Run OpenPose on every image in ``image_dir``.

    Returns:
        Sorted list of generated ``*_keypoints.json`` paths.
    """
    image_dir = image_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    json_dir = output_dir / "openpose_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    openpose_bin, model_dir = _resolve_openpose_paths(openpose_root)

    if enable_hand:
        hand_model = model_dir / "hand" / "pose_iter_102000.caffemodel"
        if not hand_model.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {hand_model}. "
                "Disable enable_hand or download hand weights."
            )

    cmd = [
        str(openpose_bin),
        "--image_dir",
        str(image_dir),
        "--write_json",
        str(json_dir),
        "--model_folder",
        str(model_dir),
        "--model_pose",
        "BODY_25",
        "--display",
        "0",
        "--render_pose",
        "0",
        "--keypoint_scale",
        "0",
        "--num_gpu",
        str(num_gpu),
        "--net_resolution",
        net_resolution,
        "--scale_number",
        str(scale_number),
    ]
    if num_gpu > 0:
        cmd.extend(["--num_gpu_start", str(num_gpu_start)])
    if enable_hand:
        cmd.append("--hand")

    if verbose:
        print("Running OpenPose command:")
        print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(openpose_root.expanduser().resolve()),
        env=_prepare_env(openpose_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if verbose:
        print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            _format_openpose_failure(result.stdout, result.returncode)
        )

    json_files = sorted(json_dir.glob("*_keypoints.json"))
    if not json_files:
        raise FileNotFoundError(f"No OpenPose JSON output found in {json_dir}")

    return json_files


def run_openpose_on_single_frame(
    image_path: Path,
    openpose_root: Path,
    output_dir: Path,
    *,
    enable_hand: bool = True,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
) -> Path:
    """Run OpenPose on one image via a temporary image directory."""
    image_path = image_path.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="openpose_single_frame_") as tmp:
        tmp_dir = Path(tmp)
        tmp_image = tmp_dir / image_path.name

        try:
            tmp_image.symlink_to(image_path)
        except OSError:
            shutil.copy2(image_path, tmp_image)

        json_files = run_openpose_on_image_dir(
            tmp_dir,
            openpose_root,
            output_dir,
            enable_hand=enable_hand,
            num_gpu=num_gpu,
            num_gpu_start=num_gpu_start,
            net_resolution=net_resolution,
            scale_number=scale_number,
            verbose=verbose,
        )

    return json_files[0]


def _keypoints_xy_conf_to_landmarks4(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Convert OpenPose [K, 3] pixel keypoints to [K, 4] normalized landmarks.

    Output channels are (x, y, z, confidence) with x/y in [0, 1], z=0, and
    confidence copied from OpenPose.
    """
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(
            f"Expected keypoints shaped [K, 3], got {tuple(keypoints.shape)}"
        )

    num_keypoints = keypoints.shape[0]
    landmarks = np.full((num_keypoints, 4), np.nan, dtype=np.float32)

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {image_width}x{image_height}"
        )

    for joint_idx, (x_px, y_px, confidence) in enumerate(keypoints):
        if confidence <= 0.0:
            continue
        landmarks[joint_idx, 0] = float(x_px) / float(image_width)
        landmarks[joint_idx, 1] = float(y_px) / float(image_height)
        landmarks[joint_idx, 2] = 0.0
        landmarks[joint_idx, 3] = float(confidence)

    return landmarks


def parse_openpose_json(
    json_path: Path,
    *,
    image_width: int,
    image_height: int,
    enable_hand: bool = False,
) -> dict[str, Any]:
    """Parse OpenPose JSON and return normalized landmark arrays."""
    with json_path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    people = data.get("people", [])
    selected_idx = _select_person(people)

    if selected_idx < 0:
        pose_body25 = np.full((NUM_POSE_LANDMARKS, 4), np.nan, dtype=np.float32)
        hand_left = np.full((NUM_HAND_LANDMARKS, 4), np.nan, dtype=np.float32)
        hand_right = np.full((NUM_HAND_LANDMARKS, 4), np.nan, dtype=np.float32)
    else:
        person = people[selected_idx]
        pose_body25 = _keypoints_xy_conf_to_landmarks4(
            _reshape_keypoints(person.get("pose_keypoints_2d", []), NUM_POSE_LANDMARKS),
            image_width,
            image_height,
        )
        hand_left = _keypoints_xy_conf_to_landmarks4(
            _reshape_keypoints(
                person.get("hand_left_keypoints_2d", []),
                NUM_HAND_LANDMARKS,
            ),
            image_width,
            image_height,
        )
        hand_right = _keypoints_xy_conf_to_landmarks4(
            _reshape_keypoints(
                person.get("hand_right_keypoints_2d", []),
                NUM_HAND_LANDMARKS,
            ),
            image_width,
            image_height,
        )

    result = {
        "selected_person": selected_idx,
        "pose_body25": pose_body25,
    }
    if enable_hand:
        result["hand_left"] = hand_left
        result["hand_right"] = hand_right

    return result


def _json_path_for_frame_stem(json_dir: Path, frame_stem: str) -> Path | None:
    direct = json_dir / f"{frame_stem}_keypoints.json"
    if direct.is_file():
        return direct

    matches = sorted(json_dir.glob(f"{frame_stem}*_keypoints.json"))
    return matches[0] if matches else None


def _extract_openpose_landmarks_single_video(
    frames: np.ndarray | torch.Tensor,
    *,
    openpose_root: Path,
    enable_hand: bool = False,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
) -> dict[str, np.ndarray | None]:
    frame_batch = _normalize_frames(frames)
    num_frames, image_height, image_width = frame_batch.shape[:3]

    pose_landmarks = np.full(
        (num_frames, NUM_POSE_LANDMARKS, 4),
        np.nan,
        dtype=np.float32,
    )
    hand_left_landmarks: np.ndarray | None = (
        np.full((num_frames, NUM_HAND_LANDMARKS, 4), np.nan, dtype=np.float32)
        if enable_hand
        else None
    )
    hand_right_landmarks: np.ndarray | None = (
        np.full((num_frames, NUM_HAND_LANDMARKS, 4), np.nan, dtype=np.float32)
        if enable_hand
        else None
    )

    with tempfile.TemporaryDirectory(prefix="openpose_frames_") as tmp:
        tmp_root = Path(tmp)
        image_dir = tmp_root / "images"
        output_dir = tmp_root / "output"
        image_dir.mkdir(parents=True, exist_ok=True)

        frame_stems: list[str] = []
        for frame_idx, frame in enumerate(frame_batch):
            frame_stem = f"{frame_idx:04d}"
            frame_stems.append(frame_stem)
            frame_path = image_dir / f"{frame_stem}.png"

            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "opencv-python is required to write frames for OpenPose. "
                    "Install it with: pip install opencv-python"
                ) from exc

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Failed to write frame image: {frame_path}")

        json_files = run_openpose_on_image_dir(
            image_dir,
            openpose_root,
            output_dir,
            enable_hand=enable_hand,
            num_gpu=num_gpu,
            num_gpu_start=num_gpu_start,
            net_resolution=net_resolution,
            scale_number=scale_number,
            verbose=verbose,
        )

        if verbose:
            print(f"OpenPose wrote {len(json_files)} JSON files")

        json_dir = json_files[0].parent
        for frame_idx, frame_stem in enumerate(frame_stems):
            json_path = _json_path_for_frame_stem(json_dir, frame_stem)
            if json_path is None:
                continue

            parsed = parse_openpose_json(
                json_path,
                image_width=image_width,
                image_height=image_height,
                enable_hand=enable_hand,
            )
            pose_landmarks[frame_idx] = parsed["pose_body25"]
            if enable_hand and hand_left_landmarks is not None:
                hand_left_landmarks[frame_idx] = parsed["hand_left"]
                assert hand_right_landmarks is not None
                hand_right_landmarks[frame_idx] = parsed["hand_right"]

    return {
        "pose_body25": pose_landmarks,
        "hand_left": hand_left_landmarks,
        "hand_right": hand_right_landmarks,
    }


def _extract_pose_landmarks_single_video(
    frames: np.ndarray | torch.Tensor,
    *,
    openpose_root: Path,
    enable_hand: bool = False,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    return _extract_openpose_landmarks_single_video(
        frames,
        openpose_root=openpose_root,
        enable_hand=enable_hand,
        num_gpu=num_gpu,
        num_gpu_start=num_gpu_start,
        net_resolution=net_resolution,
        scale_number=scale_number,
        verbose=verbose,
    )["pose_body25"]


def extract_openpose_landmarks(
    frames: np.ndarray | torch.Tensor,
    *,
    openpose_root: str | Path = DEFAULT_OPENPOSE_ROOT,
    enable_hand: bool = False,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
) -> dict[str, np.ndarray | None]:
    """
    Extract OpenPose BODY_25 landmarks and optional hand landmarks.

    Returns:
        dict with keys ``pose_body25`` ([T, 25, 4]), and when ``enable_hand`` is
        True also ``hand_left`` and ``hand_right`` ([T, 21, 4] each).
    """
    openpose_root = Path(openpose_root)

    if _frame_ndim(frames) == 5:
        batch_results = [
            _extract_openpose_landmarks_single_video(
                frames[batch_idx],
                openpose_root=openpose_root,
                enable_hand=enable_hand,
                num_gpu=num_gpu,
                num_gpu_start=num_gpu_start,
                net_resolution=net_resolution,
                scale_number=scale_number,
                verbose=verbose,
            )
            for batch_idx in range(frames.shape[0])
        ]
        result: dict[str, np.ndarray | None] = {
            "pose_body25": np.stack(
                [item["pose_body25"] for item in batch_results],
                axis=0,
            )
        }
        if enable_hand:
            result["hand_left"] = np.stack(
                [item["hand_left"] for item in batch_results],
                axis=0,
            )
            result["hand_right"] = np.stack(
                [item["hand_right"] for item in batch_results],
                axis=0,
            )
        else:
            result["hand_left"] = None
            result["hand_right"] = None
        return result

    return _extract_openpose_landmarks_single_video(
        frames,
        openpose_root=openpose_root,
        enable_hand=enable_hand,
        num_gpu=num_gpu,
        num_gpu_start=num_gpu_start,
        net_resolution=net_resolution,
        scale_number=scale_number,
        verbose=verbose,
    )


def extract_pose_landmarks(
    frames: np.ndarray | torch.Tensor,
    *,
    openpose_root: str | Path = DEFAULT_OPENPOSE_ROOT,
    enable_hand: bool = False,
    num_gpu: int = 1,
    num_gpu_start: int = 0,
    net_resolution: str = "-1x368",
    scale_number: int = 1,
    verbose: bool = False,
    model_complexity: int | None = None,
    smooth_landmarks: bool | None = None,
    min_detection_confidence: float | None = None,
    min_tracking_confidence: float | None = None,
) -> np.ndarray:
    """
    Extract OpenPose BODY_25 landmarks from RGB frame sequences.

    Args:
        frames: RGB frames as uint8 or float arrays/tensors.
            Supported shapes:
                [T, H, W, 3], [T, 3, H, W], [H, W, 3], [3, H, W],
                [B, T, H, W, 3], or [B, T, 3, H, W].
        openpose_root: Path to the compiled OpenPose repository.
        enable_hand: Also run OpenPose hand estimation. Hand tensors are omitted
            from the return value; use ``extract_openpose_landmarks`` instead.
        num_gpu_start: GPU index passed to OpenPose when ``num_gpu > 0``.
        net_resolution: OpenPose ``--net_resolution`` (lower uses less GPU memory).
        scale_number: OpenPose ``--scale_number`` (1 is lowest memory).
        verbose: Print OpenPose command output.

    Returns:
        ndarray with shape [T, 25, 4] for a single video, or
        [B, T, 25, 4] for a batched input. Each landmark stores
        (x, y, z, confidence) with x/y normalized to [0, 1], z=0, and
        confidence from OpenPose. Frames with no detected pose are NaN.

    Note:
        ``model_complexity``, ``smooth_landmarks``, ``min_detection_confidence``,
        and ``min_tracking_confidence`` are accepted for API compatibility with
        ``mediapipe_extractor.extract_pose_landmarks`` but ignored.
    """
    del model_complexity, smooth_landmarks, min_detection_confidence
    del min_tracking_confidence

    openpose_root = Path(openpose_root)

    if _frame_ndim(frames) == 5:
        batch_landmarks = [
            _extract_pose_landmarks_single_video(
                frames[batch_idx],
                openpose_root=openpose_root,
                enable_hand=enable_hand,
                num_gpu=num_gpu,
                num_gpu_start=num_gpu_start,
                net_resolution=net_resolution,
                scale_number=scale_number,
                verbose=verbose,
            )
            for batch_idx in range(frames.shape[0])
        ]
        return np.stack(batch_landmarks, axis=0)

    return _extract_pose_landmarks_single_video(
        frames,
        openpose_root=openpose_root,
        enable_hand=enable_hand,
        num_gpu=num_gpu,
        num_gpu_start=num_gpu_start,
        net_resolution=net_resolution,
        scale_number=scale_number,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenPose BODY_25 landmarks from a single image."
    )
    parser.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--openpose_root", type=Path, default=DEFAULT_OPENPOSE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num_gpu_start",
        type=int,
        default=0,
        help="GPU index when using CUDA (default: 0). Use 1 if GPU 0 is busy.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run OpenPose on CPU instead of GPU (slower, avoids CUDA OOM).",
    )
    parser.add_argument(
        "--net_resolution",
        default="-1x368",
        help="OpenPose net resolution, e.g. -1x368 or -1x256 (lower = less VRAM).",
    )
    parser.add_argument(
        "--scale_number",
        type=int,
        default=1,
        help="OpenPose scale_number (1 uses least GPU memory).",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Enable OpenPose hand keypoint extraction.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_openpose_on_single_frame(
        image_path=args.image_path,
        openpose_root=args.openpose_root,
        output_dir=args.output_dir,
        enable_hand=args.hand,
        num_gpu=0 if args.cpu else 1,
        num_gpu_start=args.num_gpu_start,
        net_resolution=args.net_resolution,
        scale_number=args.scale_number,
        verbose=True,
    )

    try:
        import cv2

        image_bgr = cv2.imread(str(args.image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {args.image_path}")
        image_height, image_width = image_bgr.shape[:2]
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Need opencv-python or pillow to read image dimensions. "
                "Install one with: pip install opencv-python  OR  pip install pillow"
            ) from exc

        with Image.open(args.image_path) as image:
            image_width, image_height = image.size

    landmarks = parse_openpose_json(
        json_path,
        image_width=image_width,
        image_height=image_height,
        enable_hand=args.hand,
    )

    npz_path = args.output_dir / f"{args.image_path.stem}_openpose_landmarks.npz"
    save_kwargs = {
        "pose_body25": landmarks["pose_body25"],
        "selected_person": np.asarray(landmarks["selected_person"], dtype=np.int32),
        "json_path": str(json_path),
        "image_path": str(args.image_path),
    }
    if args.hand:
        save_kwargs["hand_left"] = landmarks["hand_left"]
        save_kwargs["hand_right"] = landmarks["hand_right"]

    np.savez_compressed(npz_path, **save_kwargs)

    print("\nSaved:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ : {npz_path}")

    print("\nShapes:")
    print(f"  pose_body25: {landmarks['pose_body25'].shape}")
    if args.hand:
        print(f"  hand_left:   {landmarks['hand_left'].shape}")
        print(f"  hand_right:  {landmarks['hand_right'].shape}")

    print("\nFirst few BODY_25 keypoints [x, y, z, confidence]:")
    print(landmarks["pose_body25"][:5])


if __name__ == "__main__":
    main()
