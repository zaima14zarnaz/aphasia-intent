#!/usr/bin/env python3
"""Focused checks for OpenPose BODY_25 + hands loading in pre_process.load_aligned_media."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pre_process import (
    OPENPOSE_BODY_HAND_JOINTS,
    OPENPOSE_BODY_JOINTS,
    OPENPOSE_HAND_JOINTS,
    _is_missing_joint,
    _store_joint,
    _visibility_column,
    load_aligned_media,
)


FMT = ["x", "y", "z", "confidence"]
XYZ_COLS = [0, 1, 2]
VIS_COL = 3


def _png(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)


def _write_segment(root: Path, frames: list[dict], *, extractor: str = "openpose_body25_hand21",
                   body_joints: int = OPENPOSE_BODY_JOINTS, fmt: list[str] | None = None,
                   hand_shape: list[int] | None = None) -> tuple[Path, Path]:
    fmt = list(FMT if fmt is None else fmt)
    rgb = root / "rgb"
    landmarks = root / "landmarks"
    rgb.mkdir()
    landmarks.mkdir()
    for frame in frames:
        _png(rgb / frame["frame_file"])
    payload = {
        "extractor": extractor,
        "num_frames": len(frames),
        "landmark_shape": [len(frames), body_joints, len(fmt)],
        "landmark_format": fmt,
        "frames": frames,
    }
    if hand_shape is not None or extractor == "openpose_body25_hand21":
        payload["hand_landmark_shape"] = hand_shape or [len(frames), OPENPOSE_HAND_JOINTS, len(fmt)]
    path = landmarks / "landmarks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return rgb, path


def _joint(x=0.1, y=0.2, z=0.0, c=0.5):
    return [x, y, z, c]


class MissingJointHelpers(unittest.TestCase):
    def test_visibility_prefers_visibility_then_confidence(self):
        self.assertEqual(_visibility_column(["x", "y", "z", "visibility"]), 3)
        self.assertEqual(_visibility_column(["x", "y", "z", "confidence"]), 3)
        self.assertIsNone(_visibility_column(["x", "y", "z"]))

    def test_missing_patterns(self):
        self.assertTrue(_is_missing_joint(None, 4))
        self.assertTrue(_is_missing_joint([None, None, None, None], 4))
        self.assertTrue(_is_missing_joint([0.1, None, 0.0, 0.2], 4))
        self.assertTrue(_is_missing_joint([0.1, float("nan"), 0.0, 0.2], 4))
        self.assertFalse(_is_missing_joint([0.0, 0.0, 0.0, 0.0], 4))
        self.assertFalse(_is_missing_joint([0.1, 0.2, 0.0, 0.9], 4))
        self.assertFalse(_is_missing_joint([0.1, 0.2], 4))  # wrong length -> not "missing"

    def test_store_joint_masks_and_copies_confidence(self):
        xyz = np.zeros((1, 2, 3), dtype=np.float32)
        vis = np.zeros((1, 2), dtype=np.float32)
        valid = np.zeros((1, 2), dtype=np.bool_)
        _store_joint(xyz, vis, valid, 0, 0, None, fmt=FMT, xyz_columns=XYZ_COLS,
                     vis_column=VIS_COL, landmark_path=Path("x"), frame_label=0, joint_label="0")
        _store_joint(xyz, vis, valid, 0, 1, [0.0, 0.0, 0.0, 0.0], fmt=FMT, xyz_columns=XYZ_COLS,
                     vis_column=VIS_COL, landmark_path=Path("x"), frame_label=0, joint_label="1")
        self.assertFalse(valid[0, 0])
        self.assertTrue(valid[0, 1])
        np.testing.assert_array_equal(xyz[0, 1], [0.0, 0.0, 0.0])
        self.assertEqual(vis[0, 1], 0.0)

    def test_store_joint_rejects_wrong_length(self):
        xyz = np.zeros((1, 1, 3), dtype=np.float32)
        vis = np.zeros((1, 1), dtype=np.float32)
        valid = np.zeros((1, 1), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "Wrong landmark length"):
            _store_joint(xyz, vis, valid, 0, 0, [0.1, 0.2], fmt=FMT, xyz_columns=XYZ_COLS,
                         vis_column=VIS_COL, landmark_path=Path("x"), frame_label=0, joint_label="0")

    def test_store_joint_clips_openpose_confidence(self):
        xyz = np.zeros((1, 1, 3), dtype=np.float32)
        vis = np.zeros((1, 1), dtype=np.float32)
        valid = np.zeros((1, 1), dtype=np.bool_)
        _store_joint(
            xyz, vis, valid, 0, 0,
            [0.4795164167881012, 0.5940351486206055, 0.0, 1.0190800428390503],
            fmt=FMT, xyz_columns=XYZ_COLS, vis_column=VIS_COL,
            landmark_path=Path("x"), frame_label=17, joint_label="56",
        )
        self.assertTrue(valid[0, 0])
        self.assertEqual(float(vis[0, 0]), 1.0)
        np.testing.assert_allclose(xyz[0, 0], [0.4795164167881012, 0.5940351486206055, 0.0])


class OpenPoseLoadAlignedMedia(unittest.TestCase):
    def test_concat_order_missing_hands_and_null_joints(self):
        body = [_joint(i * 0.01, 0.1, 0.0, 0.8) for i in range(OPENPOSE_BODY_JOINTS)]
        body[8] = [None, None, None, None]
        body[9] = None
        body[10] = [0.5, None, 0.0, 0.1]
        left = [_joint(0.2, 0.3, 0.0, 0.4) for _ in range(OPENPOSE_HAND_JOINTS)]
        left[0] = [0.0, 0.0, 0.0, 0.0]
        right = [_joint(0.7, 0.8, 0.0, 0.6) for _ in range(OPENPOSE_HAND_JOINTS)]
        frames = [{
            "frame_index": 0,
            "frame_file": "0000.png",
            "landmarks": body,
            "hand_left": left,
            "hand_right": right,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            rgb, path = _write_segment(Path(tmp), frames)
            paths, indices, xyz, vis, valid = load_aligned_media(rgb, path)
        self.assertEqual(paths[0].endswith("0000.png"), True)
        self.assertEqual(indices, [0])
        self.assertEqual(xyz.shape, (1, OPENPOSE_BODY_HAND_JOINTS, 3))
        self.assertEqual(vis.shape, (1, OPENPOSE_BODY_HAND_JOINTS))
        self.assertEqual(valid.shape, (1, OPENPOSE_BODY_HAND_JOINTS))
        self.assertFalse(valid[0, 8])
        self.assertFalse(valid[0, 9])
        self.assertFalse(valid[0, 10])
        np.testing.assert_array_equal(xyz[0, 8], 0)
        self.assertEqual(vis[0, 8], 0.0)
        np.testing.assert_allclose(xyz[0, 0], [0.0, 0.1, 0.0])
        self.assertAlmostEqual(float(vis[0, 0]), 0.8)
        # left hand starts at 25; zero coords + zero confidence remain valid
        self.assertTrue(valid[0, 25])
        np.testing.assert_array_equal(xyz[0, 25], [0.0, 0.0, 0.0])
        self.assertEqual(float(vis[0, 25]), 0.0)
        self.assertAlmostEqual(float(vis[0, 26]), 0.4)
        # right hand starts at 46
        self.assertAlmostEqual(float(vis[0, 46]), 0.6)
        np.testing.assert_allclose(xyz[0, 46], [0.7, 0.8, 0.0])

    def test_missing_entire_hand_is_masked(self):
        body = [_joint() for _ in range(OPENPOSE_BODY_JOINTS)]
        frames = [{
            "frame_index": 0,
            "frame_file": "0000.png",
            "landmarks": body,
            "hand_left": None,
            "hand_right": [],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            rgb, path = _write_segment(Path(tmp), frames)
            _, _, xyz, vis, valid = load_aligned_media(rgb, path)
        self.assertTrue(valid[0, :25].all())
        self.assertFalse(valid[0, 25:46].any())
        self.assertFalse(valid[0, 46:].any())
        np.testing.assert_array_equal(xyz[0, 25:], 0)
        np.testing.assert_array_equal(vis[0, 25:], 0)

    def test_wrong_hand_length_raises(self):
        body = [_joint() for _ in range(OPENPOSE_BODY_JOINTS)]
        frames = [{
            "frame_index": 0,
            "frame_file": "0000.png",
            "landmarks": body,
            "hand_left": [_joint() for _ in range(3)],
            "hand_right": [_joint() for _ in range(OPENPOSE_HAND_JOINTS)],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            rgb, path = _write_segment(Path(tmp), frames)
            with self.assertRaisesRegex(ValueError, "Wrong hand_left joint count"):
                load_aligned_media(rgb, path)

    def test_mediapipe_body_only_unchanged(self):
        body = [[0.1, 0.2, 0.3, 0.9] for _ in range(33)]
        body[5] = [None, None, None, None]
        frames = [{"frame_index": 0, "frame_file": "0000.png", "landmarks": body}]
        with tempfile.TemporaryDirectory() as tmp:
            rgb, path = _write_segment(
                Path(tmp), frames, extractor="mediapipe_pose", body_joints=33,
                fmt=["x", "y", "z", "visibility"], hand_shape=None,
            )
            # Remove OpenPose-only metadata that the helper may have added.
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("hand_landmark_shape", None)
            payload["extractor"] = "mediapipe_pose"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, _, xyz, vis, valid = load_aligned_media(rgb, path)
        self.assertEqual(xyz.shape, (1, 33, 3))
        self.assertFalse(valid[0, 5])
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(float(vis[0, 0]), 0.9)
        np.testing.assert_allclose(xyz[0, 0], [0.1, 0.2, 0.3])


class RealFailingSample(unittest.TestCase):
    SAMPLE_JSON = Path(
        "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments_landmarks/"
        "openpose/81-1.mp4/gest_13/landmarks.json"
    )
    SAMPLE_RGB = Path(
        "/data/quantization/zaima/aphasiabank/neural2/control/gest_segments/81-1.mp4/gest_13"
    )

    @unittest.skipUnless(SAMPLE_JSON.is_file() and SAMPLE_RGB.is_dir(), "sample data unavailable")
    def test_failing_gest_13(self):
        paths, indices, xyz, vis, valid = load_aligned_media(self.SAMPLE_RGB, self.SAMPLE_JSON)
        self.assertEqual(xyz.shape, (44, 67, 3))
        self.assertEqual(vis.shape, (44, 67))
        self.assertEqual(valid.shape, (44, 67))
        self.assertEqual(len(paths), 44)
        self.assertEqual(indices[0], 0)
        self.assertFalse(bool(valid[0, 8]))
        np.testing.assert_array_equal(xyz[0, 8], 0)
        self.assertEqual(float(vis[0, 8]), 0.0)
        self.assertGreater(int(valid.sum()), 0)
        self.assertGreater(int((~valid).sum()), 0)
        # Representative hand confidences when present.
        if valid[0, 25]:
            self.assertGreaterEqual(float(vis[0, 25]), 0.0)
            self.assertLessEqual(float(vis[0, 25]), 1.0)
        if valid[0, 46]:
            self.assertGreaterEqual(float(vis[0, 46]), 0.0)
            self.assertLessEqual(float(vis[0, 46]), 1.0)


if __name__ == "__main__":
    unittest.main()
