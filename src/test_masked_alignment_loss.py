"""Unit tests for masked pose-to-chunk alignment loss in train.py."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_PREPROCESS_DIR = _SRC_DIR / "pre_process"
if str(_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS_DIR))

from dataloder import (  # noqa: E402
    _build_transcript_alignment_mask,
    _clean_alignment_transcript_markers,
)
from losses import (
    compute_classification_loss,
    compute_masked_alignment_loss,
    get_batch_alignment_targets,
)
from model.custom_mamba import LandmarkGestureClassifier


def _make_attention(
    batch_size: int,
    num_frames: int,
    num_chunks: int,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    logits = torch.randn(
        batch_size,
        num_frames,
        num_chunks,
        requires_grad=requires_grad,
    )
    attention = torch.softmax(logits, dim=-1)
    if requires_grad:
        attention.retain_grad()
    return attention


def _partially_annotated_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_partially_annotated_batch_contributors() -> None:
    targets = _partially_annotated_targets()
    chunk_mask = torch.ones(3, 5, dtype=torch.bool)
    attention = _make_attention(3, 4, 5)

    loss, num_annotated, _ = compute_masked_alignment_loss(
        attention,
        targets,
        chunk_mask,
    )
    assert num_annotated == 2

    unchanged = attention.clone()
    unchanged[1] = torch.softmax(torch.randn(4, 5) * 20.0, dim=-1)
    loss_unannotated_changed, num_annotated_again, _ = compute_masked_alignment_loss(
        unchanged,
        targets,
        chunk_mask,
    )
    assert num_annotated_again == 2
    assert torch.allclose(loss, loss_unannotated_changed)

    annotated_changed = attention.clone()
    annotated_changed[0] = torch.softmax(torch.randn(4, 5) * 20.0, dim=-1)
    loss_annotated_changed, _, _ = compute_masked_alignment_loss(
        annotated_changed,
        targets,
        chunk_mask,
    )
    assert not torch.allclose(loss, loss_annotated_changed)


def test_correct_attention_has_lower_loss() -> None:
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    chunk_mask = torch.ones(1, 5, dtype=torch.bool)

    bad_attention = torch.tensor(
        [[[0.70, 0.05, 0.10, 0.10, 0.05]]],
        dtype=torch.float32,
    )
    good_attention = torch.tensor(
        [[[0.05, 0.80, 0.05, 0.05, 0.05]]],
        dtype=torch.float32,
    )

    bad_loss, num_annotated, _ = compute_masked_alignment_loss(
        bad_attention,
        targets,
        chunk_mask,
    )
    good_loss, _, _ = compute_masked_alignment_loss(
        good_attention,
        targets,
        chunk_mask,
    )
    assert num_annotated == 1
    assert good_loss.item() < bad_loss.item()


def test_unannotated_sample_is_invariant() -> None:
    targets = _partially_annotated_targets()
    chunk_mask = torch.ones(3, 5, dtype=torch.bool)
    base_attention = _make_attention(3, 4, 5)

    base_loss, _, _ = compute_masked_alignment_loss(
        base_attention,
        targets,
        chunk_mask,
    )

    for scale in (0.01, 1.0, 50.0):
        perturbed = base_attention.clone()
        perturbed[1] = torch.softmax(torch.randn(4, 5) * scale, dim=-1)
        perturbed_loss, _, _ = compute_masked_alignment_loss(
            perturbed,
            targets,
            chunk_mask,
        )
        assert torch.allclose(base_loss, perturbed_loss)


def test_alignment_loss_gradients() -> None:
    targets = _partially_annotated_targets()
    chunk_mask = torch.ones(3, 5, dtype=torch.bool)
    attention = _make_attention(3, 4, 5, requires_grad=True)

    alignment_loss, num_annotated, _ = compute_masked_alignment_loss(
        attention,
        targets,
        chunk_mask,
    )
    assert num_annotated == 2
    alignment_loss.backward()
    assert attention.grad is not None
    assert torch.isfinite(attention.grad).all()
    assert float(attention.grad.abs().sum().item()) > 0.0

    batch_size = 2
    num_frames = 8
    num_chunks = 5
    num_joints = 25
    in_channels = 4
    model = LandmarkGestureClassifier(
        num_classes=2,
        num_frames=num_frames,
        num_joints=num_joints,
        in_channels=in_channels,
        temporal_embed_dim=64,
        embed_dim=32,
        gcn_layers=1,
        gcn_top_k=4,
        use_gcn=True,
        use_mamba_encoder=True,
        mamba_layers=1,
        use_transcripts=True,
        transcript_embed_dim=32,
        num_transcript_chunks=num_chunks,
        max_transcript_chunks=16,
        transcript_dropout_probability=0.0,
    )
    model.train()
    model.zero_grad(set_to_none=True)

    landmarks = torch.randn(batch_size, num_frames, num_joints, in_channels)
    transcript_chunks = [
        [f"sample{b}_chunk{c}" for c in range(num_chunks)]
        for b in range(batch_size)
    ]
    transcript_chunk_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    alignment_targets = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    _logits, attention_weights, _gate, returned_chunk_mask = model(
        landmarks,
        transcript_chunks=transcript_chunks,
        transcript_chunk_mask=transcript_chunk_mask,
        return_attention=True,
    )
    model_loss, num_annotated_model, _ = compute_masked_alignment_loss(
        attention_weights,
        alignment_targets,
        returned_chunk_mask,
    )
    assert num_annotated_model == 2
    model_loss.backward()

    assert model.pose_query_projection.weight.grad is not None
    assert model.text_key_projection.weight.grad is not None
    assert torch.isfinite(model.pose_query_projection.weight.grad).all()
    assert torch.isfinite(model.text_key_projection.weight.grad).all()
    assert float(model.pose_query_projection.weight.grad.abs().sum().item()) > 0.0
    assert float(model.text_key_projection.weight.grad.abs().sum().item()) > 0.0


def test_no_annotation_batch() -> None:
    batch_size = 4
    num_frames = 3
    num_chunks = 5
    targets = torch.zeros(batch_size, num_chunks, dtype=torch.float32)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    attention = _make_attention(batch_size, num_frames, num_chunks, requires_grad=True)

    alignment_loss, num_annotated, _ = compute_masked_alignment_loss(
        attention,
        targets,
        chunk_mask,
    )
    assert num_annotated == 0
    assert torch.isfinite(alignment_loss)
    assert alignment_loss.item() == 0.0

    logits = torch.randn(batch_size, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    classification_loss = compute_classification_loss(logits, labels)
    combined_loss = classification_loss + 0.05 * alignment_loss
    assert torch.allclose(combined_loss, classification_loss)


def test_invalid_target_on_empty_chunk_raises() -> None:
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    chunk_mask = torch.tensor([[True, False, True, False, False]], dtype=torch.bool)
    attention = _make_attention(1, 2, 5)

    try:
        compute_masked_alignment_loss(
            attention,
            targets,
            chunk_mask,
        )
    except ValueError as exc:
        assert "invalid transcript chunks" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when alignment target points to an invalid chunk."
        )


def test_configurable_num_chunks() -> None:
    for num_chunks in (3, 5, 8):
        targets = torch.zeros(2, num_chunks, dtype=torch.float32)
        targets[0, 1] = 1.0
        targets[1, num_chunks - 1] = 1.0
        chunk_mask = torch.ones(2, num_chunks, dtype=torch.bool)

        attention = _make_attention(2, 4, num_chunks)
        loss, num_annotated, mean_attention = compute_masked_alignment_loss(
            attention,
            targets,
            chunk_mask,
        )
        assert num_annotated == 2
        assert torch.isfinite(loss)
        assert mean_attention.shape == (2, num_chunks)
        assert torch.allclose(
            mean_attention.sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
        )


def test_soft_attention_preservation() -> None:
    source = inspect.getsource(compute_masked_alignment_loss)
    forbidden = ("argmax", "topk", "top_k", "one_hot", "hard")
    for token in forbidden:
        assert token not in source, (
            f"compute_masked_alignment_loss must not use hard selection token {token!r}"
        )

    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    chunk_mask = torch.ones(1, 5, dtype=torch.bool)
    logits = torch.zeros(1, 1, 5, requires_grad=True)
    logits.data[0, 0, 1] = 2.0

    attention_a = torch.softmax(logits, dim=-1)
    loss_a, _, _ = compute_masked_alignment_loss(attention_a, targets, chunk_mask)

    logits_b = logits.detach().clone().requires_grad_(True)
    logits_b.data[0, 0, 1] = 2.5
    attention_b = torch.softmax(logits_b, dim=-1)
    loss_b, _, _ = compute_masked_alignment_loss(attention_b, targets, chunk_mask)

    assert loss_b.item() < loss_a.item()
    assert (attention_a[0, 0, 1] - attention_b[0, 0, 1]).abs().item() > 0.0


def test_get_batch_alignment_targets() -> None:
    batch = {
        "transcript_alignment_mask": torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }
    targets = get_batch_alignment_targets(batch, torch.device("cpu"), 5)
    assert targets.shape == (2, 5)
    assert targets.dtype == torch.float32


def test_press_transcript_alignment_mask_example() -> None:
    """
    Verify dataloader chunk-alignment labeling on a hardcoded press example.

    Uses the same helpers as ``AphasiaGestureSegmentDataset.__getitem__``:
    ``_clean_alignment_transcript_markers`` then ``_build_transcript_alignment_mask``.
    """
    transcript_chunks = [
        "PAR: and then put",
        " the money",
        " and press start .",
    ]
    raw_full_transcript = (
        "PAR: and then put the money and <+utt_s>press<+utt_e> start ."
    )
    aligned_utterance = "press"

    full_transcript = _clean_alignment_transcript_markers(raw_full_transcript)
    alignment_mask = _build_transcript_alignment_mask(
        transcript_chunks,
        full_transcript=full_transcript,
        aligned_utterance=aligned_utterance,
    )

    # "press" appears in the third chunk (" and press start ."), which is the
    # longest valid subset of the cleaned full transcript containing the utterance.
    assert alignment_mask.tolist() == [0.0, 0.0, 1.0]

    # When the aligned utterance sits in the middle chunk text, index 1 is selected.
    middle_chunk_mask = _build_transcript_alignment_mask(
        [
            "PAR: and then put the money and ",
            "press",
            " start .",
        ],
        full_transcript=full_transcript,
        aligned_utterance=aligned_utterance,
    )
    assert middle_chunk_mask.tolist() == [0.0, 1.0, 0.0]

    # The original ``ppress`` marker typo merges into ``ppress`` after cleaning,
    # so no chunk is both a subset of the full transcript and contains ``press``.
    typo_full_transcript = _clean_alignment_transcript_markers(
        "PAR: and then put the money and <+utt_s>ppress<+utt_e> start ."
    )
    typo_mask = _build_transcript_alignment_mask(
        transcript_chunks,
        full_transcript=typo_full_transcript,
        aligned_utterance=aligned_utterance,
    )
    assert typo_mask.tolist() == [0.0, 0.0, 0.0]


def main() -> None:
    test_partially_annotated_batch_contributors()
    test_correct_attention_has_lower_loss()
    test_unannotated_sample_is_invariant()
    test_alignment_loss_gradients()
    test_no_annotation_batch()
    test_invalid_target_on_empty_chunk_raises()
    test_configurable_num_chunks()
    test_soft_attention_preservation()
    test_get_batch_alignment_targets()
    test_press_transcript_alignment_mask_example()
    print("masked alignment loss tests: OK")


if __name__ == "__main__":
    main()
