"""
Metrics for multi-class gesture classification.
"""

from __future__ import annotations

import torch


class GestureClassificationMetrics:
    """
    Accumulate gesture-classification predictions and compute aggregate metrics.
    """

    def __init__(
        self,
        num_classes: int,
        idx_to_label: dict[int, str] | None = None,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = num_classes
        self.idx_to_label = idx_to_label

        self._true_positives = torch.zeros(num_classes, dtype=torch.long)
        self._false_positives = torch.zeros(num_classes, dtype=torch.long)
        self._false_negatives = torch.zeros(num_classes, dtype=torch.long)
        self._class_correct = torch.zeros(num_classes, dtype=torch.long)
        self._class_total = torch.zeros(num_classes, dtype=torch.long)
        self._total_correct = 0
        self._total_samples = 0

    def reset(self) -> None:
        self._true_positives.zero_()
        self._false_positives.zero_()
        self._false_negatives.zero_()
        self._class_correct.zero_()
        self._class_total.zero_()
        self._total_correct = 0
        self._total_samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 2D with shape [B, num_classes], got {tuple(logits.shape)}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"labels must be 1D with shape [B], got {tuple(labels.shape)}"
            )
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "Batch size mismatch between logits and labels: "
                f"logits.shape[0]={logits.shape[0]}, labels.shape[0]={labels.shape[0]}"
            )
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits with {self.num_classes} classes, "
                f"got {logits.shape[1]}"
            )

        predictions = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu().long()

        self._total_correct += int((predictions == labels_cpu).sum().item())
        self._total_samples += int(labels_cpu.numel())

        for class_idx in range(self.num_classes):
            pred_mask = predictions == class_idx
            true_mask = labels_cpu == class_idx

            true_positives = int((pred_mask & true_mask).sum().item())
            false_positives = int((pred_mask & ~true_mask).sum().item())
            false_negatives = int((~pred_mask & true_mask).sum().item())
            class_total = int(true_mask.sum().item())

            self._true_positives[class_idx] += true_positives
            self._false_positives[class_idx] += false_positives
            self._false_negatives[class_idx] += false_negatives
            self._class_correct[class_idx] += true_positives
            self._class_total[class_idx] += class_total

    def compute(self) -> dict:
        eps = 1e-8

        if self._total_samples == 0:
            per_class_accuracy = {
                self._class_key(class_idx): 0.0
                for class_idx in range(self.num_classes)
            }
            per_class_correct = {
                self._class_key(class_idx): 0
                for class_idx in range(self.num_classes)
            }
            per_class_total = {
                self._class_key(class_idx): 0
                for class_idx in range(self.num_classes)
            }
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "per_class_accuracy": per_class_accuracy,
                "per_class_correct": per_class_correct,
                "per_class_total": per_class_total,
            }

        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []
        per_class_accuracy: dict[str | int, float] = {}
        per_class_correct: dict[str | int, int] = {}
        per_class_total: dict[str | int, int] = {}
        valid_class_accuracies: list[float] = []
        valid_class_supports: list[float] = []

        for class_idx in range(self.num_classes):
            tp = float(self._true_positives[class_idx].item())
            fp = float(self._false_positives[class_idx].item())
            fn = float(self._false_negatives[class_idx].item())
            class_total = float(self._class_total[class_idx].item())
            class_correct = int(self._class_correct[class_idx].item())
            class_key = self._class_key(class_idx)

            precision_c = tp / (tp + fp + eps)
            recall_c = tp / (tp + fn + eps)
            f1_c = (2.0 * precision_c * recall_c) / (precision_c + recall_c + eps)

            precisions.append(precision_c)
            recalls.append(recall_c)
            f1_scores.append(f1_c)

            if class_total > 0.0:
                class_accuracy = class_correct / class_total
                valid_class_accuracies.append(float(class_accuracy))
                valid_class_supports.append(class_total)
            else:
                class_accuracy = 0.0

            per_class_accuracy[class_key] = float(class_accuracy)
            per_class_correct[class_key] = class_correct
            per_class_total[class_key] = int(class_total)

        total_support = sum(valid_class_supports)
        if total_support > 0.0:
            weighted_accuracy = sum(
                accuracy * support
                for accuracy, support in zip(
                    valid_class_accuracies,
                    valid_class_supports,
                )
            ) / total_support
        else:
            weighted_accuracy = 0.0

        balanced_accuracy = (
            float(sum(valid_class_accuracies) / len(valid_class_accuracies))
            if valid_class_accuracies
            else 0.0
        )

        return {
            "accuracy": float(weighted_accuracy),
            "balanced_accuracy": balanced_accuracy,
            "precision": float(sum(precisions) / len(precisions)),
            "recall": float(sum(recalls) / len(recalls)),
            "f1": float(sum(f1_scores) / len(f1_scores)),
            "per_class_accuracy": per_class_accuracy,
            "per_class_correct": per_class_correct,
            "per_class_total": per_class_total,
        }

    def _class_key(self, class_idx: int) -> str | int:
        if self.idx_to_label is not None:
            return self.idx_to_label[class_idx]
        return class_idx


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Return a compact printable summary of classification metrics."""
    summary = (
        f"acc {metrics['accuracy']:.4f} | "
        f"balanced_acc {metrics.get('balanced_accuracy', 0.0):.4f} | "
        f"precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f} | "
        f"f1 {metrics['f1']:.4f}"
    )
    if prefix:
        return f"{prefix}{summary}"
    return summary


def print_per_class_accuracy(metrics: dict, prefix: str) -> None:
    """Print per-class accuracy with correct/total counts."""
    per_class_accuracy = metrics["per_class_accuracy"]
    per_class_correct = metrics.get("per_class_correct", {})
    per_class_total = metrics.get("per_class_total", {})

    print(prefix)
    for class_name, accuracy in per_class_accuracy.items():
        correct = per_class_correct.get(class_name, 0)
        total = per_class_total.get(class_name, 0)
        print(f"  {class_name}: {accuracy:.4f} ({correct}/{total})")
