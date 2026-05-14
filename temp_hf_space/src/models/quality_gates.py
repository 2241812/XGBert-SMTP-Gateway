from typing import Iterable

from sklearn.metrics import recall_score


def compute_per_class_recall(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int = 4) -> dict[int, float]:
    """Compute recall for each class index."""
    labels = list(range(num_classes))
    recalls = recall_score(list(y_true), list(y_pred), labels=labels, average=None, zero_division=0)
    return {label: float(recalls[label]) for label in labels}


def evaluate_quality_gates(per_class_recall: dict[int, float], min_recalls: dict[int, float]) -> dict:
    """Evaluate whether minimum class-recall constraints are met."""
    checks: dict[str, dict[str, float | bool]] = {}
    passed = True

    for label, threshold in min_recalls.items():
        actual = float(per_class_recall.get(int(label), 0.0))
        meets = actual >= float(threshold)
        checks[str(label)] = {
            "minimum": float(threshold),
            "actual": actual,
            "passed": meets,
        }
        passed = passed and meets

    return {"passed": passed, "checks": checks}
