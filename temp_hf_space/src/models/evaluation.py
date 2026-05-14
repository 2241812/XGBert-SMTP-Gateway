import json
import os
import random
from datetime import datetime
from typing import Iterable


def create_or_load_fixed_eval_indices(
    labels: Iterable[int],
    output_path: str,
    max_samples: int,
    num_classes: int = 4,
    seed: int = 42,
) -> list[int]:
    """Create or load a deterministic, stratified fixed evaluation split."""
    label_list = [int(label) for label in labels]
    if not label_list:
        return []

    existing = _load_existing_indices(output_path, len(label_list))
    if existing is not None:
        return existing

    target_size = min(max_samples, len(label_list))
    if target_size <= 0:
        return []

    rng = random.Random(seed)
    sampled: list[int] = []
    per_class = max(1, target_size // max(num_classes, 1))

    for class_id in range(num_classes):
        class_indices = [idx for idx, label in enumerate(label_list) if label == class_id]
        if not class_indices:
            continue
        take_count = min(len(class_indices), per_class)
        sampled.extend(rng.sample(class_indices, take_count))

    sampled_set = set(sampled)
    if len(sampled) < target_size:
        remaining = [idx for idx in range(len(label_list)) if idx not in sampled_set]
        fill_count = min(target_size - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, fill_count))

    rng.shuffle(sampled)
    _save_indices(output_path, sampled, len(label_list), seed, target_size)
    return sampled


def _load_existing_indices(output_path: str, dataset_size: int) -> list[int] | None:
    if not os.path.exists(output_path):
        return None

    with open(output_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    indices = payload.get("indices", [])
    if not isinstance(indices, list):
        return None

    if all(isinstance(idx, int) and 0 <= idx < dataset_size for idx in indices):
        return indices
    return None


def _save_indices(output_path: str, indices: list[int], dataset_size: int, seed: int, target_size: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_size": dataset_size,
        "target_size": target_size,
        "seed": seed,
        "indices": indices,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
