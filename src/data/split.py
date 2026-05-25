from __future__ import annotations

import numpy as np


def random_split(num_items: int, seed: int = 42, train: float = 0.6, val: float = 0.2) -> dict[str, np.ndarray]:
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError("Expected train > 0, val >= 0, and train + val < 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_items)
    train_end = int(num_items * train)
    val_end = train_end + int(num_items * val)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def split_ids(
    ids: list[str],
    seed: int = 42,
    train: float = 0.6,
    val: float = 0.2,
) -> dict[str, list[str]]:
    indices = random_split(len(ids), seed=seed, train=train, val=val)
    id_array = np.array(ids, dtype=object)
    return {name: id_array[index].tolist() for name, index in indices.items()}


def split_lookup(
    ids: list[str],
    seed: int = 42,
    train: float = 0.6,
    val: float = 0.2,
) -> dict[str, str]:
    splits = split_ids(ids, seed=seed, train=train, val=val)
    return {node_id: split for split, values in splits.items() for node_id in values}
