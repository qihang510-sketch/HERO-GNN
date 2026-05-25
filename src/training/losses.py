from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def classification_loss(logits, labels):
    if nn is None:
        raise ImportError("classification_loss requires torch.")
    return nn.CrossEntropyLoss()(logits, labels)


def binary_classification_loss(logits, labels):
    if nn is None:
        raise ImportError("binary_classification_loss requires torch.")
    return nn.BCEWithLogitsLoss()(logits.float(), labels.float())


def hero_loss(
    pred_logits,
    labels,
    mechanism_logits=None,
    mechanism_labels=None,
    selected_chain_count=None,
    lambda_mech: float = 0.1,
    lambda_sparse: float = 0.01,
):
    if nn is None or torch is None:
        raise ImportError("hero_loss requires torch.")
    loss = binary_classification_loss(pred_logits, labels)
    if mechanism_logits is not None and mechanism_labels is not None:
        loss = loss + lambda_mech * nn.CrossEntropyLoss()(mechanism_logits, mechanism_labels.long())
    if selected_chain_count is not None:
        loss = loss + lambda_sparse * selected_chain_count.float().mean()
    return loss
