from __future__ import annotations

from src.models.graphsage import GraphSAGE


class RuleHeteroGNN(GraphSAGE):
    """GraphSAGE over rule-selected heterophilous neighbors."""
