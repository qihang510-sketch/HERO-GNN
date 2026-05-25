from __future__ import annotations

from src.models.graphsage import GraphSAGE


class SemSimGNN(GraphSAGE):
    """GraphSAGE over semantic-similarity-filtered neighbors."""

