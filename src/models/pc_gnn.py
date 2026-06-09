from __future__ import annotations

from src.models.submission_base import SubmissionFeatureModel


class PCGNN(SubmissionFeatureModel):
    def __init__(self, **kwargs):
        super().__init__("pc_gnn", **kwargs)
