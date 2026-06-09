from __future__ import annotations

from src.models.submission_base import SubmissionFeatureModel


class CAREGNN(SubmissionFeatureModel):
    def __init__(self, **kwargs):
        super().__init__("care_gnn", **kwargs)
