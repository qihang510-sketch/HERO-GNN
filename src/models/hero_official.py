from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class HEROOfficial(nn.Module):
        def __init__(
            self,
            input_dim: int,
            relation_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            use_hetero: bool = True,
            use_relation: bool = True,
            use_feature_deviation: bool = True,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.use_hetero = bool(use_hetero)
            self.use_relation = bool(use_relation)
            self.use_feature_deviation = bool(use_feature_deviation)
            self.target_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.homo_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.hetero_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.feature_deviation_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.relation_encoder = nn.Sequential(nn.Linear(max(int(relation_dim), 1), hidden_dim), nn.ReLU())
            self.hetero_gate = _BranchGate(hidden_dim)
            self.feature_deviation_gate = _BranchGate(hidden_dim)
            self.relation_gate = _BranchGate(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 5, output_dim)

        def forward(
            self,
            target_features: torch.Tensor,
            relation_homo_features: torch.Tensor,
            relation_hetero_features: torch.Tensor,
            feature_deviation_features: torch.Tensor,
            relation_type_features: torch.Tensor,
            return_details: bool = False,
        ):
            target_repr = self.target_encoder(target_features)
            homo_repr = self.homo_encoder(relation_homo_features)
            raw_hetero_repr = self.hetero_encoder(relation_hetero_features)
            raw_feature_deviation_repr = self.feature_deviation_encoder(feature_deviation_features)
            raw_relation_repr = self.relation_encoder(relation_type_features)

            if self.use_hetero:
                hetero_gate = self.hetero_gate(raw_hetero_repr)
                hetero_repr = raw_hetero_repr * hetero_gate
            else:
                hetero_gate = torch.zeros((target_repr.shape[0], 1), dtype=target_repr.dtype, device=target_repr.device)
                hetero_repr = torch.zeros_like(target_repr)
            if self.use_feature_deviation:
                feature_deviation_gate = self.feature_deviation_gate(raw_feature_deviation_repr)
                feature_deviation_repr = raw_feature_deviation_repr * feature_deviation_gate
            else:
                feature_deviation_gate = torch.zeros((target_repr.shape[0], 1), dtype=target_repr.dtype, device=target_repr.device)
                feature_deviation_repr = torch.zeros_like(target_repr)
            if self.use_relation:
                relation_gate = self.relation_gate(raw_relation_repr)
                relation_repr = raw_relation_repr * relation_gate
            else:
                relation_gate = torch.zeros((target_repr.shape[0], 1), dtype=target_repr.dtype, device=target_repr.device)
                relation_repr = torch.zeros_like(target_repr)

            final_repr = torch.cat(
                [target_repr, homo_repr, hetero_repr, feature_deviation_repr, relation_repr],
                dim=1,
            )
            logits = self.classifier(self.dropout(final_repr)).squeeze(-1)
            if return_details:
                return logits, {
                    "official_hetero_gate": hetero_gate,
                    "feature_deviation_gate": feature_deviation_gate,
                    "relation_gate": relation_gate,
                    "final_repr": final_repr,
                }
            return logits


    class _BranchGate(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.linear = nn.Linear(hidden_dim, 1)
            nn.init.constant_(self.linear.bias, -1.0)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(self.norm(values)))

else:

    class HEROOfficial:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("HEROOfficial requires torch.")
