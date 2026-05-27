from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class HEROGNN(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_mechanisms: int = 6,
            use_heterophily: bool = True,
            use_mechanism: bool = True,
            use_chain: bool = True,
        ) -> None:
            super().__init__()
            self.use_heterophily = use_heterophily
            self.use_mechanism = use_mechanism
            self.use_chain = use_chain
            self.target_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.homo_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.mechanism_embedding = nn.Embedding(num_mechanisms, hidden_dim)
            self.score_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
            self.chain_encoder = nn.Sequential(nn.Linear(input_dim + hidden_dim * 2, hidden_dim), nn.ReLU())
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 3), nn.Sigmoid())
            self.classifier = nn.Linear(hidden_dim * 3, output_dim)

        def encode_chain(
            self,
            chain_node_features: torch.Tensor,
            mechanism_ids: torch.Tensor,
            chain_scores: torch.Tensor,
        ) -> torch.Tensor:
            pooled = chain_node_features.mean(dim=1)
            mech = self.mechanism_embedding(mechanism_ids)
            score = self.score_encoder(chain_scores.view(-1, 1))
            return self.chain_encoder(torch.cat([pooled, mech, score], dim=1))

        def forward(
            self,
            target_features: torch.Tensor,
            homo_neighbor_features: torch.Tensor,
            chain_repr: torch.Tensor,
        ) -> torch.Tensor:
            target_repr = self.target_encoder(target_features)
            homo_repr = self.homo_encoder(homo_neighbor_features)
            if not self.use_chain:
                chain_repr = torch.zeros_like(chain_repr)
            final_repr = torch.cat([target_repr, homo_repr, chain_repr], dim=1)
            final_repr = final_repr * self.gate(final_repr)
            return self.classifier(final_repr).squeeze(-1)

else:

    class HEROGNN:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("HEROGNN requires torch. The training script can still use the sklearn fallback backend.")
