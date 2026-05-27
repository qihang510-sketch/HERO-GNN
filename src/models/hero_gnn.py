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
            hetero_input_dim: int | None = None,
            mechanism_input_dim: int | None = None,
            chain_input_dim: int | None = None,
            min_chain_gate: float = 0.05,
        ) -> None:
            super().__init__()
            self.use_heterophily = use_heterophily
            self.use_mechanism = use_mechanism
            self.use_chain = use_chain
            self.min_chain_gate = float(min_chain_gate)
            self.target_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.homo_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            hetero_input_dim = input_dim if hetero_input_dim is None else int(hetero_input_dim)
            mechanism_input_dim = num_mechanisms if mechanism_input_dim is None else int(mechanism_input_dim)
            self.hetero_encoder = nn.Sequential(nn.Linear(hetero_input_dim, hidden_dim), nn.ReLU())
            self.mechanism_encoder = nn.Sequential(nn.Linear(mechanism_input_dim, hidden_dim), nn.ReLU())
            self.mechanism_embedding = nn.Embedding(num_mechanisms, hidden_dim)
            self.score_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
            self.chain_component_encoder = nn.Sequential(nn.Linear(input_dim + hidden_dim * 2, hidden_dim), nn.ReLU())
            chain_input_dim = hidden_dim if chain_input_dim is None else int(chain_input_dim)
            if chain_input_dim == hidden_dim:
                self.chain_encoder = nn.Identity()
            else:
                self.chain_encoder = nn.Sequential(nn.Linear(chain_input_dim, hidden_dim), nn.ReLU())
            self.classifier = nn.Linear(hidden_dim * 5, output_dim)

        def encode_chain(
            self,
            chain_node_features: torch.Tensor,
            mechanism_ids: torch.Tensor,
            chain_scores: torch.Tensor,
        ) -> torch.Tensor:
            pooled = chain_node_features.mean(dim=1)
            mech = self.mechanism_embedding(mechanism_ids)
            score = self.score_encoder(chain_scores.view(-1, 1))
            return self.chain_component_encoder(torch.cat([pooled, mech, score], dim=1))

        def forward(
            self,
            target_features: torch.Tensor,
            homo_neighbor_features: torch.Tensor,
            hetero_neighbor_features: torch.Tensor,
            mechanism_features: torch.Tensor,
            chain_features: torch.Tensor,
            force_no_chain: bool = False,
            zero_hetero: bool = False,
            zero_mechanism: bool = False,
            zero_chain: bool = False,
            return_gates: bool = False,
            return_details: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            target_repr = self.target_encoder(target_features)
            homo_repr = self.homo_encoder(homo_neighbor_features)
            if self.use_heterophily and not zero_hetero:
                hetero_repr = self.hetero_encoder(hetero_neighbor_features)
            else:
                hetero_repr = torch.zeros_like(target_repr)
            if self.use_heterophily and self.use_mechanism and not zero_mechanism:
                mechanism_repr = self.mechanism_encoder(mechanism_features)
            else:
                mechanism_repr = torch.zeros_like(target_repr)
            if self.use_chain and not force_no_chain and not zero_chain:
                chain_repr = self.chain_encoder(chain_features)
                chain_presence = (chain_features.abs().sum(dim=1, keepdim=True) > 0).to(chain_repr.dtype)
                chain_repr = chain_repr * chain_presence
            else:
                chain_repr = torch.zeros_like(target_repr)
                chain_presence = torch.zeros((target_repr.shape[0], 1), dtype=target_repr.dtype, device=target_repr.device)
            final_repr = torch.cat(
                [
                    target_repr,
                    homo_repr,
                    hetero_repr,
                    mechanism_repr,
                    chain_repr,
                ],
                dim=1,
            )
            logits = self.classifier(final_repr).squeeze(-1)
            gates = {
                "target_gate": torch.ones_like(target_repr),
                "homo_gate": torch.ones_like(homo_repr),
                "hetero_gate": torch.ones_like(hetero_repr) if self.use_heterophily and not zero_hetero else torch.zeros_like(hetero_repr),
                "mechanism_gate": torch.ones_like(mechanism_repr)
                if self.use_heterophily and self.use_mechanism and not zero_mechanism
                else torch.zeros_like(mechanism_repr),
                "chain_gate": torch.ones_like(chain_repr) * chain_presence if self.use_chain and not force_no_chain and not zero_chain else torch.zeros_like(chain_repr),
            }
            if return_details:
                return logits, {
                    **gates,
                    "target_repr": target_repr,
                    "homo_repr": homo_repr,
                    "hetero_repr": hetero_repr,
                    "mechanism_repr": mechanism_repr,
                    "chain_repr": chain_repr,
                    "final_repr": final_repr,
                }
            if return_gates:
                return logits, gates
            return logits

else:

    class HEROGNN:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("HEROGNN requires torch. The training script can still use the sklearn fallback backend.")
