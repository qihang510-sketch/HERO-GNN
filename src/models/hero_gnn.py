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
            self.fusion_gate = nn.Linear(hidden_dim * 5, hidden_dim * 5)
            with torch.no_grad():
                self.fusion_gate.bias.zero_()
                self.fusion_gate.bias[hidden_dim * 4 :].fill_(1.0)
            self.base_classifier = nn.Linear(hidden_dim * 4, output_dim)
            self.chain_classifier = nn.Linear(hidden_dim, output_dim)
            self.chain_logit_scale = 0.2
            with torch.no_grad():
                self.chain_classifier.bias.fill_(-2.0)

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
            return_gates: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            target_repr = self.target_encoder(target_features)
            homo_repr = self.homo_encoder(homo_neighbor_features)
            if self.use_heterophily:
                hetero_repr = self.hetero_encoder(hetero_neighbor_features)
            else:
                hetero_repr = torch.zeros_like(target_repr)
            if self.use_heterophily and self.use_mechanism:
                mechanism_repr = self.mechanism_encoder(mechanism_features)
            else:
                mechanism_repr = torch.zeros_like(target_repr)
            if self.use_chain and not force_no_chain:
                chain_repr = self.chain_encoder(chain_features)
                chain_presence = (chain_features.abs().sum(dim=1, keepdim=True) > 0).to(chain_repr.dtype)
            else:
                chain_repr = torch.zeros_like(target_repr)
                chain_presence = torch.zeros((target_repr.shape[0], 1), dtype=target_repr.dtype, device=target_repr.device)
            gate_input = torch.cat([target_repr, homo_repr, hetero_repr, mechanism_repr, chain_repr], dim=1)
            target_gate, homo_gate, hetero_gate, mechanism_gate, chain_gate = torch.chunk(torch.sigmoid(self.fusion_gate(gate_input)), 5, dim=1)
            if not self.use_heterophily:
                hetero_gate = torch.zeros_like(hetero_gate)
            if not (self.use_heterophily and self.use_mechanism):
                mechanism_gate = torch.zeros_like(mechanism_gate)
            if self.use_chain and not force_no_chain:
                chain_gate = chain_gate.clamp_min(self.min_chain_gate)
                chain_gate = chain_gate * chain_presence
            else:
                chain_gate = torch.zeros_like(chain_gate)
            base_repr = torch.cat(
                [
                    target_gate * target_repr,
                    homo_gate * homo_repr,
                    hetero_gate * hetero_repr,
                    mechanism_gate * mechanism_repr,
                ],
                dim=1,
            )
            base_logits = self.base_classifier(base_repr)
            if self.use_chain and not force_no_chain:
                chain_logits = torch.nn.functional.softplus(self.chain_classifier(chain_gate * chain_repr)) * self.chain_logit_scale * chain_presence
            else:
                chain_logits = torch.zeros_like(base_logits)
            logits = (base_logits + chain_logits).squeeze(-1)
            if return_gates:
                return logits, {
                    "target_gate": target_gate,
                    "homo_gate": homo_gate,
                    "hetero_gate": hetero_gate,
                    "mechanism_gate": mechanism_gate,
                    "chain_gate": chain_gate,
                }
            return logits

else:

    class HEROGNN:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("HEROGNN requires torch. The training script can still use the sklearn fallback backend.")
