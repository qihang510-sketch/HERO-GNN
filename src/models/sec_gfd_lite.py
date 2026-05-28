from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class SECGFDLite(nn.Module):
        """Controlled lite heterophily-aware graph fraud detector."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.2) -> None:
            super().__init__()
            self.self_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.low_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.high_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.gate = nn.Linear(hidden_dim * 3, 3)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            low_pass = _neighbor_mean(x, edge_index)
            high_pass = _neighbor_mean_abs_diff(x, edge_index)
            self_repr = self.self_encoder(x)
            low_repr = self.low_encoder(low_pass)
            high_repr = self.high_encoder(high_pass)
            weights = torch.softmax(self.gate(torch.cat([self_repr, low_repr, high_repr], dim=1)), dim=1)
            fused = (
                weights[:, 0:1] * self_repr
                + weights[:, 1:2] * low_repr
                + weights[:, 2:3] * high_repr
            )
            return self.classifier(self.dropout(fused))


    def _neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            agg.index_add_(0, src, x[dst])
            degree.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        return agg / degree.clamp_min(1.0)


    def _neighbor_mean_abs_diff(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            agg.index_add_(0, src, torch.abs(x[src] - x[dst]))
            degree.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        return agg / degree.clamp_min(1.0)

else:

    class SECGFDLite:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("SECGFDLite requires torch. The training script can still use the sklearn fallback backend.")
