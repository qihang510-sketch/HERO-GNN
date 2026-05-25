from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class GraphSAGELayer(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim * 2, output_dim)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            agg = torch.zeros_like(x)
            degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            if edge_index.numel() > 0:
                src = edge_index[0].long()
                dst = edge_index[1].long()
                agg.index_add_(0, src, x[dst])
                degree.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
            agg = agg / degree.clamp_min(1.0)
            return self.linear(torch.cat([x, agg], dim=1))


    class GraphSAGE(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
            self.layer2 = GraphSAGELayer(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            h = self.activation(self.layer1(x, edge_index))
            h = self.dropout(h)
            h = self.activation(self.layer2(h, edge_index))
            h = self.dropout(h)
            return self.classifier(h)

else:

    class GraphSAGE:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("GraphSAGE requires torch. Install requirements.txt to use the torch model backend.")
