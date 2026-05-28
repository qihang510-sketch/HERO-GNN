from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class FLAGLite(nn.Module):
        """Controlled lite semantic-neighbor enhanced fraud detector."""

        def __init__(
            self,
            input_dim: int,
            text_dim: int = 0,
            hidden_dim: int = 64,
            output_dim: int = 1,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.text_dim = int(text_dim)
            enhanced_dim = input_dim + self.text_dim + input_dim
            self.net = nn.Sequential(
                nn.Linear(enhanced_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            text_features: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if text_features is None:
                text_features = torch.zeros((x.shape[0], self.text_dim), dtype=x.dtype, device=x.device)
            semantic_neighbor_mean = _neighbor_mean(x, edge_index)
            enhanced = torch.cat([x, text_features.to(dtype=x.dtype, device=x.device), semantic_neighbor_mean], dim=1)
            return self.net(enhanced)


    def _neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() > 0:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            agg.index_add_(0, src, x[dst])
            degree.index_add_(0, src, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        return agg / degree.clamp_min(1.0)

else:

    class FLAGLite:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("FLAGLite requires torch. The training script can still use the sklearn fallback backend.")
