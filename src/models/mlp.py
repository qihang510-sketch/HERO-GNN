from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in environments without torch.
    torch = None
    nn = None


if nn is not None:

    class MLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class MLP:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("MLP requires torch. Install requirements.txt to use the torch model backend.")
