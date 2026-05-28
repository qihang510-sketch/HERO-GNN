from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class DGAGNNLite(nn.Module):
        """Controlled lite dynamic grouping aggregation GNN."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_attribute_groups: int = 4,
            num_neighbor_groups: int = 4,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.num_attribute_groups = int(num_attribute_groups)
            self.num_neighbor_groups = int(num_neighbor_groups)
            self.self_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.group_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.group_attention = nn.Linear(hidden_dim, 1)
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            attribute_groups: torch.Tensor | None = None,
            relation_groups: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if attribute_groups is None:
                attribute_groups = _attribute_groups_from_features(x, self.num_attribute_groups)
            if relation_groups is None:
                relation_groups = torch.zeros(edge_index.shape[1], dtype=torch.long, device=x.device)
            relation_groups = relation_groups.to(device=x.device, dtype=torch.long)
            distance_groups = _distance_groups(x, edge_index, self.num_neighbor_groups)
            neighbor_groups = (relation_groups[: edge_index.shape[1]] + distance_groups) % self.num_neighbor_groups

            attr_context = _aggregate_by_group(
                x=x,
                edge_index=edge_index,
                group_ids=attribute_groups.to(device=x.device, dtype=torch.long),
                num_groups=self.num_attribute_groups,
                use_neighbor_group=True,
            )
            neighbor_context = _aggregate_by_group(
                x=x,
                edge_index=edge_index,
                group_ids=neighbor_groups,
                num_groups=self.num_neighbor_groups,
                use_neighbor_group=False,
            )
            groups = torch.cat([attr_context, neighbor_context], dim=1)
            group_repr = self.group_encoder(groups)
            attention = torch.softmax(self.group_attention(group_repr).squeeze(-1), dim=1).unsqueeze(-1)
            context_repr = torch.sum(group_repr * attention, dim=1)
            self_repr = self.self_encoder(x)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([self_repr, context_repr], dim=1)))
            fused = gate * self_repr + (1.0 - gate) * context_repr
            return self.classifier(self.dropout(fused))


    def _attribute_groups_from_features(x: torch.Tensor, num_groups: int) -> torch.Tensor:
        score = x.mean(dim=1)
        if score.numel() == 0 or num_groups <= 1:
            return torch.zeros(score.shape[0], dtype=torch.long, device=x.device)
        order = torch.argsort(score)
        groups = torch.zeros(score.shape[0], dtype=torch.long, device=x.device)
        ranks = torch.arange(score.shape[0], dtype=torch.long, device=x.device)
        groups[order] = torch.clamp(ranks * int(num_groups) // max(score.shape[0], 1), max=int(num_groups) - 1)
        return groups


    def _distance_groups(x: torch.Tensor, edge_index: torch.Tensor, num_groups: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=x.device)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        distance = torch.linalg.norm(x[src] - x[dst], dim=1)
        span = torch.clamp(distance.max() - distance.min(), min=1e-6)
        normalized = (distance - distance.min()) / span
        return torch.clamp((normalized * int(num_groups)).long(), max=int(num_groups) - 1)


    def _aggregate_by_group(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        group_ids: torch.Tensor,
        num_groups: int,
        use_neighbor_group: bool,
    ) -> torch.Tensor:
        out = torch.zeros((x.shape[0], int(num_groups), x.shape[1]), dtype=x.dtype, device=x.device)
        degree = torch.zeros((x.shape[0], int(num_groups), 1), dtype=x.dtype, device=x.device)
        if edge_index.numel() == 0:
            return out
        src = edge_index[0].long()
        dst = edge_index[1].long()
        edge_groups = group_ids[dst] if use_neighbor_group else group_ids
        for group in range(int(num_groups)):
            mask = edge_groups == group
            if not torch.any(mask):
                continue
            group_src = src[mask]
            group_dst = dst[mask]
            out[:, group, :].index_add_(0, group_src, x[group_dst])
            degree[:, group, :].index_add_(
                0,
                group_src,
                torch.ones((group_src.shape[0], 1), dtype=x.dtype, device=x.device),
            )
        return out / degree.clamp_min(1.0)

else:

    class DGAGNNLite:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("DGAGNNLite requires torch. The training script can still use the sklearn fallback backend.")
