import torch
import torch.nn as nn

from src.models.backbones import build_backbone

torch.manual_seed(0)

class Baseline(nn.Module):
    def __init__(
        self,
        base_model: dict,
        num_classes: int,
        hidden_dim: int,
        device: str = "cuda",
    ):
        super().__init__()

        self.model_ax, self.feature_dim = build_backbone(base_model)
        self.model_front, _ = build_backbone(base_model)
        self.model_sag, _ = build_backbone(base_model)

        self.clf_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        self.to(device)

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError

    def _branch_features(self, ax, front, sag):
        return (
            self.model_ax(ax),
            self.model_front(front),
            self.model_sag(sag),
        )


class MultiBranchAttention(Baseline):
    def __init__(
        self,
        base_model: dict,
        hidden_dim: int,
        num_classes: int,
        attention_dim: int,
        attention_heads: int,
        aggregation: str = "mean",
        return_attention_scores: bool = False,
        device: str = "cuda",
    ):
        super().__init__(base_model, num_classes, hidden_dim, device=device)

        self.aggregation = aggregation

        if aggregation == "cat":
            self.clf_head[0] = nn.Linear(3 * self.feature_dim, hidden_dim, device=device)

        self.return_attention_scores = return_attention_scores

        self.input_projection = nn.Linear(self.feature_dim, attention_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=0.3,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, self.feature_dim)
        self.to(device)

    def forward(
        self,
        ax: torch.Tensor,
        front: torch.Tensor,
        sag: torch.Tensor,
    ) -> torch.Tensor:
        ax_features, front_features, sag_features = self._branch_features(ax, front, sag)

        features = torch.stack([ax_features, front_features, sag_features], dim=1)

        projected = self.input_projection(features)
        output, scores = self.multi_head_attention(
            projected,
            projected,
            projected,
            need_weights=self.return_attention_scores,
            average_attn_weights=False,
        )
        attention_outputs = self.output_projection(output) + features
        ax_attention_features, front_attention_features, sag_attention_features = (
            attention_outputs.unbind(dim=1)
        )

        if self.aggregation == "cat":
            attention_features = torch.cat(
                [ax_attention_features, front_attention_features, sag_attention_features],
                dim=1,
            )
        elif self.aggregation == "mean":
            attention_features = 1 / 3 * (
                ax_attention_features + front_attention_features + sag_attention_features
            )
        else:
            raise RuntimeError(
                f"Wrong aggregation mode: should be 'mean' or 'cat' got {self.aggregation}"
            )

        if self.return_attention_scores:
            return self.clf_head(attention_features), tuple(scores.unbind(dim=2))

        return self.clf_head(attention_features)


class SingleBranch(Baseline):
    def __init__(self, base_model, num_classes, hidden_dim, device: str = "cuda") -> None:
        super().__init__(base_model, num_classes, hidden_dim, device=device)
        del self.model_front, self.model_sag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model_ax(x)

        return self.clf_head(features)


class MultiBranchConcat(Baseline):
    def __init__(self, base_model, num_classes, hidden_dim, device: str = "cuda") -> None:
        super().__init__(base_model, num_classes, hidden_dim, device=device)
        self.clf_head[0] = nn.Linear(3 * self.feature_dim, hidden_dim, device=device)

    def forward(
        self,
        ax: torch.Tensor,
        front: torch.Tensor,
        sag: torch.Tensor,
    ) -> torch.Tensor:
        ax_features, front_features, sag_features = self._branch_features(ax, front, sag)

        features = torch.cat([ax_features, front_features, sag_features], dim=1)

        return self.clf_head(features)


class MultiBranchMean(Baseline):
    def forward(
        self,
        ax: torch.Tensor,
        front: torch.Tensor,
        sag: torch.Tensor,
    ) -> torch.Tensor:
        ax_features, front_features, sag_features = self._branch_features(ax, front, sag)

        features = (ax_features + front_features + sag_features) / 3.0

        return self.clf_head(features)
