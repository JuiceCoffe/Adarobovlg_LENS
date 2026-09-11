"""Neural network used to score multi-finger grasp candidates."""

import torch
import torch.nn as nn

from .ptv3 import PointTransformerV3
from .utils import dict2pts


class CountAwareMultiPooling(nn.Module):
    """Combine max, mean, and standard-deviation pooling using learned weights."""

    def __init__(self, hidden_dim):
        """Initialize pooling layers for the specified feature width."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.count_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )
        self.pool_weight = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, feats_list):
        """Pool each variable-length feature tensor into one vector."""
        batch_feats = []
        for features in feats_list:
            num_points, _ = features.shape
            max_pool = torch.max(features, dim=0)[0]
            mean_pool = torch.mean(features, dim=0)
            if num_points == 1:
                std_pool = torch.zeros_like(mean_pool)
            else:
                std_pool = torch.std(features, dim=0)

            count_tensor = torch.tensor(
                [[num_points]], dtype=torch.float32, device=features.device
            )
            count_encoding = self.count_encoder(count_tensor)
            pools_combined = torch.cat(
                [max_pool, mean_pool, std_pool], dim=0
            ).unsqueeze(0)
            weights = self.pool_weight(
                torch.cat([pools_combined, count_encoding], dim=1)
            )
            pools = torch.stack([max_pool, mean_pool, std_pool], dim=0)
            batch_feats.append(torch.sum(pools * weights.t(), dim=0))
        return torch.stack(batch_feats, dim=0)


class HandDecisionModel(nn.Module):
    """Predict one grasp-quality score from paired object and hand point features."""

    def __init__(self, runtime_config):
        """Build the checkpoint-compatible PTv3 encoder and scoring head."""
        super().__init__()
        self.point_encoder = PointTransformerV3(**runtime_config.ptv3)
        hidden_dim = runtime_config.ptv3["dec_channels"][0]
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool = CountAwareMultiPooling(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.device = runtime_config.device
        self.to(self.device)

        self.apply(self.weights_init)
        self.point_encoder.load_pretrained()
        torch.cuda.empty_cache()

    @staticmethod
    def weights_init(module):
        """Initialize trainable layers when no checkpoint is supplied."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data_dict, sigmoid=False):
        """Return logits or sigmoid scores for a batch of grasp candidates."""
        feature_dict = self.point_encoder(data_dict)
        _, features = dict2pts(feature_dict)
        pooled_features = self.dropout(self.pool(features))
        scores = self.score_layer(pooled_features).squeeze(-1)
        return torch.sigmoid(scores) if sigmoid else scores
