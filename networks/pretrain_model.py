
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model


class PretrainModel(nn.Module):


    def __init__(self, arch="CLIP:ViT-B/32", output_dim=128, hidden_dim=2048, batch_norm=True):

        super(PretrainModel, self).__init__()

        self.encoder = get_model(arch)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        from models.clip_models import CHANNELS

        clip_name = arch.replace("CLIP:", "")
        if clip_name in CHANNELS:
            self.encoder_dim = CHANNELS[clip_name]
        else:
            print(f"警告：未知的CLIP架构 {clip_name}，尝试动态获取维度...")
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.encoder(dummy_input, return_feature=True)
                self.encoder_dim = features.shape[1]

        print(f"编码器维度: {self.encoder_dim}")

        if batch_norm:
            self.projection_head = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x, return_features=False):

        with torch.no_grad():
            features = self.encoder(x, return_feature=True)

        projection = self.projection_head(features)

        projection = F.normalize(projection, dim=1)

        if return_features:
            return features, projection
        return projection

    def get_encoder_features(self, x):

        with torch.no_grad():
            features = self.encoder(x, return_feature=True)
        return features

    def freeze_all(self):

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze_projection_head(self):

        for param in self.projection_head.parameters():
            param.requires_grad = True
        self.projection_head.train()
