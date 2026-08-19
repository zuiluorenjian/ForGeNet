
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip
from PIL import Image


class Adapter(nn.Module):

    def __init__(self, in_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, bottleneck_dim)
        self.nonlinear = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)
        self.scale = 0.1
        self.dropout = nn.Dropout(p=0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        down = self.nonlinear(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        return up * self.scale


class AdaptedResidualAttentionBlock(nn.Module):

    def __init__(self, original_block, d_model: int, use_adaptformer: bool = True,
                 adaptformer_bottleneck_dim: int = 64):
        super().__init__()
        self.original_block = original_block

        for param in self.original_block.parameters():
            param.requires_grad = False

        self.use_adaptformer = use_adaptformer
        if use_adaptformer:
            self.adapter = Adapter(d_model, adaptformer_bottleneck_dim)
        else:
            self.adapter = None

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.original_block(x)

        if self.use_adaptformer and self.adapter is not None:
            adapter_output = self.adapter(identity)
            x = x + adapter_output

        return x


class AdaptedVisionTransformer(nn.Module):

    def __init__(self, original_vit, use_adaptformer: bool = True,
                 adaptformer_bottleneck_dim: int = 64, num_adapted_layers: int = -1):
        super().__init__()

        self.conv1 = original_vit.conv1
        self.class_embedding = original_vit.class_embedding
        self.positional_embedding = original_vit.positional_embedding
        self.ln_pre = original_vit.ln_pre
        self.ln_post = original_vit.ln_post
        self.proj = original_vit.proj

        for param in [self.conv1.parameters(), self.ln_pre.parameters(),
                     self.ln_post.parameters()]:
            for p in param:
                p.requires_grad = False

        self.class_embedding.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.proj.requires_grad = False

        d_model = original_vit.transformer.width

        original_blocks = original_vit.transformer.resblocks
        self.resblocks = nn.ModuleList()

        total_layers = len(original_blocks)
        if num_adapted_layers == -1:
            adapted_layer_indices = list(range(total_layers // 2, total_layers))
        else:
            adapted_layer_indices = list(range(max(0, total_layers - num_adapted_layers), total_layers))

        print(f"在层 {adapted_layer_indices} 中添加AdaptFormer")

        for i, block in enumerate(original_blocks):
            use_adapter_in_layer = use_adaptformer and (i in adapted_layer_indices)
            adapted_block = AdaptedResidualAttentionBlock(
                block, d_model, use_adapter_in_layer, adaptformer_bottleneck_dim
            )
            self.resblocks.append(adapted_block)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for block in self.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class AdaptFormerCLIPModel(nn.Module):

    def __init__(self, clip_model_name="ViT-L/14", num_classes=1,
                 use_adaptformer=True, adaptformer_bottleneck_dim=64,
                 num_adapted_layers=-1):
        super(AdaptFormerCLIPModel, self).__init__()

        print(f"加载CLIP模型: {clip_model_name}")
        self.original_clip, self.preprocess = clip.load(clip_model_name, device="cpu")

        if hasattr(self.original_clip.visual, 'transformer'):
            print("检测到ViT架构，集成AdaptFormer...")
            self.visual = AdaptedVisionTransformer(
                self.original_clip.visual,
                use_adaptformer=use_adaptformer,
                adaptformer_bottleneck_dim=adaptformer_bottleneck_dim,
                num_adapted_layers=num_adapted_layers
            )
        else:
            print("检测到ResNet架构，暂不支持AdaptFormer")
            self.visual = self.original_clip.visual
            for param in self.visual.parameters():
                param.requires_grad = False

        from .clip_models import CHANNELS
        self.feature_dim = CHANNELS[clip_model_name]

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x, return_feature=False):

        features = self.visual(x.type(self.visual.conv1.weight.dtype))

        if return_feature:
            return features

        return self.classifier(features)

    def get_trainable_params(self):

        trainable_params = []

        for name, param in self.visual.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))

        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                trainable_params.append((f"classifier.{name}", param))

        return trainable_params

    def print_trainable_params(self):

        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for n, p in trainable_params)
        all_params = sum(p.numel() for p in self.parameters())

        print(f"可训练参数: {total_params:,} ({total_params/all_params*100:.2f}%)")
        print(f"总参数: {all_params:,}")

        adapter_params = sum(p.numel() for n, p in trainable_params if 'adapter' in n)
        classifier_params = sum(p.numel() for n, p in trainable_params if 'classifier' in n)

        print(f"  - AdaptFormer参数: {adapter_params:,}")
        print(f"  - 分类器参数: {classifier_params:,}")

        return trainable_params


def load_adaptformer_clip_model(clip_model_name="ViT-L/14", use_adaptformer=True,
                               adaptformer_bottleneck_dim=64, num_adapted_layers=-1):

    return AdaptFormerCLIPModel(
        clip_model_name=clip_model_name,
        num_classes=1,
        use_adaptformer=use_adaptformer,
        adaptformer_bottleneck_dim=adaptformer_bottleneck_dim,
        num_adapted_layers=num_adapted_layers
    )
