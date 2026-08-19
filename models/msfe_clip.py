
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from .clip import clip
from PIL import Image


class MSFEOp(nn.Module):
    """Fuse depthwise-convolution features from multiple receptive fields."""

    def __init__(self, channels: int, kernel_sizes: Tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        self.depthwise = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels)
            for k in kernel_sizes
        ])

        reduction = 16
        hidden_dim = max(channels // reduction, self.num_kernels)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.num_kernels, 1),
            nn.Softmax(dim=1)
        )

        self.projector = nn.Conv2d(channels, channels, kernel_size=1)

        self._initialize_attention()

    def _initialize_attention(self):

        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        scale_weights = self.attention(x)  # (B, num_kernels, 1, 1)

        multi_scale_features = [conv(x) for conv in self.depthwise]

        accum = sum(scale_weights[:, i:i+1] * multi_scale_features[i]
                   for i in range(self.num_kernels))

        x = accum + identity
        x = x + self.projector(x)
        return x

    def get_scale_weights(self, x: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            weights = self.attention(x).squeeze(-1).squeeze(-1)  # (B, num_kernels)
        return weights


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class MSFEAdapter(nn.Module):
    """Apply MSFE in a bottleneck branch over transformer patch tokens."""

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            kernel_sizes: Tuple[int, ...] = (3, 5, 7),
            dropout: float = 0.1,
            scale: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.scale = scale

        self.norm = LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.ones(embed_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(embed_dim))

        self.project1 = nn.Linear(embed_dim, hidden_dim)
        self.nonlinear = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project2 = nn.Linear(hidden_dim, embed_dim)

        self.msfe_op = MSFEOp(hidden_dim, kernel_sizes)

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(self.project1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.project1.bias)
        nn.init.zeros_(self.project2.weight)
        nn.init.zeros_(self.project2.bias)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int] = (16, 16)) -> torch.Tensor:

        if x.size(0) <= 1:
            return torch.zeros_like(x)

        cls_token = x[:1]  # (1, B, C)
        patch_tokens = x[1:]  # (L-1, B, C)

        batch_size = patch_tokens.shape[1]
        num_tokens = patch_tokens.shape[0]
        expected_tokens = hw_shape[0] * hw_shape[1]

        if expected_tokens != num_tokens:
            side = int(math.sqrt(num_tokens))
            if side * side != num_tokens:
                return torch.zeros_like(x)
            hw_shape = (side, side)
            expected_tokens = num_tokens

        if expected_tokens == 0:
            return torch.zeros_like(x)

        h, w = hw_shape

        identity = patch_tokens.permute(1, 0, 2).contiguous()  # B, N, C

        normed = self.norm(identity)
        gamma = self.gamma.to(dtype=identity.dtype, device=identity.device)
        gammax = self.gammax.to(dtype=identity.dtype, device=identity.device)
        normed = normed * gamma + identity * gammax

        hidden = self.project1(normed)  # B, N, hidden_dim

        hidden = hidden.permute(0, 2, 1).contiguous().view(batch_size, self.hidden_dim, h, w)
        hidden = self.msfe_op(hidden)
        hidden = hidden.view(batch_size, self.hidden_dim, -1).permute(0, 2, 1).contiguous()

        hidden = self.nonlinear(hidden)
        hidden = self.dropout(hidden)
        hidden = self.project2(hidden) * self.scale

        adapter_output = hidden.permute(1, 0, 2).contiguous()

        cls_output = torch.zeros_like(cls_token)

        result = torch.cat([cls_output, adapter_output], dim=0)
        return result.to(dtype=x.dtype)


class MSFEResidualAttentionBlock(nn.Module):
    """Wrap a frozen CLIP attention block with a trainable MSFE branch."""

    def __init__(self, original_block, d_model: int, use_msfe: bool = True,
                 msfe_hidden_dim: int = 128, msfe_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
                 msfe_dropout: float = 0.1, msfe_scale: float = 0.1):
        super().__init__()
        self.original_block = original_block

        for param in self.original_block.parameters():
            param.requires_grad = False

        self.use_msfe = use_msfe
        if use_msfe:
            self.msfe_adapter = MSFEAdapter(
                embed_dim=d_model,
                hidden_dim=msfe_hidden_dim,
                kernel_sizes=msfe_kernel_sizes,
                dropout=msfe_dropout,
                scale=msfe_scale
            )
        else:
            self.msfe_adapter = None

    def forward(self, x: torch.Tensor):
        identity = x
        x = x + self.original_block.attn(
            self.original_block.ln_1(x),
            self.original_block.ln_1(x),
            self.original_block.ln_1(x),
            need_weights=False,
            attn_mask=self.original_block.attn_mask
        )[0]

        mlp_identity = x
        x = mlp_identity + self.original_block.mlp(self.original_block.ln_2(x))

        if self.use_msfe and self.msfe_adapter is not None:
            msfe_output = self.msfe_adapter(mlp_identity)
            x = x + msfe_output

        return x


class MSFEVisionTransformer(nn.Module):
    """CLIP vision transformer with MSFE adapters in selected blocks."""

    def __init__(self, original_vit, use_msfe: bool = True,
                 msfe_hidden_dim: int = 128, msfe_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
                 msfe_dropout: float = 0.1, msfe_scale: float = 0.1,
                 num_adapted_layers: int = -1):
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
            adapted_layer_indices = list(range(total_layers))
        else:
            adapted_layer_indices = list(range(max(0, total_layers - num_adapted_layers), total_layers))

        print(f"Adding MSFE adapters to layers: {adapted_layer_indices}")

        for i, block in enumerate(original_blocks):
            use_msfe_in_layer = use_msfe and (i in adapted_layer_indices)
            adapted_block = MSFEResidualAttentionBlock(
                block, d_model, use_msfe_in_layer, msfe_hidden_dim,
                msfe_kernel_sizes, msfe_dropout, msfe_scale
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


class MSFECLIPModel(nn.Module):
    """Binary CLIP classifier with a frozen backbone and MSFE adapters."""

    def __init__(self, clip_model_name="ViT-L/14", num_classes=1,
                 use_msfe=True, msfe_hidden_dim=128, msfe_kernel_sizes=(3, 5, 7),
                 msfe_dropout=0.1, msfe_scale=0.1, num_adapted_layers=-1):
        super(MSFECLIPModel, self).__init__()

        print(f"Loading CLIP model: {clip_model_name}")
        self.original_clip, self.preprocess = clip.load(clip_model_name, device="cpu")

        if hasattr(self.original_clip.visual, 'transformer'):
            print("Detected a ViT backbone; enabling MSFE adapters.")
            self.visual = MSFEVisionTransformer(
                self.original_clip.visual,
                use_msfe=use_msfe,
                msfe_hidden_dim=msfe_hidden_dim,
                msfe_kernel_sizes=msfe_kernel_sizes,
                msfe_dropout=msfe_dropout,
                msfe_scale=msfe_scale,
                num_adapted_layers=num_adapted_layers
            )
        else:
            print("Detected a ResNet backbone; MSFE adapters are not supported.")
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

        msfe_params = sum(p.numel() for n, p in trainable_params if 'msfe' in n)
        classifier_params = sum(p.numel() for n, p in trainable_params if 'classifier' in n)

        print(f"  - MSFE适配器参数: {msfe_params:,}")
        print(f"  - 分类器参数: {classifier_params:,}")

        return trainable_params


def load_msfe_clip_model(clip_model_name="ViT-L/14", use_msfe=True,
                        msfe_hidden_dim=128, msfe_kernel_sizes=(3, 5, 7),
                        msfe_dropout=0.1, msfe_scale=0.1, num_adapted_layers=-1):

    return MSFECLIPModel(
        clip_model_name=clip_model_name,
        num_classes=1,
        use_msfe=use_msfe,
        msfe_hidden_dim=msfe_hidden_dim,
        msfe_kernel_sizes=msfe_kernel_sizes,
        msfe_dropout=msfe_dropout,
        msfe_scale=msfe_scale,
        num_adapted_layers=num_adapted_layers
    )

















