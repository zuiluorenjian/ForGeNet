"""Stage-1 contrastive encoder with Q/V LoRA and a trainable projection head."""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from models import get_model


class QVLoRAParametrization(nn.Module):
    """Low-rank updates for the Q and V slices of MHA in_proj_weight."""

    def __init__(self, embed_dim, rank=8, alpha=16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.embed_dim = embed_dim
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.q_A = nn.Parameter(torch.empty(rank, embed_dim))
        self.q_B = nn.Parameter(torch.zeros(embed_dim, rank))
        self.v_A = nn.Parameter(torch.empty(rank, embed_dim))
        self.v_B = nn.Parameter(torch.zeros(embed_dim, rank))
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))

    def forward(self, base_weight):
        q_delta = (self.q_B @ self.q_A) * self.scaling
        v_delta = (self.v_B @ self.v_A) * self.scaling
        k_delta = torch.zeros_like(q_delta)
        delta = torch.cat((q_delta, k_delta, v_delta), dim=0)
        return base_weight + delta.to(dtype=base_weight.dtype)


class PretrainModel(nn.Module):
    """CLIP encoder with frozen base weights, optional Q/V LoRA, and a trainable contrastive projection head."""

    def __init__(self, arch="CLIP:ViT-B/32", output_dim=128, hidden_dim=2048,
                 batch_norm=True, use_lora=False, lora_rank=8,
                 lora_alpha=16.0, lora_layers=1):
        """Initialize the CLIP encoder and projection head."""
        super(PretrainModel, self).__init__()

        # Load CLIP and freeze its base weights.
        self.encoder = get_model(arch)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.use_lora = False
        self.lora_layer_indices = []

        # Read the CLIP output dimension from CHANNELS.
        from models.clip_models import CHANNELS

        clip_name = arch.replace("CLIP:", "")
        if clip_name in CHANNELS:
            self.encoder_dim = CHANNELS[clip_name]
        else:
            # Infer the feature dimension as a fallback.
            print(f"Warning: unknown CLIP architecture {clip_name}, inferring the feature dimension...")
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.encoder(dummy_input, return_feature=True)
                self.encoder_dim = features.shape[1]

        print(f"Encoder feature dimension: {self.encoder_dim}")

        # Create the contrastive MLP projection head.
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

        if use_lora:
            self.enable_lora(lora_rank, lora_alpha, lora_layers)


    def forward(self, x, return_features=False):
        """Return normalized projections, optionally together with encoder features."""
        # Keep the encoder graph for LoRA; otherwise use a frozen feature extractor.
        if self.use_lora:
            features = self.encoder(x, return_feature=True)
        else:
            with torch.no_grad():
                features = self.encoder(x, return_feature=True)

        # Apply the projection head.
        projection = self.projection_head(features)

        # L2-normalize the projections.
        projection = F.normalize(projection, dim=1)

        if return_features:
            return features, projection
        return projection

    def get_encoder_features(self, x):
        """Extract encoder features without constructing a gradient graph."""
        with torch.no_grad():
            features = self.encoder(x, return_feature=True)
        return features

    def freeze_all(self):
        """Freeze all parameters and switch to evaluation mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze_projection_head(self):
        """Enable training for the contrastive projection head."""
        for param in self.projection_head.parameters():
            param.requires_grad = True
        self.projection_head.train()


    def enable_lora(self, rank=8, alpha=16.0, num_layers=1):
        visual = getattr(self.encoder.model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise ValueError("LoRA mode currently supports CLIP ViT backbones only")

        total_layers = len(blocks)
        if num_layers == -1:
            start_layer = 0
        elif 0 < num_layers <= total_layers:
            start_layer = total_layers - num_layers
        else:
            raise ValueError(
                f"lora_layers must be -1 or within [1, {total_layers}], got {num_layers}"
            )

        self.lora_layer_indices = list(range(start_layer, total_layers))
        for layer_idx in self.lora_layer_indices:
            attention = blocks[layer_idx].attn
            if attention.in_proj_weight is None:
                raise ValueError("Expected MultiheadAttention with a combined in_proj_weight")
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                QVLoRAParametrization(
                    embed_dim=attention.embed_dim,
                    rank=rank,
                    alpha=alpha,
                ),
            )

        self.use_lora = True
        print(
            f"LoRA enabled on ViT layers {self.lora_layer_indices}: "
            f"Q/V, rank={rank}, alpha={alpha}"
        )

    def lora_parameters(self):
        if not self.use_lora:
            return []
        blocks = self.encoder.model.visual.transformer.resblocks
        parameters = []
        for layer_idx in self.lora_layer_indices:
            adapter = blocks[layer_idx].attn.parametrizations.in_proj_weight[0]
            parameters.extend(adapter.parameters())
        return parameters

    def lora_state_dict(self):
        if not self.use_lora:
            return OrderedDict()
        return OrderedDict(
            (key, value.detach())
            for key, value in self.state_dict().items()
            if ".parametrizations.in_proj_weight.0." in key
        )

    def merged_state_dict(self):
        """Return a standard checkpoint with LoRA deltas merged into ViT weights."""
        if not self.use_lora:
            return self.state_dict()

        merged = OrderedDict()
        for key, value in self.state_dict().items():
            if ".parametrizations.in_proj_weight." not in key:
                merged[key] = value.detach()

        blocks = self.encoder.model.visual.transformer.resblocks
        for layer_idx in self.lora_layer_indices:
            key = (
                "encoder.model.visual.transformer.resblocks."
                f"{layer_idx}.attn.in_proj_weight"
            )
            merged[key] = blocks[layer_idx].attn.in_proj_weight.detach()
        return merged
