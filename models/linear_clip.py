"""Stage-1 CLIP visual encoder with a frozen backbone and one linear head."""

import os

import torch
from torch import nn

from .clip import clip
from .clip_models import CHANNELS


class LinearCLIPModel(nn.Module):
    def __init__(self, clip_model_name='ViT-L/14', dropout=0.1):
        super().__init__()
        self.original_clip, self.preprocess = clip.load(
            clip_model_name, device='cpu',
            download_root=os.environ.get('CLIP_DOWNLOAD_ROOT'))
        self.original_clip.requires_grad_(False)
        self.original_clip.eval()
        self.model_config = dict(clip_model_name=clip_model_name, use_msfe=False, readout='cls')
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(CHANNELS[clip_model_name], 1))

    def train(self, mode=True):
        super().train(mode)
        self.original_clip.eval()
        return self

    def forward(self, images, return_feature=False):
        with torch.no_grad():
            features = self.original_clip.encode_image(images)
        return features if return_feature else self.classifier(features)

    def load_stage1_visual(self, checkpoint):
        """Load every merged visual tensor; reject missing/unmerged LoRA weights."""
        state = checkpoint.get('model_state_dict', checkpoint)
        prefix = 'encoder.model.visual.'
        visual_state = {key[len(prefix):]: value for key, value in state.items()
                        if key.startswith(prefix)}
        if not visual_state:
            raise ValueError('Stage-1 checkpoint lacks encoder.model.visual.*; refusing to fall back to original CLIP')
        self.original_clip.visual.load_state_dict(visual_state, strict=True)
        return len(visual_state)

    def print_trainable_params(self):
        trainable = [(name, parameter) for name, parameter in self.named_parameters()
                     if parameter.requires_grad]
        count = sum(parameter.numel() for _, parameter in trainable)
        total = sum(parameter.numel() for parameter in self.parameters())
        print(f'Linear-only trainable parameters: {count:,}; Total parameters: {total:,}; MSFE adapter parameters: 0')
        print('Trainable tensors: ' + ', '.join(name for name, _ in trainable))
        return trainable
