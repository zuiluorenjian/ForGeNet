"""Input-conditioned loss weighting supervised by shared-parameter gradients.

This module only controls Stage 1 training. It is stored separately from the
detector, so merged LoRA checkpoints remain compatible with Stage 2.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def select_gradient_probe(model, location):
    """Select exactly the parameters whose two objective norms are compared.

    LoRA feedback measures encoder adapter gradients, not the frozen CLIP base
    or the disposable projection head. Never silently fall back to the head.
    """
    probes = {'lora': model.lora_parameters, 'projection': model.projection_head.parameters}
    return tuple(parameter for parameter in probes[location]() if parameter.requires_grad)


def measure_gradient_norms(real_loss, fake_loss, parameters):
    """Probe two unweighted objectives on the SAME parameters, leaving .grad alone.

    Retain the original graph for the subsequent detector backward. No higher
    order derivatives are constructed. The current trainer uses single GPU FP32.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)
    gradients = [torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False,
        allow_unused=False,
    ) for loss in (real_loss, fake_loss)]
    squared_norms = torch.stack([
        torch.stack([g.detach().float().square().sum() for g in group]).sum()
        for group in gradients
    ])
    norms = squared_norms.sqrt()
    if not torch.isfinite(norms).all():
        raise FloatingPointError("Nonfinite gradient feedback")
    dot = torch.stack([
        (gr.detach().float() * gf.detach().float()).sum()
        for gr, gf in zip(*gradients)
    ]).sum()
    cosine = (dot / norms.prod().clamp_min(1e-12)).clamp(-1, 1)
    return norms.detach(), cosine.detach()


class LearnableGradientGate(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, lower=0.1, upper=0.9,
                 ema_decay=0.9, eps=1e-8):
        super().__init__()
        if not (0 < lower < 0.5 < upper < 1 and 0 <= ema_decay < 1 and eps > 0):
            raise ValueError("Invalid gate bounds, EMA decay, or epsilon")
        self.feature_dim = feature_dim
        self.lower, self.upper = lower, upper
        self.ema_decay, self.eps = ema_decay, eps
        # Two class centroids, two scalar dispersions, and real-class fraction.
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start at lambda=0.5, recovering the original batch-mean objective.
        nn.init.zeros_(self.net[-1].weight)
        initial_probability = (0.5 - lower) / (upper - lower)
        nn.init.constant_(self.net[-1].bias,
                          math.log(initial_probability / (1 - initial_probability)))
        self.register_buffer("gradient_ema", torch.zeros(2))
        self.register_buffer("ema_initialized", torch.tensor(False))

    def forward(self, encoder_features, labels):
        """One batch-level weight; features and labels include both views."""
        labels = labels.detach().reshape(-1)
        features = F.normalize(encoder_features.detach().float(), dim=-1)
        groups = [features[labels == label] for label in (0, 1)]
        if any(group.shape[0] == 0 for group in groups):
            raise ValueError("Gate prediction requires both classes")
        means = [group.mean(dim=0) for group in groups]
        dispersions = [
            (group - mean).square().sum(dim=-1).mean().reshape(1)
            for group, mean in zip(groups, means)
        ]
        fraction = features.new_tensor([groups[0].shape[0] / labels.numel()])
        statistics = torch.cat(means + dispersions + [fraction])
        prediction = self.net(statistics).squeeze(-1).sigmoid()
        return self.lower + (self.upper - self.lower) * prediction

    @torch.no_grad()
    def observe(self, norms):
        norms = norms.detach().to(self.gradient_ema)
        if not self.ema_initialized.item():
            self.gradient_ema.copy_(norms)
            self.ema_initialized.fill_(True)
        else:
            self.gradient_ema.mul_(self.ema_decay).add_(
                norms, alpha=1 - self.ema_decay,
            )

    def formula_weight(self):
        """Analytic inverse-norm baseline, using exactly the same EMA feedback."""
        real_norm, fake_norm = self.gradient_ema.detach().unbind()
        weight = (real_norm + self.eps) / (real_norm + fake_norm + 2 * self.eps)
        return weight.clamp(self.lower, self.upper)

    def balance_loss(self, fake_weight):
        if not self.ema_initialized.item():
            raise RuntimeError("Observe gradient norms before training the gate")
        real_norm, fake_norm = self.gradient_ema.detach().clone().unbind()
        difference = (1 - fake_weight) * real_norm - fake_weight * fake_norm
        return (difference / (real_norm + fake_norm + self.eps)).square()


def combine_objectives(real_loss, fake_loss, fake_weight):
    """Only detector parameters receive gradients from the contrastive objective."""
    weight = fake_weight.detach()
    return 2 * ((1 - weight) * real_loss + weight * fake_loss)
