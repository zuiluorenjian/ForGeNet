"""Reuse Stage-2 optimization/checkpoint handling with an actual adapter-free CLIP."""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from models.linear_clip import LinearCLIPModel
from .msfe_trainer import MSFEFinetuneTrainer


class LinearFinetuneTrainer(MSFEFinetuneTrainer):
    def __init__(self, opt, pretrain_model_path):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = 0
        self.model = LinearCLIPModel(opt.arch.replace('CLIP:', ''))
        checkpoint = torch.load(pretrain_model_path, map_location='cpu',
                                weights_only=False, mmap=True)
        count = self.model.load_stage1_visual(checkpoint)
        del checkpoint
        print(f'Strictly loaded merged Stage-1 CLIP visual tensors: {count} tensors; source: {pretrain_model_path}')
        self.model.to(self.device)
        trainable = self.model.print_trainable_params()
        if {name for name, _ in trainable} != {'classifier.1.weight', 'classifier.1.bias'}:
            raise RuntimeError('The linear-only ablation must train only classifier weight and bias')
        self.criterion = nn.BCEWithLogitsLoss()
        parameters = [parameter for _, parameter in trainable]
        if opt.optim == 'adam':
            self.optimizer = optim.Adam(parameters, lr=opt.lr,
                                        betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = optim.SGD(parameters, lr=opt.lr,
                                       momentum=0.9, weight_decay=opt.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {opt.optim}')
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.7)
        self.input = self.label = self.loss = self.output = None

    def forward(self):
        self.output = self.model(self.input).flatten()

    def train(self):
        self.model.train()
