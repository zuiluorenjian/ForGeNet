import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model
from networks.pretrain_model import PretrainModel

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time
            time.sleep(3)
            params = self.model.parameters()



        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class FinetuneTrainer(BaseModel):


    def name(self):
        return 'FinetuneTrainer'

    def __init__(self, opt, pretrain_model_path):

        super(FinetuneTrainer, self).__init__(opt)
        self.opt = opt

        print(f"加载预训练模型: {pretrain_model_path}")
        checkpoint = torch.load(pretrain_model_path, map_location='cpu', weights_only=False)

        self.pretrain_model = PretrainModel(
            arch=opt.arch if hasattr(opt, 'arch') else 'CLIP:ViT-L/14',
            output_dim=128,
            hidden_dim=768,
            batch_norm=True
        )

        if 'model_state_dict' in checkpoint:
            self.pretrain_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.pretrain_model.load_state_dict(checkpoint)

        self.pretrain_model.freeze_all()

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.pretrain_model.encoder_dim, 1)
        )

        torch.nn.init.normal_(self.classifier[1].weight.data, 0.0, opt.init_gain)
        torch.nn.init.constant_(self.classifier[1].bias.data, 0.0)

        class FinetuneModel(nn.Module):
            def __init__(self, pretrain_model, classifier):
                super().__init__()
                self.pretrain_model = pretrain_model
                self.classifier = classifier

            def forward(self, x):
                features = self.pretrain_model.get_encoder_features(x)
                return self.classifier(features)

        self.model = FinetuneModel(self.pretrain_model, self.classifier)

        params = list(self.classifier.parameters())

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay
            )
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model.to(opt.gpu_ids[0])

        print(f"可训练参数数量: {sum(p.numel() for p in params if p.requires_grad)}")
        print(f"总参数数量: {sum(p.numel() for p in self.model.parameters())}")

    def adjust_learning_rate(self, min_lr=1e-6):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):

        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):

        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):

        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):

        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class AdaptFormerFinetuneTrainer(BaseModel):


    def name(self):
        return 'AdaptFormerFinetuneTrainer'

    def __init__(self, opt, pretrain_model_path=None, use_adaptformer=True,
                 adaptformer_bottleneck_dim=64, num_adapted_layers=-1):

        super(AdaptFormerFinetuneTrainer, self).__init__(opt)
        self.opt = opt
        self.use_adaptformer = use_adaptformer

        from models.adaptformer_clip import load_adaptformer_clip_model

        clip_name = opt.arch.replace("CLIP:", "") if hasattr(opt, 'arch') else 'ViT-L/14'

        print(f"创建AdaptFormer增强的CLIP模型: {clip_name}")
        self.model = load_adaptformer_clip_model(
            clip_model_name=clip_name,
            use_adaptformer=use_adaptformer,
            adaptformer_bottleneck_dim=adaptformer_bottleneck_dim,
            num_adapted_layers=num_adapted_layers
        )

        if pretrain_model_path is not None:
            print(f"从预训练模型加载CLIP权重: {pretrain_model_path}")
            self._load_pretrained_clip_weights(pretrain_model_path)

        self.model.print_trainable_params()

        trainable_params = [p for n, p in self.model.get_trainable_params()]

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay
            )
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model.to(opt.gpu_ids[0])

    def _load_pretrained_clip_weights(self, pretrain_model_path):

        try:
            checkpoint = torch.load(pretrain_model_path, map_location='cpu', weights_only=False)

            from networks.pretrain_model import PretrainModel
            temp_model = PretrainModel(
                arch=self.opt.arch if hasattr(self.opt, 'arch') else 'CLIP:ViT-L/14',
                output_dim=128,
                hidden_dim=768,
                batch_norm=True
            )

            if 'model_state_dict' in checkpoint:
                temp_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                temp_model.load_state_dict(checkpoint)

            clip_state_dict = temp_model.encoder.state_dict()

            model_state_dict = self.model.state_dict()
            updated_state_dict = {}

            for key, value in clip_state_dict.items():
                if key.startswith('model.visual.'):
                    new_key = key.replace('model.visual.', 'visual.')
                    if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                        updated_state_dict[new_key] = value

            self.model.load_state_dict(updated_state_dict, strict=False)
            print(f"成功加载 {len(updated_state_dict)} 个预训练权重")

        except Exception as e:
            print(f"警告：无法加载预训练权重: {e}")
            print("将使用原始CLIP权重继续训练")

    def adjust_learning_rate(self, min_lr=1e-6):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):

        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):

        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):

        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):

        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


