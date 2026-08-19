
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.msfe_clip import MSFECLIPModel


class MSFEFinetuneTrainer:
    """Train MSFE adapters and the classifier while keeping CLIP frozen."""

    def __init__(self, opt, pretrain_model_path=None, use_msfe=True,
                 msfe_hidden_dim=128, msfe_kernel_sizes=(3, 5, 7),
                 msfe_dropout=0.1, msfe_scale=0.1, num_adapted_layers=-1):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = 0

        print("Creating an MSFE-enhanced CLIP model...")
        self.model = MSFECLIPModel(
            clip_model_name="ViT-L/14",
            num_classes=1,
            use_msfe=use_msfe,
            msfe_hidden_dim=msfe_hidden_dim,
            msfe_kernel_sizes=msfe_kernel_sizes,
            msfe_dropout=msfe_dropout,
            msfe_scale=msfe_scale,
            num_adapted_layers=num_adapted_layers
        )

        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f"加载预训练权重: {pretrain_model_path}")
            self._load_pretrained_weights(pretrain_model_path)

        self.model = self.model.to(self.device)

        print("\nMSFE模型参数统计:")
        self.model.print_trainable_params()

        self.criterion = nn.BCEWithLogitsLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"优化器将训练 {len(trainable_params)} 个参数组")

        if opt.optim == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay
            )
        elif opt.optim == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt.optim}")

        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.7)
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.input = None
        self.label = None
        self.loss = None
        self.output = None

    def _load_pretrained_weights(self, pretrain_path):

        try:
            checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)

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

            if updated_state_dict:
                self.model.load_state_dict(updated_state_dict, strict=False)
                print(f"✅ 成功加载 {len(updated_state_dict)} 个预训练CLIP权重")
            else:
                print("⚠️ 没有找到匹配的CLIP权重，将使用原始CLIP权重")

            if isinstance(checkpoint, dict) and 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
                print(f"恢复训练步数: {self.total_steps}")

        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            print("将使用原始CLIP权重继续训练")

    def set_input(self, data):

        if isinstance(data, dict):
            self.input = data['img'].to(self.device)
            self.label = data['label'].to(self.device).float()
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            self.input = data[0].to(self.device)
            self.label = data[1].to(self.device).float()
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}, 内容: {data}")

    def forward(self):

        self.output = self.model(self.input).squeeze()

    def backward(self):

        self.loss = self.criterion(self.output, self.label)
        self.loss.backward()

    def optimize_parameters(self):

        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):

        return {'loss': self.loss.item() if self.loss is not None else 0.0}

    def get_current_accuracy(self):

        if self.output is None or self.label is None:
            return 0.0

        with torch.no_grad():
            pred = torch.sigmoid(self.output) > 0.5
            correct = (pred == self.label.bool()).float().sum()
            accuracy = correct / len(self.label)
            return accuracy.item()

    def save_networks(self, save_filename):

        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'opt': self.opt.__dict__ if hasattr(self.opt, '__dict__') else str(self.opt)
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")

    def load_networks(self, load_filename):

        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, load_filename)

        if not os.path.exists(load_path):
            print(f"模型文件不存在: {load_path}")
            return False

        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model'])

            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']

            print(f"✅ 模型加载成功: {load_path}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def update_learning_rate(self):

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"学习率更新为: {current_lr}")
        return current_lr

    def adjust_learning_rate(self):

        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1

        if new_lr < 1e-7:
            print(f"学习率过小 ({new_lr})，停止训练")
            return False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"学习率从 {current_lr} 调整为 {new_lr}")
        return True

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):

        self.model.eval()

    def train(self):

        self.model.train()
        for name, module in self.model.named_modules():
            if 'msfe' in name.lower() or 'classifier' in name.lower():
                module.train()
            else:
                module.eval()
