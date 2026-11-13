"""Train MSFE adapters and a binary classifier with a frozen CLIP backbone."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.msfe_clip import MSFECLIPModel


class MSFEFinetuneTrainer:
    """Optimize the MSFE adapters and classifier while keeping the visual backbone frozen."""
    def __init__(self, opt, pretrain_model_path=None, use_msfe=True,
                 msfe_hidden_dim=64, msfe_kernel_sizes=(3, 5, 7),
                 msfe_dropout=0.1, msfe_scale=0.1, num_adapted_layers=12):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_steps = 0

        # Create MSFE-enhanced CLIP.
        print("Creating MSFE-enhanced CLIP...")
        clip_model_name = opt.arch.replace("CLIP:", "")
        self.model = MSFECLIPModel(
            clip_model_name=clip_model_name,
            num_classes=1,
            use_msfe=use_msfe,
            msfe_hidden_dim=msfe_hidden_dim,
            msfe_kernel_sizes=msfe_kernel_sizes,
            msfe_dropout=msfe_dropout,
            msfe_scale=msfe_scale,
            num_adapted_layers=num_adapted_layers
        )

        # Load the Stage-1 weights when supplied.
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f"Loading the Stage-1 checkpoint: {pretrain_model_path}")
            self._load_pretrained_weights(pretrain_model_path)
        elif pretrain_model_path or getattr(opt, 'require_pretrain', False):
            raise FileNotFoundError(f"Missing Stage-1 checkpoint: {pretrain_model_path}")

        # Move the model to the training device.
        self.model = self.model.to(self.device)

        # Report trainable parameter counts.
        print("\nMSFE model parameter counts:")
        self.model.print_trainable_params()

        # Create the classification loss.
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimize only parameters with requires_grad=True.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Optimizer trainable tensors: {len(trainable_params)}")

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

        # Configure the learning-rate scheduler.
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.7)
        #self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        # Store the current inputs, labels, outputs, and loss.
        self.input = None
        self.label = None
        self.loss = None
        self.output = None

    def _load_pretrained_weights(self, pretrain_path):
        from networks.pretrain_model import PretrainModel
        from networks.checkpoints import read_checkpoint, load_stage1_visual

        checkpoint = read_checkpoint(pretrain_path)
        # Retain the model-construction RNG sequence of the original experiments.
        temporary = PretrainModel(
            arch=self.opt.arch, output_dim=128, hidden_dim=768, batch_norm=True,
        )
        temporary.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.stage1_transfer = load_stage1_visual(self.model, pretrain_path)
        print(f"Strict Stage-1 transfer: {self.stage1_transfer}")

    def set_input(self, data):
        """Move an input batch and its labels to the training device."""
        # Accept dictionary or tuple/list batches.
        if isinstance(data, dict):
            # Dictionary batch: {'img': tensor, 'label': tensor}.
            self.input = data['img'].to(self.device)
            self.label = data['label'].to(self.device).float()
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            # Tuple/list batch: [image_tensor, label_tensor].
            self.input = data[0].to(self.device)
            self.label = data[1].to(self.device).float()
        else:
            raise ValueError(f"Unsupported input format: {type(data)}, data: {data}")

    def forward(self):
        """Run the forward pass."""
        self.output = self.model(self.input).reshape(-1)

    def backward(self):
        """Backpropagate the classification loss."""
        self.loss = self.criterion(self.output, self.label)
        self.loss.backward()

    def optimize_parameters(self):
        """Update the trainable parameters for the current batch."""
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        """Return the current scalar training loss."""
        return {'loss': self.loss.item() if self.loss is not None else 0.0}

    def get_current_accuracy(self):
        """Compute accuracy on the current batch."""
        if self.output is None or self.label is None:
            return 0.0

        with torch.no_grad():
            pred = torch.sigmoid(self.output) > 0.5
            correct = (pred == self.label.bool()).float().sum()
            accuracy = correct / len(self.label)
            return accuracy.item()

    def save_networks(self, save_filename):
        """Save the model and its training state."""
        best_test_only = getattr(self.opt, 'checkpoint_policy', None) == 'best_test_only'
        if best_test_only and save_filename != 'model_best_test.pth':
            raise ValueError('best_test_only permits only model_best_test.pth')
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save a complete training checkpoint.
        checkpoint = {
            'model': self.model.state_dict(),
            'model_config': self.model.model_config,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'opt': self.opt.__dict__ if hasattr(self.opt, '__dict__') else str(self.opt)
        }

        if hasattr(self, 'checkpoint_metadata'):
            checkpoint['selection'] = dict(self.checkpoint_metadata)
        if best_test_only:
            temporary = save_path + '.tmp'
            try:
                torch.save(checkpoint, temporary)
                os.replace(temporary, save_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
            torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")

    def load_networks(self, load_filename):
        """Restore model tensors and any available optimizer/scheduler state."""
        from networks.checkpoints import canonical_msfe_state, read_checkpoint

        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, load_filename)
        checkpoint = read_checkpoint(load_path)
        self.model.load_state_dict(canonical_msfe_state(checkpoint['model']), strict=True)
        for name in ('optimizer', 'scheduler'):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name])
        self.total_steps = checkpoint.get('total_steps', self.total_steps)
        print(f"Checkpoint loaded: {load_path}")
        return True

    def update_learning_rate(self):
        """Advance the learning-rate schedule."""
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Learning rate updated to: {current_lr}")
        return current_lr

    def adjust_learning_rate(self):
        """Reduce the learning rate when continuing after early stopping."""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1

        if new_lr < 1e-7:  # Stop when the learning rate becomes too small.
            print(f"Learning rate is too small ({new_lr}), stopping training")
            return False

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f"Learning rate changed from {current_lr} to {new_lr}")
        return True

    def set_requires_grad(self, nets, requires_grad=False):
        """Enable or disable gradients for the supplied networks."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        """Switch the model to evaluation mode."""
        self.model.eval()

    def train(self):
        """Enable training for the adaptation modules and classifier."""
        self.model.train()
        # Enable training only for MSFE and the classifier.
        for name, module in self.model.named_modules():
            if 'msfe' in name.lower() or 'classifier' in name.lower():
                module.train()
            else:
                module.eval()
