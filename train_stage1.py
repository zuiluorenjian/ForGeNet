"""Stage-1 representation learning with asymmetric contrastive losses and an adaptive gate."""
import os
import time
import argparse
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.datasets import RealFakeDataset
from networks.pretrain_model import PretrainModel
from networks.gradient_gate import (
    LearnableGradientGate, combine_objectives, measure_gradient_norms,
    select_gradient_probe,
)
from loss import SupConLoss, ConSupPrototypeLoss  # Contrastive and prototype objectives.
from options.train_options import TrainOptions
from earlystop import EarlyStopping


GATE_METRIC_FIELDS = [
    'seed', 'epoch', 'step', 'loss', 'original_loss', 'real_loss', 'fake_loss',
    'real_mean', 'fake_mean', 'real_count', 'fake_count', 'fake_weight',
    'real_weight', 'grad_real', 'grad_fake', 'ema_real', 'ema_fake',
    'weighted_grad_real', 'weighted_grad_fake', 'gradient_cosine',
    'gate_balance_loss', 'feedback_active', 'near_bound',
    'gate_probe', 'gate_probe_parameters',
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Stage 1: Contrastive Pretraining')

    # Basic training settings.
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Projection-head learning rate (default: 1e-4); LoRA uses --lora_lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # Optimizer settings.
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer: adam or sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (used only with SGD)')

    # Model settings.
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14', help='CLIP architecture')
    parser.add_argument('--output_dim', type=int, default=128, help='Projection-head output dimension')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Projection-head hidden dimension')
    parser.add_argument('--batch_norm', action=argparse.BooleanOptionalAction, default=True, help='Enable batch normalization')
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True,
                       help="Adapt the Q/V projections of CLIP ViT attention with LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha")
    parser.add_argument("--lora_layers", type=int, default=4,
                       help="Number of final ViT blocks with LoRA; -1 selects all blocks")
    parser.add_argument("--lora_lr", type=float, default=1e-4,
                       help="LoRA learning rate")

    # Contrastive-learning settings.
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature')
    parser.add_argument('--base_temperature', type=float, default=0.07, help='Base temperature')
    parser.add_argument('--min_class', type=int, default=1, help='Minority-class label (1=AI-generated)')
    parser.add_argument('--ratio_supervised_majority', type=float, default=0.0,
                       help='Fraction of majority-class supervised positives (0.0 keeps only instance positives for real images)')
    parser.add_argument(
        '--supervision_mode', type=str, default='legacy',
        choices=['legacy', 'majority_supervised'],
        help='Positive-pair strategy: legacy or majority-supervised with minority instance positives',
    )

    # Loss selection.
    parser.add_argument('--loss_type', type=str, default='supcon',
                       choices=['supcon', 'prototype'],
                       help='Loss: supcon for supervised contrastive learning, prototype for prototype contrastive learning')

    # Prototype-loss settings.
    parser.add_argument('--negatives_weight', type=float, default=1.0, help='Negative-sample weight')
    parser.add_argument('--eps', type=float, default=0.1, help='Prototype distance margin')
    parser.add_argument('--eps_0', type=float, default=0.1, help='Class-0 margin')
    parser.add_argument('--eps_1', type=float, default=0.1, help='Class-1 margin')

    # Sampling settings.
    parser.add_argument('--use_ratio_sampling', action='store_true',
                       help='Enable ratio sampling; disabled by default to use the full imbalanced split')
    parser.add_argument('--real_ratio', type=float, default=0.5,
                       help='Retention fraction for real images when ratio sampling is enabled')
    parser.add_argument('--fake_ratio', type=float, default=0.5,
                       help='Retention fraction for AI-generated images when ratio sampling is enabled')

    # Dataset paths.
    parser.add_argument('--wang2020_data_path', type=str, required=True,
                       help='Dataset root containing train/ and val/')
    parser.add_argument('--data_mode', type=str, default='wang2020',
                       choices=['ours', 'wang2020', 'ours_wang2020'],
                       help='Dataset layout')
    parser.add_argument('--use_forgelens_style', action='store_true',
                       help='Tile small images before cropping and disable extra JPEG/blur augmentation')

    # Output directories.
    parser.add_argument('--save_path', type=str, default='./checkpoints/stage1',
                       help='Checkpoint directory')
    parser.add_argument('--log_path', type=str, default='./logs/stage1',
                       help='Log directory')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10, help='Early-stopping patience')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint interval in epochs')
    parser.add_argument('--seed', type=int, default=2026, help='Stage-1 random seed (default: 2026)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Resume from a training checkpoint produced by this implementation')
    parser.add_argument('--loss_weighting', default='learned',
                        choices=['none', 'fixed', 'gradient', 'learned'],
                        help='Original objective, fixed weighting, analytic gradient weighting, or learned gate')
    parser.add_argument('--component_reduction', default='batch', choices=['batch', 'class'],
                        help='batch preserves the original batch-normalized loss; class uses per-class anchor means')
    parser.add_argument('--fixed_fake_weight', type=float, default=0.5)
    parser.add_argument('--gate_hidden_dim', type=int, default=64)
    parser.add_argument('--gate_lr', type=float, default=3e-4)
    parser.add_argument('--gate_probe', choices=['projection', 'lora'], default='lora',
                        help='Shared parameters used for gradient feedback; main runs use lora, legacy runs may use projection')
    parser.add_argument('--gate_min', type=float, default=0.1)
    parser.add_argument('--gate_max', type=float, default=0.9)
    parser.add_argument('--gate_ema_decay', type=float, default=0.9)
    parser.add_argument('--gate_eps', type=float, default=1e-8)
    args = parser.parse_args(argv)
    if args.loss_weighting != 'none' and args.loss_type != 'supcon':
        parser.error('Gradient weighting currently supports only the SupCon loss')
    if args.loss_weighting != 'none' and args.gate_probe == 'lora' and not args.use_lora:
        parser.error('gate_probe=lora requires --use_lora')
    if args.component_reduction == 'class' and args.loss_weighting == 'none':
        parser.error('Class-wise reduction requires fixed, gradient, or learned weighting')
    if args.loss_weighting == 'fixed' and not 0 < args.fixed_fake_weight < 1:
        parser.error('fixed_fake_weight must be in (0, 1)')
    return args


def initialize_prototypes(feature_dim, device):
    """Initialize unit-normalized class prototypes."""
    # Initialize one prototype vector for each class.
    prototypes = torch.randn(2, feature_dim, device=device)
    # Normalize prototypes to the unit sphere.
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
    return prototypes


def convert_labels_to_onehot(labels, num_classes=2):
    """Convert class labels to one-hot vectors."""
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1).long(), 1)
    return onehot


def create_dataloader(opt, args, is_train=True):
    """Build the data loader."""
    # Select the dataset split.
    data_opt = opt
    data_opt.data_label = 'train' if is_train else 'val'
    data_opt.isTrain = is_train

    # Enable ratio sampling only for the training split.
    if is_train and args.use_ratio_sampling:
        data_opt.ratio_sampling = True
        data_opt.real_ratio = args.real_ratio
        data_opt.fake_ratio = args.fake_ratio
        print(f"Ratio sampling: real images {args.real_ratio*100:.1f}%, AI-generated images {args.fake_ratio*100:.1f}%")
    else:
        data_opt.ratio_sampling = False
        if is_train:
            print("Using the full imbalanced training split")

    # Return two independently augmented views for contrastive learning.
    dataset = RealFakeDataset(data_opt, is_pretraining=True)

    # Build the data loader.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # Use the requested batch size.
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=seed_worker,
    )

    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch,
                use_prototype=False, args=None, gate=None, gate_optimizer=None,
                writer=None, metrics_writer=None, global_step=0):
    """Train for one epoch."""
    model.train()
    model.projection_head.train()  # Keep the projection head in training mode.

    total_loss = 0.0
    num_batches = 0
    weighting = args.loss_weighting if args is not None else 'none'
    probe_location = getattr(args, 'gate_probe', 'projection')
    probe_parameters = select_gradient_probe(model, probe_location) if weighting != 'none' else ()
    probe_parameter_count = sum(p.numel() for p in probe_parameters)
    if gate is not None:
        gate.train()

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Each sample supplies two augmented views: [view1, view2].
        # Concatenate views along the batch dimension.
        view1, view2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        # Combine both views.
        batch_size = labels.size(0)
        all_images = torch.cat([view1, view2], dim=0)

        # Forward pass.
        encoder_features = None
        if weighting == 'learned':
            encoder_features, projections = model(all_images, return_features=True)
        else:
            projections = model(all_images)

        # Separate the projections of the two views.
        proj1, proj2 = torch.split(projections, [batch_size, batch_size], dim=0)

        # Reshape to [batch_size, n_views, feature_dim].
        features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)

        # Prepare labels for the selected contrastive objective.
        if use_prototype:
            # Prototype losses require one-hot labels.
            labels_input = convert_labels_to_onehot(labels, num_classes=2)
        else:
            # The standard contrastive objective uses integer labels.
            labels_input = labels

        # Compute the loss.
        gate_loss = None
        metrics = None
        if weighting == 'none':
            loss, _, _ = criterion(features, labels_input)
        else:
            original_loss, _, _, parts = criterion(features, labels_input, return_components=True)
            suffix = '_mean' if args.component_reduction == 'class' else ''
            real_loss, fake_loss = parts['real' + suffix], parts['fake' + suffix]
            active = parts['real_count'] > 0 and parts['fake_count'] > 0
            fake_weight = real_loss.new_tensor(
                args.fixed_fake_weight if weighting == 'fixed' else 0.5,
            )
            norms, cosine = None, None
            if active:
                norms, cosine = measure_gradient_norms(
                    real_loss, fake_loss, probe_parameters,
                )
                gate.observe(norms)
                if weighting == 'learned':
                    fake_weight = gate(encoder_features, labels.repeat(2))
                    gate_loss = gate.balance_loss(fake_weight)
                elif weighting == 'gradient':
                    fake_weight = gate.formula_weight()
                else:
                    fake_weight = real_loss.new_tensor(args.fixed_fake_weight)
            # A single-class batch cannot supervise balance. Adaptive modes use
            # the original contribution; fixed mode retains its configured weight.
            loss = combine_objectives(real_loss, fake_loss, fake_weight)
            weight = float(fake_weight.detach())
            metrics = dict(
                seed=args.seed, epoch=epoch + 1, step=global_step,
                loss=float(loss.detach()), original_loss=float(original_loss.detach()),
                real_loss=float(real_loss.detach()), fake_loss=float(fake_loss.detach()),
                real_mean=float(parts['real_mean'].detach()),
                fake_mean=float(parts['fake_mean'].detach()),
                real_count=parts['real_count'], fake_count=parts['fake_count'],
                fake_weight=weight, real_weight=1 - weight,
                grad_real=float(norms[0]) if active else None,
                grad_fake=float(norms[1]) if active else None,
                ema_real=float(gate.gradient_ema[0]), ema_fake=float(gate.gradient_ema[1]),
                weighted_grad_real=2 * (1 - weight) * float(norms[0]) if active else None,
                weighted_grad_fake=2 * weight * float(norms[1]) if active else None,
                gradient_cosine=float(cosine) if active else None,
                gate_balance_loss=float(gate_loss.detach()) if gate_loss is not None else None,
                feedback_active=int(active),
                near_bound=int(active and weighting in ('gradient', 'learned')
                               and min(weight - args.gate_min, args.gate_max - weight)
                               <= 0.01 * (args.gate_max - args.gate_min)),
                gate_probe=probe_location, gate_probe_parameters=probe_parameter_count,
            )

        # Backward pass.
        optimizer.zero_grad(set_to_none=True)
        if gate_optimizer is not None:
            gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gate_loss is not None:
            # Gate inputs/norm targets are detached; this backward only trains MLP.
            gate_loss.backward()
        optimizer.step()
        if gate_loss is not None:
            gate_optimizer.step()

        if metrics is not None:
            if metrics_writer is not None:
                metrics_writer.writerow(metrics)
            if writer is not None:
                for name, value in metrics.items():
                    if value is not None and name not in ('seed', 'epoch', 'step', 'gate_probe'):
                        writer.add_scalar('Gate/' + name, value, global_step)
        global_step += 1

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f}' + (
                      f' Fake weight: {metrics["fake_weight"]:.4f}'
                      f' Feedback: {metrics["feedback_active"]}' if metrics is not None else ''
                  ))

    if num_batches == 0:
        raise ValueError('No complete training batch; reduce batch_size')
    return total_loss / num_batches, global_step


def validate_epoch(model, dataloader, criterion, device, use_prototype=False):
    """Evaluate the contrastive objective for one epoch."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            view1, view2 = images[0].to(device), images[1].to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            all_images = torch.cat([view1, view2], dim=0)

            projections = model(all_images)
            proj1, proj2 = torch.split(projections, [batch_size, batch_size], dim=0)
            features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)

            # Prepare labels for the selected loss.
            if use_prototype:
                # Prototype losses require one-hot labels.
                labels_input = convert_labels_to_onehot(labels, num_classes=2)
            else:
                # The standard contrastive objective uses integer labels.
                labels_input = labels

            loss, _, _ = criterion(features, labels_input)
            total_loss += loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError('The validation split is empty')
    return total_loss / num_batches


def make_checkpoint(model, criterion, optimizer, scheduler, gate, gate_optimizer,
                    args, epoch, global_step, train_loss, val_loss, best_loss,
                    early_stopping, prototypes):
    checkpoint = {
        'epoch': epoch,
        # Stage 2 consumes only this standard, merged detector state.
        'model_state_dict': model.merged_state_dict(),
        # Preserve unmerged LoRA bases/adapters for exact continuation.
        'training_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss, 'train_loss': train_loss, 'best_loss': best_loss,
        'global_step': global_step, 'args': args,
        'early_stopping': {
            'best_score': early_stopping.best_score,
            'counter': early_stopping.counter,
            'early_stop': early_stopping.early_stop,
        },
        'rng_state': {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if prototypes is not None:
        checkpoint['prototypes'] = prototypes
    if args.use_lora:
        checkpoint['lora_state_dict'] = model.lora_state_dict()
    if gate is not None:
        checkpoint['gate_state_dict'] = gate.state_dict()
    if gate_optimizer is not None:
        checkpoint['gate_optimizer_state_dict'] = gate_optimizer.state_dict()
    return checkpoint


def main():
    args = parse_args()
    set_seed(args.seed)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous_args = vars(resume['args'])
        # Checkpoints predating this option always used the projection head.
        previous_probe = previous_args.get('gate_probe', 'projection')
        if args.loss_weighting != 'none' and previous_probe != args.gate_probe:
            raise ValueError(
                f'Gradient-probe mismatch when resuming: {previous_probe} -> {args.gate_probe}; '
                'Retrain Stage 1; the old gate/EMA state cannot be reused with a different probe',
            )
        for name in (
            'arch', 'output_dim', 'hidden_dim', 'batch_norm', 'use_lora',
            'lora_rank', 'lora_alpha', 'lora_layers', 'loss_type', 'loss_weighting',
            'component_reduction', 'gate_hidden_dim', 'gate_min', 'gate_max',
            'gate_ema_decay', 'gate_eps', 'fixed_fake_weight', 'seed', 'epochs',
            'batch_size', 'temperature', 'base_temperature', 'min_class',
            'ratio_supervised_majority', 'supervision_mode', 'wang2020_data_path',
            'use_ratio_sampling', 'real_ratio', 'fake_ratio', 'optimizer',
        ):
            if previous_args.get(name) != getattr(args, name):
                raise ValueError(f'Training configuration mismatch when resuming: {name}; Use the original experiment settings')
        if resume['epoch'] + 1 >= args.epochs:
            print('The checkpoint has already reached the requested number of epochs.')
            return

    # Select the training device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Create output directories.
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    with open(os.path.join(args.log_path, 'config.json'), 'w') as config_file:
        json.dump(vars(args), config_file, indent=2, ensure_ascii=False)
    print(f'Random seed: {args.seed}; Loss weighting: {args.loss_weighting}; '
          f'Component reduction: {args.component_reduction}')
    print('Validation and best-checkpoint selection use the original unweighted contrastive loss')

    # Construct the dataset options without invoking another argument parser.
    class SimpleOpt:
        def __init__(self, args):
            # Set the required dataset defaults.
            self.data_label = 'train'
            self.data_mode = args.data_mode
            self.wang2020_data_path = args.wang2020_data_path
            self.real_list_path = None  # Not used by the wang2020 dataset layout.
            self.fake_list_path = None  # Not used by the wang2020 dataset layout.
            self.isTrain = True
            self.no_crop = False
            self.no_flip = False
            self.no_resize = False
            self.cropSize = 224
            self.loadSize = 256
            self.rz_interp = ['bilinear']
            self.blur_prob = 0.5
            self.blur_sig = [0.0, 3.0]
            self.jpg_prob = 0.5
            self.jpg_method = ['pil']
            self.jpg_qual = [30, 100]
            self.arch = args.arch
            self.use_forgelens_style = args.use_forgelens_style
            self.batch_size = args.batch_size
            # Ratio-sampling settings.
            self.ratio_sampling = False
            self.real_ratio = 0.5
            self.fake_ratio = 0.5

    opt = SimpleOpt(args)

    # Build the data loader.
    print("Creating data loaders...")
    train_loader = create_dataloader(opt, args, is_train=True)
    val_loader = create_dataloader(opt, args, is_train=False)

    print(f"Training split size: {len(train_loader.dataset)}")
    print(f"Validation split size: {len(val_loader.dataset)}")

    # Create the model.
    print("Creating the Stage-1 model...")
    model = PretrainModel(
        arch=args.arch,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        batch_norm=args.batch_norm,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_layers=args.lora_layers,
    ).to(device)

    # Enable training of the projection head.
    model.unfreeze_projection_head()

    # Create the loss function.
    use_prototype = (args.loss_type == 'prototype')
    prototypes = None

    if use_prototype:
        print("Using prototype contrastive learning...")
        # Initialize the prototypes.
        prototypes = initialize_prototypes(args.output_dim, device)

        # Create the prototype objective.
        criterion = ConSupPrototypeLoss(
            temperature=args.temperature,
            base_temperature=args.base_temperature,
            negatives_weight=args.negatives_weight,
            eps=args.eps,
            eps_0=args.eps_0,
            eps_1=args.eps_1,
            minority_cls=args.min_class,
            max_epoch=args.epochs
        ).to(device)

        # Register the prototypes.
        criterion.set_prototypes(prototypes)
    else:
        print("Using the supervised contrastive loss implementation...")
        # Create the standard contrastive objective.
        criterion = SupConLoss(
            temperature=args.temperature,
            base_temperature=args.base_temperature,
            min_class=args.min_class,
            ratio_supervised_majority=args.ratio_supervised_majority,
            supervision_mode=args.supervision_mode,
        ).to(device)

    # Train the projection head and, when enabled, the low-rank attention updates.
    parameter_groups = [
        {"params": [p for p in model.projection_head.parameters() if p.requires_grad],
         "lr": args.lr}
    ]
    if args.use_lora:
        lora_params = list(model.lora_parameters())
        parameter_groups.append({"params": lora_params, "lr": args.lora_lr})
        print(
            f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}, "
            f"Learning rate: {args.lora_lr}"
        )

    if args.optimizer.lower() == 'adam':
        print(f"Adam learning rate: {args.lr}")
        optimizer = torch.optim.Adam(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        print(f"SGD learning rate: {args.lr}, momentum: {args.momentum}")
        optimizer = torch.optim.SGD(parameter_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Learning-rate schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    gate, gate_optimizer = None, None
    if args.loss_weighting != 'none':
        probe_parameters = select_gradient_probe(model, args.gate_probe)
        print(f'Gate gradient probe: {args.gate_probe}; '
              f'Trainable parameter count: {sum(p.numel() for p in probe_parameters):,}')
        # Gate initialization must not change detector initialization/data order
        # relative to the fixed/analytic ablations under the same seed.
        with torch.random.fork_rng(devices=[]):
            gate = LearnableGradientGate(
                model.encoder_dim, args.gate_hidden_dim,
                args.gate_min, args.gate_max, args.gate_ema_decay, args.gate_eps,
            ).to(device)
        if args.loss_weighting == 'learned':
            gate_optimizer = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)
            print(f'Trainable gate parameters: {sum(p.numel() for p in gate.parameters()):,}')
        else:
            gate.requires_grad_(False)

    # Create TensorBoard logs.
    writer = SummaryWriter(args.log_path)

    # Early stopping.
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training loop.
    print("Starting training...")
    best_loss = float('inf')
    start_time = time.time()
    start_epoch, global_step = 0, 0
    if resume is not None:
        model.load_state_dict(resume['training_state_dict'])
        criterion.load_state_dict(resume['criterion_state_dict'])
        if prototypes is not None:
            prototypes = resume['prototypes'].to(device)
            criterion.set_prototypes(prototypes)
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        if gate is not None:
            gate.load_state_dict(resume['gate_state_dict'])
        if gate_optimizer is not None:
            gate_optimizer.load_state_dict(resume['gate_optimizer_state_dict'])
        for name, value in resume['early_stopping'].items():
            setattr(early_stopping, name, value)
        best_loss = resume['best_loss']
        start_epoch, global_step = resume['epoch'] + 1, resume['global_step']
        state = resume['rng_state']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if torch.cuda.is_available() and state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
        del resume
        print(f'Resuming training: epoch={start_epoch + 1}, step={global_step}')

    metrics_file, metrics_writer = None, None
    if gate is not None:
        metrics_path = os.path.join(args.log_path, 'gate_metrics.csv')
        append_metrics = args.resume and os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0
        metric_fields = GATE_METRIC_FIELDS
        if append_metrics:
            with open(metrics_path, newline='') as previous_metrics:
                previous_fields = next(csv.reader(previous_metrics))
            if previous_fields == GATE_METRIC_FIELDS[:-2] and args.gate_probe == 'projection':
                # Keep old projection runs resumable without corrupting their CSV.
                metric_fields = previous_fields
            elif previous_fields != GATE_METRIC_FIELDS:
                raise ValueError('Incompatible gate_metrics.csv header; cannot append gradient statistics')
        metrics_file = open(metrics_path, 'a' if append_metrics else 'w', newline='')
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=metric_fields,
            extrasaction='ignore' if metric_fields != GATE_METRIC_FIELDS else 'raise',
        )
        if not append_metrics:
            metrics_writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        # Training phase.
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_prototype,
            args=args, gate=gate, gate_optimizer=gate_optimizer, writer=writer,
            metrics_writer=metrics_writer, global_step=global_step,
        )
        if metrics_file is not None:
            metrics_file.flush()

        # Validation phase.
        val_loss = validate_epoch(model, val_loader, criterion, device, use_prototype)

        # Update the learning rate.
        scheduler.step()

        # Log metrics.
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Val Loss: {val_loss:.4f} '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # EarlyStopping expects a score to maximize; validation loss is minimized.
        early_stopping(-val_loss, model)
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
        save_dict = make_checkpoint(
            model, criterion, optimizer, scheduler, gate, gate_optimizer, args,
            epoch, global_step, train_loss, val_loss, best_loss, early_stopping,
            prototypes,
        )
        if improved:
            torch.save(save_dict, os.path.join(args.save_path, 'best_model.pth'))
            print(f'Saving the best checkpoint; validation loss: {val_loss:.4f}')

        # Save periodic checkpoints.
        if (epoch + 1) % args.save_freq == 0:
            torch.save(save_dict, os.path.join(args.save_path, f'model_epoch_{epoch+1}.pth'))

        # Check the early-stopping condition.
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save the final checkpoint.
    torch.save(save_dict, os.path.join(args.save_path, 'final_model.pth'))

    total_time = time.time() - start_time
    print(f'Training completed. Total time: {total_time/3600:.2f} hours')

    # Close log writers.
    writer.close()
    if metrics_file is not None:
        metrics_file.close()


if __name__ == '__main__':
    main()
