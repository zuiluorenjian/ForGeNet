"""Stage-2 decision boundary learning with MSFE on a balanced training subset."""
import os
import time
import argparse
import copy
import json
import numpy as np
import torch
import random
from tensorboardX import SummaryWriter

# Seed all random-number generators.
def set_seed(seed=2026):
    """Seed random-number generators and enable deterministic cuDNN behavior."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Enable deterministic cuDNN behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# Set the import-time default; main() resets the seed from the arguments.
set_seed(2026)

from validate import validate
from data import create_dataloader
from data import get_ratio_subset
from earlystop import EarlyStopping
from networks.msfe_trainer import MSFEFinetuneTrainer
from options.train_options import TrainOptions


def create_balanced_subset_options(opt, real_count=3000, fake_count=3000):
    """Clone options and configure fixed real and AI-generated sample counts."""
    balanced_opt = copy.copy(opt)

    # Configure fixed-count sampling.
    balanced_opt.fixed_sampling = True
    balanced_opt.real_count = real_count  # Number of real images.
    balanced_opt.fake_count = fake_count  # Number of AI-generated images.

    # Supply required defaults when an option is absent.
    if not hasattr(balanced_opt, 'serial_batches'):
        balanced_opt.serial_batches = False
    if not hasattr(balanced_opt, 'class_bal'):
        balanced_opt.class_bal = False
    if not hasattr(balanced_opt, 'ratio_sampling'):
        balanced_opt.ratio_sampling = False
    if not hasattr(balanced_opt, 'num_threads'):
        balanced_opt.num_threads = 4
    if not hasattr(balanced_opt, 'batch_size'):
        balanced_opt.batch_size = opt.batch_size  # Use the configured batch size.

    return balanced_opt


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Stage 2: MSFE-Enhanced Balanced Finetuning')
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14',
                       choices=['CLIP:ViT-B/16', 'CLIP:ViT-B/32', 'CLIP:ViT-L/14'],
                       help='CLIP visual backbone (default: CLIP:ViT-L/14)')


    # Stage-1 checkpoint.
    parser.add_argument('--pretrain_model_path', type=str,
                       help='Stage-1 checkpoint; omit only when intentionally adapting original CLIP')
    parser.add_argument('--require_pretrain', action='store_true',
                       help='Require successful Stage-1 checkpoint loading; abort on failure')
    parser.add_argument('--linear_probe', action='store_true',
                       help='Linear-only ablation: freeze the Stage-1 encoder and train the classifier without MSFE')

    # Dataset paths.
    parser.add_argument('--wang2020_data_path', type=str, required=True,
                       help='Dataset root containing train/ and val/')
    parser.add_argument('--data_mode', type=str, default='wang2020',
                       choices=['ours', 'wang2020', 'ours_wang2020'],
                       help='Dataset layout')

    # Training settings.
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--real_count', type=int, default=360,
                       help='Number of selected real training images (ProGAN default: 360)')
    parser.add_argument('--fake_count', type=int, default=360,
                       help='Number of selected AI-generated training images (ProGAN default: 360)')

    # MSFE settings.
    parser.add_argument('--use_msfe', action='store_true', default=True,
                       help='Enable MSFE adapters (default: True)')
    parser.add_argument('--msfe_hidden_dim', type=int, default=64,
                       help='MSFE bottleneck dimension (default: 64)')
    parser.add_argument('--msfe_kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                       help='MSFE kernel sizes (default: 3, 5, 7)')
    parser.add_argument('--msfe_dropout', type=float, default=0.1,
                       help='MSFE dropout probability (default: 0.1)')
    parser.add_argument('--msfe_scale', type=float, default=0.1,
                       help='MSFE residual scale (default: 0.1)')
    parser.add_argument('--num_adapted_layers', type=int, default=12,
                       help='Number of final blocks with MSFE; -1 selects all blocks (default: 12)')

    # Preprocessing selection.
    parser.add_argument('--use_forgelens_style', action='store_true', default=False,
                       help='Tile small images and crop without extra JPEG/blur augmentation')
    parser.add_argument('--train_fake_jpeg_quality', type=int, default=None,
                       help='Recompress full AI-generated training images before cropping (PIL, 4:2:0, Q=1..100); disabled by default')

    # Evaluation subsets.
    parser.add_argument('--test_data_path', type=str,
                       default=None,
                       help='Evaluation dataset root; use --test_config or --test_root_override to configure subsets')
    parser.add_argument('--test_max_sample', type=int, default=1000,
                       help='Maximum images per class in each evaluation subset (default: 1000)')
    parser.add_argument('--test_root_override', type=str, default=None,
                       help='Replace evaluation roots from dataset_paths while preserving subset directory names')
    parser.add_argument('--test_config', type=str, default=None,
                       help='JSON evaluation configuration overriding dataset_paths.py')
    parser.add_argument('--data_seed', type=int, default=None,
                       help='Training-subset sampling seed; omit to retain legacy sampling')
    parser.add_argument('--test_seed', type=int, default=None,
                       help='Evaluation-subset sampling seed; omit to retain legacy sampling')

    # Output paths.
    parser.add_argument('--save_path', type=str, default='./checkpoints/stage2_msfe',
                       help='Checkpoint directory')
    parser.add_argument('--log_path', type=str, default='./logs/stage2_msfe',
                       help='Log directory')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15, help='Early-stopping patience')
    parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint interval in epochs')
    parser.add_argument('--best_test_only', action='store_true',
                       help='Only retain model_best_test.pth, selected by mean test accuracy')

    # Random seed.
    parser.add_argument('--seed', type=int, default=2026, help='Random seed (default: 2026)')

    args = parser.parse_args()
    if args.linear_probe and args.best_test_only:
        parser.error('--best_test_only is for MSFE training, not linear_probe')
    if args.linear_probe and not args.pretrain_model_path:
        parser.error('--linear_probe requires --pretrain_model_path')
    if args.train_fake_jpeg_quality is not None and not 1 <= args.train_fake_jpeg_quality <= 100:
        parser.error('--train_fake_jpeg_quality must be between 1 and 100')
    return args


def main():
    args = parse_args()

    # Apply the requested random seed.
    set_seed(args.seed)

    # Create output directories.
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Print configuration information
    print("=" * 60)
    print("Linear-only Stage 2 Configuration" if args.linear_probe
          else "MSFE-Enhanced Stage 2 Fine-tuning Configuration")
    print("=" * 60)
    print(f"Use MSFE: {args.use_msfe and not args.linear_probe}")
    if args.linear_probe:
        print('Frozen Stage-1 CLIP; Dropout(0.1) + Linear classifier; no MSFE modules')
    else:
        print(f"MSFE hidden dimension: {args.msfe_hidden_dim}")
        print(f"MSFE kernel sizes: {args.msfe_kernel_sizes}")
        print(f"MSFE dropout: {args.msfe_dropout}")
        print(f"MSFE scale: {args.msfe_scale}")
        print(f"MSFE layers: {args.num_adapted_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training data: Real {args.real_count} + Fake {args.fake_count}")
    print(f"Training fake JPEG quality: {args.train_fake_jpeg_quality} "
          "(train fake only, full image before crop, PIL subsampling=2)")
    if args.pretrain_model_path:
        print(f"Pretrained model: {args.pretrain_model_path}")
    else:
        print("Pretrained model: Using original CLIP weights")
    print("=" * 60)

    # Construct the dataset and trainer options.
    class SimpleOpt:
        def __init__(self, args):
            self.data_mode = args.data_mode
            self.wang2020_data_path = args.wang2020_data_path
            self.real_list_path = None
            self.fake_list_path = None
            self.isTrain = True
            self.no_crop = False
            self.no_flip = False
            self.no_resize = False
            self.cropSize = 224
            self.loadSize = 256
            self.rz_interp = ['bilinear']
            self.blur_prob = 0.1
            self.blur_sig = [0.0, 3.0]
            self.jpg_prob = 0.1
            self.jpg_method = ['pil']
            self.jpg_qual = [30, 100]
            self.arch = args.arch
            self.batch_size = args.batch_size
            self.fix_backbone = False
            self.optim = 'adam'
            self.lr = args.lr
            self.beta1 = 0.9
            self.weight_decay = 0.01
            self.init_gain = 0.02
            self.gpu_ids = [0]
            self.loss_freq = 400
            # Tiling/cropping preprocessing settings.
            self.use_forgelens_style = args.use_forgelens_style
            self.train_fake_jpeg_quality = args.train_fake_jpeg_quality
            # Sampling options.
            self.ratio_sampling = False
            self.real_ratio = 0.5
            self.fake_ratio = 0.5
            self.class_bal = False
            self.serial_batches = False
            self.fixed_sampling = False
            self.num_threads = 4
            # Checkpoint output settings.
            self.checkpoints_dir = os.path.dirname(args.save_path)
            self.require_pretrain = args.require_pretrain
            self.linear_probe = args.linear_probe
            self.msfe_hidden_dim = args.msfe_hidden_dim
            self.msfe_kernel_sizes = args.msfe_kernel_sizes
            self.msfe_dropout = args.msfe_dropout
            self.msfe_scale = args.msfe_scale
            self.num_adapted_layers = args.num_adapted_layers
            self.checkpoint_policy = ('best_test_only' if args.best_test_only else
                                      'final_only' if args.linear_probe else 'best_and_periodic')
            self.name = os.path.basename(args.save_path)

        def set_train_mode(self):
            self.data_label = 'train'
            self.isTrain = True

        def set_val_mode(self):
            self.data_label = 'val'
            self.isTrain = False
            self.no_resize = False
            self.no_crop = False
            self.serial_batches = True

    opt = SimpleOpt(args)

    # Create balanced training dataset
    print("Creating balanced training dataset...")
    balanced_train_opt = create_balanced_subset_options(opt, args.real_count, args.fake_count)
    balanced_train_opt.data_label = 'train'
    balanced_train_opt.isTrain = True
    balanced_train_opt.data_seed = args.data_seed

    # Create data loader
    train_data_loader = create_dataloader(balanced_train_opt)
    print(f"Balanced training set size: {len(train_data_loader.sampler)}")

    # Build the validation data loader.
    val_opt = copy.copy(opt)

    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    val_loader = create_dataloader(val_opt)
    print(f"Validation set size: {len(val_loader.dataset)}")

    # Create test data loaders (using Ojha datasets for testing)
    print("Creating test data loaders...")
    print(f"Test dataset path: {args.test_data_path}")

    # Import the default evaluation configuration.
    from dataset_paths import DATASET_PATHS

    # Get all available test datasets
    test_datasets = [dict(d) for d in DATASET_PATHS if d]
    if args.test_config:
        with open(args.test_config) as config_file:
            test_datasets = json.load(config_file)
    if args.test_root_override:
        for dataset in test_datasets:
            for path_key in ('real_path', 'fake_path'):
                subdirectory = os.path.basename(os.path.normpath(dataset[path_key]))
                dataset[path_key] = os.path.join(args.test_root_override, subdirectory)
    print(f"Found {len(test_datasets)} test datasets")

    test_loaders = []
    test_dataset_names = []

    # Import the evaluation dataset class.
    from validate import RealFakeDataset
    import torch.utils.data

    for dataset in test_datasets:
        dataset_name = dataset['key']
        print(f"  Preparing test dataset: {dataset_name}")

        # Create the evaluation subset.
        test_dataset = RealFakeDataset(
            real_path=dataset['real_path'],
            fake_path=dataset['fake_path'],
            data_mode=dataset['data_mode'],
            max_sample=args.test_max_sample,
            arch=opt.arch,
            sampling_seed=args.test_seed,
        )

        if dataset.get('transform') == 'resize224':
            from data.evaluation import FullImageResize
            test_dataset.transform = FullImageResize(224)

        # Build the data loader.
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_loaders.append(test_loader)
        test_dataset_names.append(dataset_name)
        print(f"    {dataset_name}: {len(test_dataset)} samples")

    print(f"Successfully created {len(test_loaders)} test data loaders")

    # Create MSFE-enhanced fine-tuning model
    if args.linear_probe:
        from networks.linear_trainer import LinearFinetuneTrainer
        print('Creating frozen Stage-1 CLIP + linear classifier (MSFE disabled)...')
        model = LinearFinetuneTrainer(opt, args.pretrain_model_path)
    else:
        print("Creating MSFE-enhanced fine-tuning model...")
        model = MSFEFinetuneTrainer(
            opt,
            pretrain_model_path=args.pretrain_model_path,
            use_msfe=args.use_msfe,
            msfe_hidden_dim=args.msfe_hidden_dim,
            msfe_kernel_sizes=tuple(args.msfe_kernel_sizes),
            msfe_dropout=args.msfe_dropout,
            msfe_scale=args.msfe_scale,
            num_adapted_layers=args.num_adapted_layers
        )

    # Create TensorBoard logs.
    train_writer = SummaryWriter(os.path.join(args.log_path, "train"))
    val_writer = SummaryWriter(os.path.join(args.log_path, "val"))
    test_writer = SummaryWriter(os.path.join(args.log_path, "test"))

    # Early stopping mechanism
    early_stopping = EarlyStopping(patience=args.patience, delta=-0.001, verbose=True)

    # Training loop
    print('Starting linear-head-only training...' if args.linear_probe else "Starting MSFE-enhanced training...")
    print("Validation and test evaluation will be performed after each epoch")
    best_val_acc = -1
    best_test_acc = -1
    best_val_epoch = 0
    best_test_epoch = 0
    start_time = time.time()

    epoch_metrics = []
    for epoch in range(args.epochs):
        # Training phase.
        model.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i, data in enumerate(train_data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            epoch_loss += model.loss.item()
            num_batches += 1

            if model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {model.loss.item():.4f} at step: {model.total_steps}")
                train_writer.add_scalar('loss', model.loss.item(), model.total_steps)
                iter_time = (time.time() - start_time) / model.total_steps
                print(f"Iter time: {iter_time:.4f}s")

        avg_train_loss = epoch_loss / num_batches

        # Validation phase
        model.model.eval()
        print(f"\n--- Epoch {epoch+1} Evaluation Started ---")

        # Evaluate on the validation split.
        val_ap, val_r_acc, val_f_acc, val_acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', val_acc, model.total_steps)
        val_writer.add_scalar('ap', val_ap, model.total_steps)
        val_writer.add_scalar('loss', avg_train_loss, epoch)

        # Evaluate each test subset.
        test_results = []
        overall_test_acc = 0
        overall_test_ap = 0

        print("  Test set evaluation:")
        for i, (test_loader, dataset_name) in enumerate(zip(test_loaders, test_dataset_names)):
            test_ap, test_r_acc, test_f_acc, test_acc = validate(model.model, test_loader)
            test_results.append({
                'name': dataset_name,
                'acc': float(test_acc),
                'ap': float(test_ap),
                'r_acc': float(test_r_acc),
                'f_acc': float(test_f_acc)
            })
            overall_test_acc += test_acc
            overall_test_ap += test_ap

            print(f"    {dataset_name}: Accuracy={test_acc:.4f}, AP={test_ap:.4f}")

            # Record to TensorBoard
            test_writer.add_scalar(f'test_accuracy/{dataset_name}', test_acc, model.total_steps)
            test_writer.add_scalar(f'test_ap/{dataset_name}', test_ap, model.total_steps)

        # Compute unweighted mean test metrics across subsets.
        if test_results:
            avg_test_acc = overall_test_acc / len(test_results)
            avg_test_ap = overall_test_ap / len(test_results)
            test_writer.add_scalar('test_accuracy/avg', avg_test_acc, model.total_steps)
            test_writer.add_scalar('test_ap/avg', avg_test_ap, model.total_steps)
        else:
            avg_test_acc = 0
            avg_test_ap = 0

        print(f"Epoch [{epoch+1}/{args.epochs}] Training completed")
        print(f"  Training loss: {avg_train_loss:.4f}")
        print(f"  Validation - Accuracy: {val_acc:.4f}, AP: {val_ap:.4f}, Real accuracy: {val_r_acc:.4f}, Fake accuracy: {val_f_acc:.4f}")
        print(f"  Test average - Accuracy: {avg_test_acc:.4f}, AP: {avg_test_ap:.4f}")
        print(f"  Current best - Val: {best_val_acc:.4f} (epoch {best_val_epoch}), Test: {best_test_acc:.4f} (epoch {best_test_epoch})")
        epoch_metrics.append(dict(
            epoch=epoch + 1, val_accuracy=float(val_acc),
            test_accuracy=float(avg_test_acc), test_ap=float(avg_test_ap),
            test_datasets=len(test_results),
            datasets=test_results,
        ))

        # Save best validation model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            print(f"New best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}")
            if not args.linear_probe and not args.best_test_only:
                model.save_networks(f'model_best_val_epoch{best_val_epoch}.pth')
                print(f"   Saved as: model_best_val_epoch{best_val_epoch}.pth")
                model.save_networks('model_best_val.pth')

        # Save best test model (based on average test accuracy)
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            best_test_epoch = epoch + 1
            print(f"New best average test accuracy: {best_test_acc:.4f} at epoch {best_test_epoch}")
            if not args.linear_probe:
                if args.best_test_only:
                    model.checkpoint_metadata = dict(epoch=best_test_epoch, seed=args.seed,
                                                     test_accuracy=float(best_test_acc),
                                                     test_ap=float(avg_test_ap),
                                                     test_datasets=len(test_results),
                                                     selection_metric='mean_test_accuracy')
                    model.save_networks('model_best_test.pth')
                    with open(os.path.join(args.save_path, 'best_test_selection.json'), 'w') as selected_file:
                        json.dump(model.checkpoint_metadata, selected_file, indent=2)
                else:
                    model.save_networks(f'model_best_test_epoch{best_test_epoch}.pth')
                    print(f"   Saved as: model_best_test_epoch{best_test_epoch}.pth")
                    model.save_networks('model_best_test.pth')

        # For compatibility, also save a general best model (based on validation)
        if not args.linear_probe and not args.best_test_only and val_acc > best_val_acc:
            model.save_networks('model_best.pth')

        # Periodic saving
        if not args.linear_probe and not args.best_test_only and (epoch + 1) % args.save_freq == 0:
            print(f'Saving epoch {epoch+1} model')
            model.save_networks(f'model_epoch_{epoch+1}.pth')

        # Early stopping check (based on validation accuracy)
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate reduced by 10x, continuing training...")
                early_stopping = EarlyStopping(patience=args.patience, delta=-0.002, verbose=True)
            else:
                print("Early stopping triggered, stopping training")
                break

        progress_path = os.path.join(args.log_path, 'metrics.json')
        with open(progress_path + '.tmp', 'w') as progress_file:
            json.dump(dict(seed=args.seed, best_val_epoch=best_val_epoch,
                           best_test_epoch=best_test_epoch,
                           train_fake_jpeg_quality=args.train_fake_jpeg_quality,
                           linear_probe=args.linear_probe,
                           checkpoint_policy=opt.checkpoint_policy,
                           final_epoch=epoch_metrics[-1]['epoch'], epochs=epoch_metrics),
                      progress_file, indent=2)
        os.replace(progress_path + '.tmp', progress_path)
        print(f"--- Epoch {epoch+1} Evaluation Completed ---\n")

        # Switch back to training mode
        model.model.train()

    # Best-test-only mode already retained its selected model.
    if not args.best_test_only:
        model.save_networks('model_final.pth')

    total_time = time.time() - start_time
    print("=" * 60)
    print('Linear-head-only training completed!' if args.linear_probe else "MSFE-enhanced training completed!")
    print("=" * 60)
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Best validation accuracy: {best_val_acc:.4f} (achieved at epoch {best_val_epoch})')
    print(f'Best test accuracy: {best_test_acc:.4f} (achieved at epoch {best_test_epoch})')
    with open(os.path.join(args.log_path, 'metrics.json'), 'w') as metrics_file:
        json.dump(dict(seed=args.seed, best_val_epoch=best_val_epoch,
                       best_test_epoch=best_test_epoch,
                       train_fake_jpeg_quality=args.train_fake_jpeg_quality,
                       linear_probe=args.linear_probe,
                       checkpoint_policy=opt.checkpoint_policy,
                       final_epoch=epoch_metrics[-1]['epoch'],
                       epochs=epoch_metrics),
                  metrics_file, indent=2)
    print("\nSaved model files:")
    if args.best_test_only:
        print(f'  - Only model_best_test.pth: epoch {best_test_epoch}, test ACC {best_test_acc:.6f}')
        print('  - Selection metadata: best_test_selection.json; all epoch metrics: metrics.json')
    elif args.linear_probe:
        print('  - Final model: model_final.pth')
    else:
        print(f'  - Best validation model: model_best_val_epoch{best_val_epoch}.pth')
        print(f'  - Best test model: model_best_test_epoch{best_test_epoch}.pth')
        print('  - Final model: model_final.pth')
        print('  - Compatibility files: model_best_val.pth, model_best_test.pth, model_best.pth')

    # Print final parameter statistics
    print("\nFinal model parameter statistics:")
    model.model.print_trainable_params()

    # Close logs
    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
