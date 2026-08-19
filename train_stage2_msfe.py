
import os
import time
import argparse
import numpy as np
import torch
import random
from tensorboardX import SummaryWriter

def set_seed(seed=42):
    """Seed all random-number generators for reproducible experiments."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

set_seed(42)

from validate import validate
from data import create_dataloader
from data import get_ratio_subset
from earlystop import EarlyStopping
from networks.msfe_trainer import MSFEFinetuneTrainer
from options.train_options import TrainOptions


def create_balanced_subset_options(opt, real_count=3000, fake_count=3000):
    """Clone options and configure fixed real/fake sample counts."""

    class BalancedOpt:
        pass

    balanced_opt = BalancedOpt()

    for attr in dir(opt):
        if not attr.startswith('_') and not callable(getattr(opt, attr)):
            try:
                setattr(balanced_opt, attr, getattr(opt, attr))
            except (AttributeError, TypeError):
                pass

    balanced_opt.fixed_sampling = True
    balanced_opt.real_count = real_count
    balanced_opt.fake_count = fake_count

    if not hasattr(balanced_opt, 'serial_batches'):
        balanced_opt.serial_batches = False
    if not hasattr(balanced_opt, 'class_bal'):
        balanced_opt.class_bal = False
    if not hasattr(balanced_opt, 'ratio_sampling'):
        balanced_opt.ratio_sampling = False
    if not hasattr(balanced_opt, 'num_threads'):
        balanced_opt.num_threads = 4
    if not hasattr(balanced_opt, 'batch_size'):
        balanced_opt.batch_size = opt.batch_size

    return balanced_opt


def parse_args():
    """Parse stage-two training and MSFE configuration."""

    parser = argparse.ArgumentParser(description='Stage 2: MSFE-Enhanced Balanced Finetuning')

    parser.add_argument('--pretrain_model_path', type=str,
                       help='预训练模型权重路径（可选，如果不提供则使用原始CLIP权重）')

    parser.add_argument('--wang2020_data_path', type=str, required=True,
                       help='数据集根路径（包含train和val子目录）')
    parser.add_argument('--data_mode', type=str, default='wang2020',
                       choices=['ours', 'wang2020', 'ours_wang2020'],
                       help='数据模式')

    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--real_count', type=int, default=3000,
                       help='选取的真实图像数量（默认3000张）')
    parser.add_argument('--fake_count', type=int, default=3000,
                       help='选取的假图像数量（默认3000张）')

    parser.add_argument('--use_msfe', action='store_true', default=True,
                       help='是否使用MSFE适配器（默认True）')
    parser.add_argument('--msfe_hidden_dim', type=int, default=128,
                       help='MSFE适配器隐藏层维度（默认128）')
    parser.add_argument('--msfe_kernel_sizes', type=int, nargs='+', default=[3, 5, 7],
                       help='MSFE卷积核尺寸（默认[3, 5, 7]）')
    parser.add_argument('--msfe_dropout', type=float, default=0.1,
                       help='MSFE Dropout率（默认0.1）')
    parser.add_argument('--msfe_scale', type=float, default=0.1,
                       help='MSFE输出缩放因子（默认0.1）')
    parser.add_argument('--num_adapted_layers', type=int, default=24,
                       help='使用MSFE的层数（-1表示所有层，默认24）')

    parser.add_argument('--use_forgelens_style', action='store_true', default=False,
                       help='是否使用ForgeLens风格的数据处理（无额外数据增强，使用translate_duplicate）')

    parser.add_argument('--test_data_path', type=str,
                       default='/opt/data/private/ysb/NPR-DeepfakeDetection/dataset/GenImage',
                       help='测试数据集路径（默认使用Ojha数据集）')
    parser.add_argument('--test_max_sample', type=int, default=500,
                       help='每个测试数据集的采样数量（默认500）')

    parser.add_argument('--save_path', type=str, default='./checkpoints/stage2_msfe',
                       help='模型保存路径')
    parser.add_argument('--log_path', type=str, default='./logs/stage2_msfe',
                       help='日志保存路径')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--save_freq', type=int, default=5, help='保存频率（轮数）')

    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Print configuration information
    print("=" * 60)
    print("MSFE-Enhanced Stage 2 Fine-tuning Configuration")
    print("=" * 60)
    print(f"Use MSFE: {args.use_msfe}")
    print(f"MSFE hidden dimension: {args.msfe_hidden_dim}")
    print(f"MSFE kernel sizes: {args.msfe_kernel_sizes}")
    print(f"MSFE dropout: {args.msfe_dropout}")
    print(f"MSFE scale: {args.msfe_scale}")
    print(f"MSFE layers: {args.num_adapted_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training data: Real {args.real_count} + Fake {args.fake_count}")
    if args.pretrain_model_path:
        print(f"Pretrained model: {args.pretrain_model_path}")
    else:
        print("Pretrained model: Using original CLIP weights")
    print("=" * 60)

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
            self.arch = 'CLIP:ViT-L/14'
            self.batch_size = args.batch_size
            self.fix_backbone = False
            self.optim = 'adam'
            self.lr = args.lr
            self.beta1 = 0.9
            self.weight_decay = 0.01
            self.init_gain = 0.02
            self.gpu_ids = [0]
            self.loss_freq = 400
            self.use_forgelens_style = args.use_forgelens_style
            self.ratio_sampling = False
            self.real_ratio = 0.5
            self.fake_ratio = 0.5
            self.class_bal = False
            self.serial_batches = False
            self.fixed_sampling = False
            self.num_threads = 4
            self.checkpoints_dir = os.path.dirname(args.save_path)
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

    # Create data loader
    train_data_loader = create_dataloader(balanced_train_opt)
    print(f"Balanced training set size: {len(train_data_loader.dataset)}")

    class ValOpt:
        pass

    val_opt = ValOpt()

    for attr in dir(opt):
        if not attr.startswith('_') and not callable(getattr(opt, attr)):
            try:
                setattr(val_opt, attr, getattr(opt, attr))
            except (AttributeError, TypeError):
                pass

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

    from dataset_paths import DATASET_PATHS

    # Get all available test datasets
    test_datasets = [d for d in DATASET_PATHS if d]  # Filter out empty datasets
    print(f"Found {len(test_datasets)} test datasets")

    test_loaders = []
    test_dataset_names = []

    from validate import RealFakeDataset
    import torch.utils.data

    for dataset in test_datasets:
        dataset_name = dataset['key']
        print(f"  Preparing test dataset: {dataset_name}")

        try:
            test_dataset = RealFakeDataset(
                real_path=dataset['real_path'],
                fake_path=dataset['fake_path'],
                data_mode=dataset['data_mode'],
                max_sample=args.test_max_sample,
                arch=opt.arch
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=4
            )

            test_loaders.append(test_loader)
            test_dataset_names.append(dataset_name)
            print(f"    ✅ {dataset_name}: {len(test_dataset)} samples")

        except Exception as e:
            print(f"    ❌ {dataset_name}: Failed to create - {e}")
            import traceback
            traceback.print_exc()

    print(f"Successfully created {len(test_loaders)} test data loaders")

    # Create MSFE-enhanced fine-tuning model
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

    train_writer = SummaryWriter(os.path.join(args.log_path, "train"))
    val_writer = SummaryWriter(os.path.join(args.log_path, "val"))
    test_writer = SummaryWriter(os.path.join(args.log_path, "test"))

    # Early stopping mechanism
    early_stopping = EarlyStopping(patience=args.patience, delta=-0.001, verbose=True)

    # Training loop
    print("Starting MSFE-enhanced training...")
    print("Validation and test evaluation will be performed after each epoch")
    best_val_acc = -1
    best_test_acc = -1
    best_val_epoch = 0
    best_test_epoch = 0
    start_time = time.time()

    for epoch in range(args.epochs):
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

        val_ap, val_r_acc, val_f_acc, val_acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', val_acc, model.total_steps)
        val_writer.add_scalar('ap', val_ap, model.total_steps)
        val_writer.add_scalar('loss', avg_train_loss, epoch)

        test_results = []
        overall_test_acc = 0
        overall_test_ap = 0

        print("  Test set evaluation:")
        for i, (test_loader, dataset_name) in enumerate(zip(test_loaders, test_dataset_names)):
            try:
                test_ap, test_r_acc, test_f_acc, test_acc = validate(model.model, test_loader)
                test_results.append({
                    'name': dataset_name,
                    'acc': test_acc,
                    'ap': test_ap,
                    'r_acc': test_r_acc,
                    'f_acc': test_f_acc
                })
                overall_test_acc += test_acc
                overall_test_ap += test_ap

                print(f"    {dataset_name}: Accuracy={test_acc:.4f}, AP={test_ap:.4f}")

                # Record to TensorBoard
                test_writer.add_scalar(f'test_accuracy/{dataset_name}', test_acc, model.total_steps)
                test_writer.add_scalar(f'test_ap/{dataset_name}', test_ap, model.total_steps)

            except Exception as e:
                print(f"    {dataset_name}: Evaluation failed - {e}")

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

        # Save best validation model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            print(f"🎉 New best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}, saving best validation model...")
            model.save_networks(f'model_best_val_epoch{best_val_epoch}.pth')
            print(f"   Saved as: model_best_val_epoch{best_val_epoch}.pth")
            model.save_networks('model_best_val.pth')

        # Save best test model (based on average test accuracy)
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            best_test_epoch = epoch + 1
            print(f"🎉 New best average test accuracy: {best_test_acc:.4f} at epoch {best_test_epoch}, saving best test model...")
            model.save_networks(f'model_best_test_epoch{best_test_epoch}.pth')
            print(f"   Saved as: model_best_test_epoch{best_test_epoch}.pth")
            model.save_networks('model_best_test.pth')

        # For compatibility, also save a general best model (based on validation)
        if val_acc > best_val_acc:
            model.save_networks('model_best.pth')

        # Periodic saving
        if (epoch + 1) % args.save_freq == 0:
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

        print(f"--- Epoch {epoch+1} Evaluation Completed ---\n")

        # Switch back to training mode
        model.model.train()

    # Save final model
    model.save_networks('model_final.pth')

    total_time = time.time() - start_time
    print("=" * 60)
    print("MSFE-enhanced training completed!")
    print("=" * 60)
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Best validation accuracy: {best_val_acc:.4f} (achieved at epoch {best_val_epoch})')
    print(f'Best test accuracy: {best_test_acc:.4f} (achieved at epoch {best_test_epoch})')
    print("\nSaved model files:")
    print(f'  - Best validation model: model_best_val_epoch{best_val_epoch}.pth (validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch})')
    print(f'  - Best test model: model_best_test_epoch{best_test_epoch}.pth (test accuracy: {best_test_acc:.4f} at epoch {best_test_epoch})')
    print(f'  - Final model: model_final.pth')
    print(f'  - Compatibility files: model_best_val.pth, model_best_test.pth, model_best.pth')

    # Print final parameter statistics
    print("\nFinal model parameter statistics:")
    model.model.print_trainable_params()

    # Close logs
    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
