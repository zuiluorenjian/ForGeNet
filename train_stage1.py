
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data.datasets import RealFakeDataset
from networks.pretrain_model import PretrainModel
from loss import SupConLoss
from options.train_options import TrainOptions
from earlystop import EarlyStopping


def parse_args():

    parser = argparse.ArgumentParser(description='Stage 1: Contrastive Pretraining')

    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14', help='CLIP模型架构')
    parser.add_argument('--output_dim', type=int, default=128, help='投影头输出维度')
    parser.add_argument('--hidden_dim', type=int, default=768, help='投影头隐藏层维度')
    parser.add_argument('--batch_norm', action='store_true', help='是否使用BatchNorm')

    parser.add_argument('--temperature', type=float, default=0.07, help='温度参数')
    parser.add_argument('--base_temperature', type=float, default=0.07, help='基础温度参数')
    parser.add_argument('--min_class', type=int, default=1, help='少数类标签（1=伪造图片）')
    parser.add_argument('--ratio_supervised_majority', type=float, default=0.0,
                       help='大类正样本对的比例（0.0=只关注少数类）')

    parser.add_argument('--use_ratio_sampling', action='store_true',
                       help='是否使用比例采样控制真假图片比例（默认使用全部不平衡数据）')
    parser.add_argument('--real_ratio', type=float, default=0.5,
                       help='真实图片选取比例（建议0.5创建平衡数据集，或调整为其他比例）')
    parser.add_argument('--fake_ratio', type=float, default=0.5,
                       help='伪造图片选取比例（建议0.5创建平衡数据集，或调整为其他比例）')

    parser.add_argument('--wang2020_data_path', type=str, required=True,
                       help='数据集根路径（包含train和val子目录）')
    parser.add_argument('--data_mode', type=str, default='wang2020',
                       choices=['ours', 'wang2020', 'ours_wang2020'],
                       help='数据模式')

    parser.add_argument('--save_path', type=str, default='./checkpoints/stage1',
                       help='模型保存路径')
    parser.add_argument('--log_path', type=str, default='./logs/stage1',
                       help='日志保存路径')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率（轮数）')

    return parser.parse_args()


def create_dataloader(opt, args, is_train=True):

    data_opt = opt
    data_opt.data_label = 'train' if is_train else 'val'
    data_opt.isTrain = is_train

    if is_train and args.use_ratio_sampling:
        data_opt.ratio_sampling = True
        data_opt.real_ratio = args.real_ratio
        data_opt.fake_ratio = args.fake_ratio
        print(f"使用比例采样：真实图片 {args.real_ratio*100:.1f}%, 伪造图片 {args.fake_ratio*100:.1f}%")
    else:
        data_opt.ratio_sampling = False
        if is_train:
            print("使用全部不平衡训练数据")

    dataset = RealFakeDataset(data_opt, is_pretraining=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train
    )

    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):

    model.train()
    model.projection_head.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        view1, view2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        all_images = torch.cat([view1, view2], dim=0)

        projections = model(all_images)

        proj1, proj2 = torch.split(projections, [batch_size, batch_size], dim=0)

        features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)

        loss, _, _ = criterion(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f}')

    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):

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

            loss, _, _ = criterion(features, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    class SimpleOpt:
        def __init__(self, args):
            self.data_label = 'train'
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
            self.ratio_sampling = False
            self.real_ratio = 0.5
            self.fake_ratio = 0.5

    opt = SimpleOpt(args)

    print("创建数据加载器...")
    train_loader = create_dataloader(opt, args, is_train=True)
    val_loader = create_dataloader(opt, args, is_train=False)

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")

    print("创建预训练模型...")
    model = PretrainModel(
        arch=args.arch,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        batch_norm=args.batch_norm
    ).to(device)

    model.unfreeze_projection_head()

    criterion = SupConLoss(
        temperature=args.temperature,
        base_temperature=args.base_temperature,
        min_class=args.min_class,
        ratio_supervised_majority=args.ratio_supervised_majority
    ).to(device)

    trainable_params = [p for p in model.projection_head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(args.log_path)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print("开始训练...")
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        val_loss = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Val Loss: {val_loss:.4f} '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': args
            }, os.path.join(args.save_path, 'best_model.pth'))
            print(f'保存最佳模型，验证损失: {val_loss:.4f}')

        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': args
            }, os.path.join(args.save_path, f'model_epoch_{epoch+1}.pth'))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'args': args
    }, os.path.join(args.save_path, 'final_model.pth'))

    total_time = time.time() - start_time
    print(f'训练完成！总时间: {total_time/3600:.2f} 小时')

    writer.close()


if __name__ == '__main__':
    main()
