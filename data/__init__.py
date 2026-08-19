import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset



def get_bal_sampler(dataset):
    """Create a sampler that balances classes by inverse frequency."""
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def get_fixed_subset(dataset, real_count=3000, fake_count=3000):
    """Select fixed numbers of real and fake samples."""

    if hasattr(dataset, 'datasets'):
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Source dataset: {len(real_indices)} real, {len(fake_indices)} fake")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; returning the full dataset")
        return list(range(len(targets)))

    actual_real_count = min(real_count, len(real_indices))
    actual_fake_count = min(fake_count, len(fake_indices))

    np.random.seed(42)
    selected_real_indices = np.random.choice(real_indices, actual_real_count, replace=False)
    selected_fake_indices = np.random.choice(fake_indices, actual_fake_count, replace=False)

    subset_indices = np.concatenate([selected_real_indices, selected_fake_indices])

    final_real_ratio = actual_real_count / (actual_real_count + actual_fake_count)
    final_fake_ratio = actual_fake_count / (actual_real_count + actual_fake_count)

    print(f"Subset: {actual_real_count} real, {actual_fake_count} fake")
    print(f"Final ratio: {final_real_ratio*100:.1f}% real, {final_fake_ratio*100:.1f}% fake")
    print(f"Total training samples: {len(subset_indices)}")

    return subset_indices.tolist()


def get_ratio_subset(dataset, real_ratio=0.9, fake_ratio=0.1):
    """Select real and fake samples using independent retention ratios."""

    if real_ratio < 0 or real_ratio > 1 or fake_ratio < 0 or fake_ratio > 1:
        raise ValueError("real_ratio and fake_ratio must be between 0 and 1")

    if hasattr(dataset, 'datasets'):
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Source dataset: {len(real_indices)} real, {len(fake_indices)} fake")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; returning the full dataset")
        return list(range(len(targets)))

    num_real_selected = int(len(real_indices) * real_ratio)
    num_fake_selected = int(len(fake_indices) * fake_ratio)

    np.random.seed(42)
    selected_real_indices = np.random.choice(real_indices, num_real_selected, replace=False)
    selected_fake_indices = np.random.choice(fake_indices, num_fake_selected, replace=False)

    subset_indices = np.concatenate([selected_real_indices, selected_fake_indices])

    final_real_ratio = num_real_selected / (num_real_selected + num_fake_selected)
    final_fake_ratio = num_fake_selected / (num_real_selected + num_fake_selected)

    print(f"Subset: {num_real_selected} real ({real_ratio*100:.1f}% retained), "
          f"{num_fake_selected} fake ({fake_ratio*100:.1f}% retained)")
    print(f"Final ratio: {final_real_ratio*100:.1f}% real, {final_fake_ratio*100:.1f}% fake")
    print(f"Total training samples: {len(subset_indices)}")

    return subset_indices.tolist()


def get_ratio_sampler(dataset, real_ratio=0.9, fake_ratio=0.1):
    """Create a weighted sampler for a target real-to-fake ratio."""

    print("Warning: get_ratio_sampler uses weighted sampling; use get_ratio_subset for a fixed imbalanced subset")

    if abs(real_ratio + fake_ratio - 1.0) > 1e-6:
        print(f"Warning: normalizing real_ratio ({real_ratio}) and fake_ratio ({fake_ratio}) to sum to 1.0")
        total = real_ratio + fake_ratio
        real_ratio = real_ratio / total
        fake_ratio = fake_ratio / total

    if hasattr(dataset, 'datasets'):
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Dataset: {len(real_indices)} real, {len(fake_indices)} fake")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; using default sampling")
        return None

    sample_weights = np.zeros(len(targets))

    real_weight = real_ratio / len(real_indices)
    fake_weight = fake_ratio / len(fake_indices)

    total_weight = real_weight * len(real_indices) + fake_weight * len(fake_indices)
    real_weight = real_weight / total_weight
    fake_weight = fake_weight / total_weight

    sample_weights[real_indices] = real_weight
    sample_weights[fake_indices] = fake_weight

    print(f"Sampling weights: real={real_weight:.6f}, fake={fake_weight:.6f}")
    print(f"Expected batch ratio: {real_ratio*100:.1f}% real, {fake_ratio*100:.1f}% fake")

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_dataloader(opt, preprocess=None):
    """Build the project data loader from training options."""
    serial_batches = getattr(opt, 'serial_batches', False)
    is_train = getattr(opt, 'isTrain', True)
    class_bal = getattr(opt, 'class_bal', False)
    ratio_sampling = getattr(opt, 'ratio_sampling', False)
    fixed_sampling = getattr(opt, 'fixed_sampling', False)

    shuffle = not serial_batches if (is_train and not class_bal and not ratio_sampling and not fixed_sampling) else False
    dataset = RealFakeDataset(opt)
    if '2b' in opt.arch:
        dataset.transform = preprocess

    sampler = None
    if is_train:
        if fixed_sampling:
            real_count = getattr(opt, 'real_count', 3000)
            fake_count = getattr(opt, 'fake_count', 3000)

            subset_indices = get_fixed_subset(dataset, real_count, fake_count)
            sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            shuffle = False
            print(f"Using fixed-count sampling ({real_count} real, {fake_count} fake)")

        elif ratio_sampling:
            real_ratio = getattr(opt, 'real_ratio', 0.9)
            fake_ratio = getattr(opt, 'fake_ratio', 0.1)

            subset_indices = get_ratio_subset(dataset, real_ratio, fake_ratio)
            sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            shuffle = False
            print(f"Using ratio sampling ({real_ratio*100:.1f}% real, {fake_ratio*100:.1f}% fake)")
        elif class_bal:
            sampler = get_bal_sampler(dataset)
            shuffle = False
            print("Using class-balanced sampling")
    else:
        if class_bal:
            sampler = get_bal_sampler(dataset)
            shuffle = False

    batch_size = getattr(opt, 'batch_size', 64)
    num_threads = getattr(opt, 'num_threads', 4)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(num_threads)
    )
    return data_loader
