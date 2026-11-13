import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import RealFakeDataset



def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def get_fixed_subset(dataset, real_count=3000, fake_count=3000, seed=None):
    """Select fixed counts of real and AI-generated samples without replacement."""
    # Collect labels according to the dataset type.
    if hasattr(dataset, 'datasets'):
        # ConcatDataset layout.
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        # UniversalFakeDetect RealFakeDataset layout.
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    # Labels: 0=real, 1=AI-generated.
    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Source dataset: real images {len(real_indices)}, AI-generated images {len(fake_indices)}")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; returning the full dataset")
        return list(range(len(targets)))

    # Cap requested counts at the number of available images.
    actual_real_count = min(real_count, len(real_indices))
    actual_fake_count = min(fake_count, len(fake_indices))

    # Select the requested counts without replacement.
    if seed is None:
        np.random.seed(42)  # Retain the original sampling behavior.
        generator = np.random
    else:
        generator = np.random.RandomState(seed)
        if hasattr(dataset, 'total_list'):
            real_indices = np.array(sorted(real_indices, key=lambda i: dataset.total_list[i]))
            fake_indices = np.array(sorted(fake_indices, key=lambda i: dataset.total_list[i]))
    selected_real_indices = generator.choice(real_indices, actual_real_count, replace=False)
    selected_fake_indices = generator.choice(fake_indices, actual_fake_count, replace=False)

    # Combine the selected indices.
    subset_indices = np.concatenate([selected_real_indices, selected_fake_indices])

    # Report the resulting class ratio.
    final_real_ratio = actual_real_count / (actual_real_count + actual_fake_count)
    final_fake_ratio = actual_fake_count / (actual_real_count + actual_fake_count)

    print(f"Selected subset: real images {actual_real_count}, AI-generated images {actual_fake_count}")
    print(f"Final training ratio: real images {final_real_ratio*100:.1f}%, AI-generated images {final_fake_ratio*100:.1f}%")
    print(f"Total training samples: {len(subset_indices)}")

    return subset_indices.tolist()


def get_ratio_subset(dataset, real_ratio=0.9, fake_ratio=0.1):
    """Select real and AI-generated subsets using independent retention fractions."""
    # Validate the retention fractions.
    if real_ratio < 0 or real_ratio > 1 or fake_ratio < 0 or fake_ratio > 1:
        raise ValueError("real_ratio and fake_ratio must be between 0 and 1")

    # Inspect the dataset layout.
    if hasattr(dataset, 'datasets'):
        # ConcatDataset layout.
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        # UniversalFakeDetect RealFakeDataset layout.
        # Build targets from total_list and labels_dict.
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    # Labels: 0=real, 1=AI-generated.
    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Source dataset: real images {len(real_indices)}, AI-generated images {len(fake_indices)}")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; returning the full dataset")
        return list(range(len(targets)))

    # Compute the number of samples to retain from each class.
    num_real_selected = int(len(real_indices) * real_ratio)
    num_fake_selected = int(len(fake_indices) * fake_ratio)

    # Select each class subset without replacement.
    np.random.seed(42)  # Retain the original NumPy subset-sampling seed.
    selected_real_indices = np.random.choice(real_indices, num_real_selected, replace=False)
    selected_fake_indices = np.random.choice(fake_indices, num_fake_selected, replace=False)

    # Combine the selected indices.
    subset_indices = np.concatenate([selected_real_indices, selected_fake_indices])

    # Report the resulting class ratio.
    final_real_ratio = num_real_selected / (num_real_selected + num_fake_selected)
    final_fake_ratio = num_fake_selected / (num_real_selected + num_fake_selected)

    print(f"Selected subset: real images {num_real_selected} ({real_ratio*100:.1f}% retention), "
          f"AI-generated images {num_fake_selected} ({fake_ratio*100:.1f}% retention)")
    print(f"Final training ratio: real images {final_real_ratio*100:.1f}%, AI-generated images {final_fake_ratio*100:.1f}%")
    print(f"Total training samples: {len(subset_indices)}")

    return subset_indices.tolist()


def get_ratio_sampler(dataset, real_ratio=0.9, fake_ratio=0.1):
    """Build a weighted ratio sampler; use get_ratio_subset for a fixed imbalanced subset."""
    print("Warning: get_ratio_sampler uses weighted sampling; use get_ratio_subset for a fixed imbalanced subset")

    # Validate the retention fractions.
    if abs(real_ratio + fake_ratio - 1.0) > 1e-6:
        print(f"Warning: real_ratio ({real_ratio}) + fake_ratio ({fake_ratio}) != 1.0, normalizing automatically")
        total = real_ratio + fake_ratio
        real_ratio = real_ratio / total
        fake_ratio = fake_ratio / total

    # Inspect the dataset layout.
    if hasattr(dataset, 'datasets'):
        # ConcatDataset layout.
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        targets = np.array(targets)
    else:
        # UniversalFakeDetect RealFakeDataset layout.
        # Build targets from total_list and labels_dict.
        targets = []
        for i, img_path in enumerate(dataset.total_list):
            label = dataset.labels_dict[img_path]
            targets.append(label)
        targets = np.array(targets)

    # Labels: 0=real, 1=AI-generated.
    real_indices = np.where(targets == 0)[0]
    fake_indices = np.where(targets == 1)[0]

    print(f"Dataset: real images {len(real_indices)}, AI-generated images {len(fake_indices)}")

    if len(real_indices) == 0 or len(fake_indices) == 0:
        print("Warning: a class is missing; using default sampling")
        return None

    # Compute one sampling weight per example.
    sample_weights = np.zeros(len(targets))

    # Class weights are proportional to target fraction divided by class count.
    real_weight = real_ratio / len(real_indices)
    fake_weight = fake_ratio / len(fake_indices)

    # Normalize weights to obtain the requested class proportions.
    total_weight = real_weight * len(real_indices) + fake_weight * len(fake_indices)
    real_weight = real_weight / total_weight
    fake_weight = fake_weight / total_weight

    # Assign per-example weights.
    sample_weights[real_indices] = real_weight
    sample_weights[fake_indices] = fake_weight

    print(f"Sampling weights: real {real_weight:.6f}, AI-generated {fake_weight:.6f}")
    print(f"Expected batch ratio: real images {real_ratio*100:.1f}%, AI-generated images {fake_ratio*100:.1f}%")

    # Create the weighted random sampler.
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_dataloader(opt, preprocess=None):
    # Read optional settings with safe defaults.
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
        # Apply sampling controls to the training split.
        if fixed_sampling:
            # Fixed counts create a balanced training subset.
            real_count = getattr(opt, 'real_count', 3000)
            fake_count = getattr(opt, 'fake_count', 3000)

            subset_indices = get_fixed_subset(dataset, real_count, fake_count,
                                              seed=getattr(opt, 'data_seed', None))
            sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            shuffle = False  # A custom sampler and shuffle=True cannot be combined.
            print(f"Fixed-count sampling (real: {real_count}, AI-generated: {fake_count})")

        elif ratio_sampling:
            # Retention fractions create an imbalanced subset.
            real_ratio = getattr(opt, 'real_ratio', 0.9)
            fake_ratio = getattr(opt, 'fake_ratio', 0.1)

            subset_indices = get_ratio_subset(dataset, real_ratio, fake_ratio)
            sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            shuffle = False  # A custom sampler and shuffle=True cannot be combined.
            print(f"Ratio sampling (real retention: {real_ratio*100:.1f}%, AI-generated retention: {fake_ratio*100:.1f}%)")
        elif class_bal:
            # Optional inverse-frequency class balancing.
            sampler = get_bal_sampler(dataset)
            shuffle = False
            print("Using class-balanced sampling")
    else:
        # Optional class balancing for evaluation.
        if class_bal:
            sampler = get_bal_sampler(dataset)
            shuffle = False
            # Optional class-balanced evaluation sampling.

    # Read data-loader settings with safe defaults.
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
