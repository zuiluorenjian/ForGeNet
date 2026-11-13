"""Labeled test folders and evaluation transforms."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FullImageResize:
    """Exact OpenSDI path: OpenCV linear resize, Albumentations CLIP normalize."""

    def __init__(self, size=224):
        import albumentations as albu
        import cv2
        from albumentations.pytorch import ToTensorV2

        self.transform = albu.Compose([
            albu.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            albu.Normalize(mean=MEAN, std=STD),
            albu.Crop(0, 0, size, size),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=np.asarray(image))["image"]


def build_transform(name):
    if name == "resize224":
        return FullImageResize(224)
    if name in ("center_crop224", "center224"):
        return transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    raise ValueError(f"Unknown evaluation transform: {name}")


def folder_manifest(root, transform="center_crop224"):
    """Collect labeled test images, optionally grouped by generator subfolder."""
    root = Path(root)
    subsets = ([root] if (root / "0_real").is_dir() and (root / "1_fake").is_dir()
               else sorted(path for path in root.iterdir() if path.is_dir()))
    entries = []
    for subset in subsets:
        samples = []
        for directory, subdirectories, filenames in os.walk(subset, followlinks=True):
            subdirectories.sort()
            parts = Path(directory).relative_to(root).parts
            label = 0 if "0_real" in parts else 1 if "1_fake" in parts else None
            if label is None:
                continue
            for filename in sorted(filenames):
                if Path(filename).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
                    samples.append([str((Path(directory) / filename).relative_to(root)), label])
        if samples:
            entries.append(dict(name=subset.name, root="data", transform=transform, samples=samples))
    if not entries:
        raise ValueError("No labeled images found; the test set must contain 0_real and 1_fake folders")
    return dict(format="forgenet-evaluation-v1", datasets=entries)


class ManifestDataset(Dataset):
    def __init__(self, entry, roots):
        self.entry = entry
        root_key = entry["root"]
        if not roots.get(root_key):
            raise ValueError(f"Supply --{root_key}-root for {entry['name']}")
        self.root = Path(roots[root_key])
        self.samples = entry["samples"]
        self.transform = build_transform(entry["transform"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(self.root / path) as source:
            image = source.convert("RGB")
        return self.transform(image), label
