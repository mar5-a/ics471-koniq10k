"""Image loading, M2 split preservation, and training-only augmentation."""

import hashlib
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

METADATA_URL = (
    "https://raw.githubusercontent.com/subpic/koniq/master/metadata/koniq10k_distributions_sets.csv"
)
ARCHIVE_URL = "https://zenodo.org/api/records/19500652/files/KonIQ-10k.zip/content"


def prepare_dataset(data_dir):
    """Reuse local data, or download the public archive directly on Colab."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "koniq10k_distributions_sets.csv"
    if not metadata_path.exists():
        print("Downloading MOS metadata...")
        urllib.request.urlretrieve(METADATA_URL, metadata_path)
    frame = pd.read_csv(metadata_path)
    image_paths = {p.name: p for p in data_dir.rglob("*.jpg")}
    if not set(frame.image_name).issubset(image_paths):
        archive = data_dir / "KonIQ-10k.zip"
        if not archive.exists():
            print("Downloading the 768 MB image archive; this may take a few minutes...")
            partial = archive.with_suffix(".zip.part")
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(ARCHIVE_URL, headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(request) as response:
                with partial.open("ab" if response.status == 206 else "wb") as destination:
                    shutil.copyfileobj(response, destination)
            partial.rename(archive)
        with archive.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "md5").hexdigest()
        if checksum != "ca88afc3569f4962f10b4efe45aa9ed7":
            raise ValueError("Image archive checksum differs from the published archive")
        with zipfile.ZipFile(archive) as images:
            images.extractall(data_dir)
        image_paths = {p.name: p for p in data_dir.rglob("*.jpg")}
    missing = set(frame.image_name) - image_paths.keys()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labeled images")
    frame["path"] = frame.image_name.map(image_paths)
    return frame


def milestone2_split(frame):
    """Exactly the same official-test + seeded development split as M2."""
    development = np.flatnonzero(frame["set"].to_numpy() != "test")
    test = np.flatnonzero(frame["set"].to_numpy() == "test")
    # Preserve the official test set and the exact seeded M2 development split.
    train, validation = train_test_split(development, test_size=0.2, random_state=42)
    return {
        "training": frame.iloc[np.sort(train)].reset_index(drop=True),
        "validation": frame.iloc[np.sort(validation)].reset_index(drop=True),
        "test": frame.iloc[test].reset_index(drop=True),
    }


class ImageQualityDataset(Dataset):
    def __init__(self, frame, transform, target_mean, target_std):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.transform(image)
        target = torch.tensor([(row.MOS - self.target_mean) / self.target_std], dtype=torch.float32)
        return pixels, target


def training_statistics(train_frame, size=(192, 256)):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    dataset = ImageQualityDataset(train_frame, transform, 0.0, 1.0)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    total, squared, pixels = (
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        0,
    )
    for images, _ in loader:
        images = images.double()
        total += images.sum((0, 2, 3))
        squared += images.square().sum((0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    # Accumulate only training pixels; validation and test never fit normalization.
    mean = total / pixels
    std = (squared / pixels - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float().tolist(), std.float().tolist()


def image_transforms(model_name, training, channel_mean=None, channel_std=None):
    if model_name == "scratch":
        size = (192, 256)
        mean, std = channel_mean, channel_std
    else:
        weights = {
            "vgg19": models.VGG19_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "inception_v3": models.Inception_V3_Weights.DEFAULT,
        }[model_name]
        preset = weights.transforms()
        mean, std = preset.mean, preset.std
        if not training:
            return preset
        size = (preset.crop_size[0], preset.crop_size[0])
    if training:
        # 80–100% AREA retained, with the source photograph's 4:3 aspect ratio.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(4 / 3, 4 / 3)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def make_loaders(
    splits,
    model_name,
    channel_mean,
    channel_std,
    target_mean,
    target_std,
    batch_size=64,
    workers=2,
    seed=42,
):
    train_transform = image_transforms(model_name, True, channel_mean, channel_std)
    eval_transform = image_transforms(model_name, False, channel_mean, channel_std)
    frames = {
        "training": splits["training"],
        "train_eval": splits["training"],
        "validation": splits["validation"],
        "test": splits["test"],
    }
    loaders = {}
    for name, frame in frames.items():
        dataset = ImageQualityDataset(
            frame,
            train_transform if name == "training" else eval_transform,
            target_mean,
            target_std,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "training",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(seed),
            persistent_workers=workers > 0,
        )
    return loaders
