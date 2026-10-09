"""Reusable, leakage-free image regression data pipeline for Milestone 2."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.audit_koniq import find_image_paths


def split_indices(n, predefined=None, labels=None, seed=42):
    """Keep an official test set; split development 80/20.

    Without predefined sets, split 80/20 first (overall 64/16/20).
    For reuse with classification, pass encoded class labels to stratify BOTH
    random splits. Regression callers must leave labels=None.
    """
    indices = np.arange(n)
    labels = None if labels is None else np.asarray(labels)
    if labels is not None and len(labels) != n:
        raise ValueError("Class labels must match the number of rows")
    if predefined is None:
        development, test = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
    else:
        sets = np.asarray(predefined)
        if len(sets) != n or not set(sets).issubset({"training", "validation", "test"}):
            raise ValueError("Predefined sets must contain training/validation/test labels")
        test = indices[sets == "test"]
        development = indices[sets != "test"]
        if len(test) < 2:
            raise ValueError("Predefined split must contain at least two test samples")
    train, validation = train_test_split(
        development, test_size=0.2, random_state=seed,
        stratify=None if labels is None else labels[development],
    )
    # Stable metadata order also keeps saved prediction filenames aligned.
    splits = {"training": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
    if any(len(ids) < 2 for ids in splits.values()):
        raise ValueError("Each split needs at least two samples for correlation")
    return splits


@dataclass
class Preprocessor:
    """Training-only RGB channel and target standardization statistics."""

    channel_mean: torch.Tensor
    channel_std: torch.Tensor
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, images, targets):
        # Double-precision accumulation avoids precision loss in large images.
        mean = images.double().mean(dim=(0, 2, 3), keepdim=True).float()
        std = images.double().std(dim=(0, 2, 3), correction=0, keepdim=True).float()
        target_std = float(targets.double().std(correction=0))
        if target_std < 1e-8:
            raise ValueError("Training targets must have nonzero variance")
        return cls(mean, std.clamp_min(1e-6), float(targets.double().mean()), target_std)

    def transform(self, images, targets):
        x = ((images - self.channel_mean) / self.channel_std).flatten(start_dim=1)
        y = (targets - self.target_mean) / self.target_std
        return TensorDataset(x.float(), y.float().reshape(-1, 1))

    def inverse_target(self, targets):
        return targets * self.target_std + self.target_mean

    def state_dict(self):
        return {
            "channel_mean": self.channel_mean, "channel_std": self.channel_std,
            "target_mean": self.target_mean, "target_std": self.target_std,
        }


@dataclass
class DataBundle:
    datasets: dict
    preprocessor: Preprocessor
    manifest: pd.DataFrame
    paths: dict
    image_size: tuple

    def loaders(self, batch_size=64, seed=42):
        """Fresh RNG per experiment gives identical minibatch order across runs."""
        return {
            name: DataLoader(
                dataset, batch_size=batch_size, shuffle=name == "training",
                generator=torch.Generator().manual_seed(seed), num_workers=0,
                drop_last=False,
            )
            for name, dataset in self.datasets.items()
        }


def prepare_data(metadata_path, images_root, image_size=(64, 48), seed=42):
    """Load RGB images, partition before fitting, and return PyTorch datasets.

    image_size is (width, height); 64x48 preserves the release's 4:3 ratio.
    Only pixels are inputs: rating proportions, SD, and c_total leak the target.
    Extra archive images are ignored; missing or ambiguous images fail loudly.
    """
    frame = pd.read_csv(metadata_path)
    if not {"image_name", "MOS"}.issubset(frame.columns):
        raise ValueError("Metadata requires image_name and MOS columns")
    if frame.image_name.isna().any() or frame.image_name.duplicated().any():
        raise ValueError("Image names must be present and unique")
    frame["image_name"] = frame.image_name.astype(str)
    frame["MOS"] = pd.to_numeric(frame.MOS, errors="raise")
    if not np.isfinite(frame.MOS).all() or not frame.MOS.between(0, 100).all():
        raise ValueError("Expected finite released MOS targets in [0, 100]")
    splits = split_indices(len(frame), frame["set"] if "set" in frame else None, seed=seed)
    paths, collisions = find_image_paths(images_root)
    missing = set(frame.image_name) - paths.keys()
    ambiguous = set(frame.image_name) & collisions.keys()
    if missing or ambiguous:
        raise ValueError(f"Missing {len(missing)} images; ambiguous basenames {len(ambiguous)}. "
                         f"Examples: {sorted(missing | ambiguous)[:5]}")
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("image_size must have two positive dimensions")
    pixels = torch.empty(len(frame), 3, image_size[1], image_size[0], dtype=torch.float32)
    for i, name in enumerate(frame.image_name):
        try:
            with Image.open(paths[name]) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB").resize(
                    image_size, Image.Resampling.BILINEAR
                )
                pixels[i] = torch.from_numpy(np.array(rgb, dtype=np.float32)).permute(2, 0, 1) / 255
        except Exception as exc:
            raise ValueError(f"Unable to decode image {name}") from exc
    targets = torch.tensor(frame.MOS.to_numpy(), dtype=torch.float32)
    preprocessor = Preprocessor.fit(pixels[splits["training"]], targets[splits["training"]])
    datasets = {
        name: preprocessor.transform(pixels[ids], targets[ids])
        for name, ids in splits.items()
    }
    manifest = frame[["image_name", "MOS"]].copy()
    if "set" in frame:
        manifest["official_set"] = frame["set"]
    manifest["split"] = ""
    for name, ids in splits.items():
        manifest.loc[ids, "split"] = name
    return DataBundle(datasets, preprocessor, manifest, paths, tuple(image_size))


def plot_samples(bundle, output_path):
    """Regression has no classes: display training examples across MOS ranks."""
    import matplotlib.pyplot as plt

    train = bundle.manifest.query("split == 'training'").sort_values("MOS")
    examples = train.iloc[[0, len(train) // 2, len(train) - 1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, band, (_, row) in zip(axes, ["Low", "Middle", "High"], examples.iterrows()):
        with Image.open(bundle.paths[row.image_name]) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f"{band} quality · MOS {row.MOS:.2f}")
        ax.axis("off")
    fig.suptitle("Training samples (continuous regression target; no classes)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
