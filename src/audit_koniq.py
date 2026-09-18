"""Reproducible Milestone 1 audit for the KonIQ-10k dataset.

This module intentionally stops at dataset inspection. It does not train or
fine-tune a model. It produces the tables and figures used by the proposal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_COLUMNS = {"image_name", "MOS", "SD", "c_total", "set"}
SPLIT_ORDER = ["training", "validation", "test"]


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load and validate the released metadata CSV."""

    path = Path(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    frame["image_name"] = frame["image_name"].astype(str)
    frame["MOS"] = pd.to_numeric(frame["MOS"], errors="raise")
    frame["SD"] = pd.to_numeric(frame["SD"], errors="raise")
    frame["c_total"] = pd.to_numeric(frame["c_total"], errors="raise")
    frame["set"] = frame["set"].astype(str)
    return frame


def find_image_paths(images_root: str | Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Find images by basename and preserve collisions for the audit."""

    root = Path(images_root)
    grouped: dict[str, list[Path]] = {}
    if not root.exists():
        return {}, {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            grouped.setdefault(path.name, []).append(path)
    selected = {name: paths[0] for name, paths in grouped.items()}
    collisions = {name: paths for name, paths in grouped.items() if len(paths) > 1}
    return selected, collisions


def inspect_image_files(metadata: pd.DataFrame, images_root: str | Path) -> dict[str, Any]:
    """Audit image availability, readability, and dimensions."""

    image_paths, collisions = find_image_paths(images_root)
    expected = set(metadata["image_name"])
    dimensions: Counter[str] = Counter()
    unreadable: list[str] = []
    for name, path in image_paths.items():
        if name not in expected:
            continue
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
        except Exception:
            unreadable.append(name)

    found = expected & set(image_paths)
    missing = sorted(expected - set(image_paths))
    return {
        "expected_images": len(expected),
        "found_images": len(found),
        "missing_images": missing,
        "missing_count": len(missing),
        "unlisted_count": len(set(image_paths) - expected),
        "unreadable_images": sorted(unreadable),
        "unreadable_count": len(unreadable),
        "duplicate_basename_count": len(collisions),
        "dimensions": dict(dimensions),
        "image_paths": image_paths,
    }


def choose_representatives(metadata: pd.DataFrame, n_per_band: int = 3) -> pd.DataFrame:
    """Choose deterministic low, middle, and high MOS examples."""

    ordered = metadata.sort_values(["MOS", "image_name"]).reset_index(drop=True)
    n = max(n_per_band * 4, n_per_band)
    low = ordered.head(n).copy()
    high = ordered.tail(n).copy()
    median = float(ordered["MOS"].median())
    middle = ordered.assign(_distance=(ordered["MOS"] - median).abs()).sort_values(
        ["_distance", "image_name"]
    ).head(n).sort_values(["MOS", "image_name"])

    selected: list[pd.DataFrame] = []
    for label, band in (("Low quality", low), ("Middle quality", middle), ("High quality", high)):
        indexes = np.linspace(0, len(band) - 1, n_per_band).round().astype(int)
        subset = band.iloc[indexes].copy()
        subset["quality_band"] = label
        selected.append(subset)
    return pd.concat(selected, ignore_index=True)


def plot_representatives(
    representatives: pd.DataFrame,
    image_paths: dict[str, Path],
    output_path: str | Path,
) -> None:
    """Save a 3-by-3 representative sample grid."""

    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6))
    for ax, (_, row) in zip(axes.flat, representatives.iterrows()):
        name = row["image_name"]
        path = image_paths.get(name)
        if path is None:
            ax.text(0.5, 0.5, f"Missing\n{name}", ha="center", va="center")
            ax.set_facecolor("#F3F4F6")
        else:
            try:
                with Image.open(path) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(f"{row['quality_band']}\nMOS {row['MOS']:.1f}", fontsize=10)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Unreadable\n{exc}", ha="center", va="center")
                ax.set_facecolor("#F3F4F6")
        ax.axis("off")
    fig.suptitle("Representative KonIQ-10k images across the released MOS range", fontsize=15, y=0.98)
    fig.text(0.5, 0.02, "MOS is the released 0-100 target; lower values indicate lower perceived technical quality.",
             ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mos_distribution(metadata: pd.DataFrame, output_path: str | Path) -> None:
    """Save the required target-value distribution plot."""

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(metadata["MOS"], bins=24, color="#4F46E5", alpha=0.88, edgecolor="white", linewidth=0.7)
    mean = float(metadata["MOS"].mean())
    median = float(metadata["MOS"].median())
    ax.axvline(mean, color="#111827", linestyle="--", linewidth=1.6, label=f"Mean {mean:.1f}")
    ax.axvline(median, color="#D95F59", linestyle=":", linewidth=2.0, label=f"Median {median:.1f}")
    ax.set_title("Distribution of released mean opinion scores (MOS)", fontsize=15, pad=12)
    ax.set_xlabel("Released MOS (0-100 scale)")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_summary(metadata: pd.DataFrame, output_path: str | Path) -> None:
    """Save a compact split-count and split-mean summary figure."""

    summary = metadata.groupby("set", sort=False).agg(count=("image_name", "size"), mean_mos=("MOS", "mean"))
    summary = summary.reindex([s for s in SPLIT_ORDER if s in summary.index])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [s.title() for s in summary.index]
    axes[0].bar(labels, summary["count"], color=["#4F46E5", "#D99A29", "#2A9D8F"][: len(labels)])
    axes[0].set_title("Images per official split")
    axes[0].set_ylabel("Images")
    axes[0].grid(axis="y", alpha=0.18)
    axes[1].bar(labels, summary["mean_mos"], color=["#4F46E5", "#D99A29", "#2A9D8F"][: len(labels)])
    axes[1].set_title("Mean released MOS by split")
    axes[1].set_ylabel("MOS (0-100)")
    axes[1].grid(axis="y", alpha=0.18)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(metadata: pd.DataFrame, image_audit: dict[str, Any] | None) -> dict[str, Any]:
    """Build JSON-safe summary values for the proposal and README."""

    summary: dict[str, Any] = {
        "rows": int(len(metadata)),
        "columns": list(metadata.columns),
        "split_counts": {str(k): int(v) for k, v in metadata["set"].value_counts().to_dict().items()},
        "duplicate_metadata_filenames": int(metadata["image_name"].duplicated().sum()),
        "missing_metadata_values": {str(k): int(v) for k, v in metadata.isna().sum().to_dict().items()},
        "mos": {
            "min": float(metadata["MOS"].min()),
            "max": float(metadata["MOS"].max()),
            "mean": float(metadata["MOS"].mean()),
            "median": float(metadata["MOS"].median()),
            "std": float(metadata["MOS"].std()),
            "q25": float(metadata["MOS"].quantile(0.25)),
            "q75": float(metadata["MOS"].quantile(0.75)),
        },
        "rating_count": {
            "min": int(metadata["c_total"].min()),
            "max": int(metadata["c_total"].max()),
            "mean": float(metadata["c_total"].mean()),
            "median": float(metadata["c_total"].median()),
        },
        "rating_sd": {
            "min": float(metadata["SD"].min()),
            "max": float(metadata["SD"].max()),
            "mean": float(metadata["SD"].mean()),
            "median": float(metadata["SD"].median()),
        },
    }
    if image_audit is not None:
        summary["image_audit"] = {k: v for k, v in image_audit.items() if k != "image_paths"}
    return summary


def run_audit(metadata_path: str | Path, images_root: str | Path | None, output_dir: str | Path) -> dict[str, Any]:
    """Run the complete proposal-supporting audit."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)
    metadata[["image_name", "set", "MOS", "SD", "c_total"]].to_csv(output_dir / "split_manifest.csv", index=False)
    metadata.groupby("set", sort=False).agg(
        count=("image_name", "size"),
        mos_mean=("MOS", "mean"),
        mos_median=("MOS", "median"),
        mos_std=("MOS", "std"),
    ).reindex([s for s in SPLIT_ORDER if s in metadata["set"].unique()]).to_csv(output_dir / "split_summary.csv")

    plot_mos_distribution(metadata, output_dir / "mos_distribution.png")
    plot_split_summary(metadata, output_dir / "split_summary.png")

    image_audit: dict[str, Any] | None = None
    if images_root is not None and Path(images_root).exists():
        image_audit = inspect_image_files(metadata, images_root)
        representatives = choose_representatives(metadata)
        representatives.to_csv(output_dir / "representative_samples.csv", index=False)
        plot_representatives(representatives, image_audit["image_paths"], output_dir / "representative_samples.png")

    summary = build_summary(metadata, image_audit)
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to koniq10k_distributions_sets.csv")
    parser.add_argument("--images", default=None, help="Directory containing extracted KonIQ-10k images")
    parser.add_argument("--output-dir", default="figures", help="Directory for figures and audit tables")
    args = parser.parse_args()
    summary = run_audit(args.metadata, args.images, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
