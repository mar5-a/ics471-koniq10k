"""Build the reproducible Milestone 1 audit notebook."""

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build(output: str | Path = "notebooks/01_koniq_audit.ipynb") -> None:
    cells = [
        md(
            """# KonIQ-10k Milestone 1 audit

This notebook supports the course proposal for blind/no-reference image quality assessment. It inspects the released metadata and local image archive, creates the required sample and distribution figures, and records the planned split and validation metric.

**Scope note:** this notebook intentionally does not train or fine-tune a neural network."""
        ),
        code(
            '''from pathlib import Path
import sys
import urllib.request
import pandas as pd
from IPython.display import Image as IPImage, Markdown, display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_koniq import (
    inspect_image_files,
    load_metadata,
    run_audit,
)

METADATA_URL = (
    "https://raw.githubusercontent.com/subpic/koniq/master/metadata/"
    "koniq10k_distributions_sets.csv"
)
metadata_path = PROJECT_ROOT / "data/koniq/metadata/koniq10k_distributions_sets.csv"
images_root = PROJECT_ROOT / "data/koniq/images"
output_dir = PROJECT_ROOT / "figures"
metadata_path.parent.mkdir(parents=True, exist_ok=True)
if not metadata_path.exists():
    print("Metadata not found locally; downloading the small metadata file...")
    urllib.request.urlretrieve(METADATA_URL, metadata_path)
print("Metadata:", metadata_path)
print("Images:", images_root, "(present:", images_root.exists(), ")")'''
        ),
        md("## 1. Metadata and target audit"),
        code(
            '''metadata = load_metadata(metadata_path)
print(f"Rows: {len(metadata):,}")
display(metadata.head())

summary = metadata.groupby("set", sort=False).agg(
    images=("image_name", "size"),
    mos_mean=("MOS", "mean"),
    mos_median=("MOS", "median"),
    mos_std=("MOS", "std"),
)
display(summary)

display(metadata[["MOS", "SD", "c_total"]].describe().T)
print("Duplicate filenames:", int(metadata["image_name"].duplicated().sum()))
print("Missing metadata values:")
display(metadata.isna().sum().to_frame("missing"))'''
        ),
        md(
            """The original ratings use a 1-5 quality scale. The released CSV stores the derived MOS target on a 0-100 scale; this project uses that released numeric target consistently. It is a continuous target, so the task is regression rather than classification."""
        ),
        md("## 2. Required distribution plot"),
        code(
            '''summary_json = run_audit(metadata_path, images_root if images_root.exists() else None, output_dir)
display(IPImage(filename=str(output_dir / "mos_distribution.png")))
display(IPImage(filename=str(output_dir / "split_summary.png")))'''
        ),
        md("## 3. Representative real samples"),
        code(
            '''if images_root.exists():
    image_audit = inspect_image_files(metadata, images_root)
    print("Expected images:", image_audit["expected_images"])
    print("Found images:", image_audit["found_images"])
    print("Missing images:", image_audit["missing_count"])
    print("Unlisted archive images ignored:", image_audit["unlisted_count"])
    print("Unreadable images:", image_audit["unreadable_count"])
    print("Dimensions:", image_audit["dimensions"])
    display(pd.read_csv(output_dir / "representative_samples.csv"))
    display(IPImage(filename=str(output_dir / "representative_samples.png")))
else:
    display(Markdown("Image archive not found. Download and extract the 512x384 archive under data/koniq/images, then rerun this cell."))'''
        ),
        md(
            """## 4. Pre-registered experimental plan

- **Input:** one RGB photograph from the 512x384 release.
- **Output:** one continuous MOS prediction on the released 0-100 scale.
- **Split:** use the released official split: 7,058 training, 1,000 validation, and 2,015 test images. The test set remains untouched until final evaluation.
- **Primary validation metric:** Spearman rank correlation (SROCC), which measures agreement between the ordering of predicted and human quality scores.
- **Planned model:** ImageNet-pretrained ResNet18 with a one-value regression head.
- **Planned controlled experiments:** frozen versus limited fine-tuning, then 224x224 versus 512x384 input if compute permits.

No training is performed in this milestone notebook."""
        ),
        code(
            '''def spearman_rank_correlation(y_true, y_pred):
    """Compute SROCC without requiring a training framework."""
    true_ranks = pd.Series(y_true).rank(method="average").to_numpy()
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy()
    return float(pd.Series(true_ranks).corr(pd.Series(pred_ranks)))

print("Metric selected: Spearman rank correlation (SROCC)")'''
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


if __name__ == "__main__":
    build()
