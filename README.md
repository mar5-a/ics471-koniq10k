# KonIQ-10k image quality assessment

ICS 471 course project by Muhammad Ammar Sohail (202356790) and Mohamed
Kamaleldin (202338790).

We predict a photograph's perceived technical quality without a reference
image. The target is **MOS (Mean Opinion Score)**: an aggregate of human quality
ratings. The released metadata used here expresses MOS on a 0–100 scale.

## Milestones

- **Milestone 1:** [proposal](proposal/proposal.pdf),
  [dataset audit](notebooks/01_koniq_audit.ipynb), and sample figures in `figures/`.
- **Milestone 2:** [executed submission notebook](notebooks/02_m2_submission.ipynb).
  All model code, comparison tables, and plots are visible in the notebook.
  [02_m2.ipynb](notebooks/02_m2.ipynb) is the earlier notebook that imports `src/`.
- **Milestone 3:** [self-contained CNN notebook](notebooks/03_m3_submission.ipynb) with scratch training, transfer learning, partial fine-tuning, and SVR features. All outputs are embedded from a completed Colab A100 run.

Milestone 2 compares two-hidden-layer and five-hidden-layer MLPs. The selected
settings were MSE loss, SGD with momentum, and learning rate 0.003.

| Model | Test SROCC ↑ | Test MAE ↓ | Test RMSE ↓ |
| --- | ---: | ---: | ---: |
| Shallow MLP | 0.4710 | 10.6558 | 13.4859 |
| Deep MLP | 0.4790 | 10.4517 | 13.2535 |
| Scratch CNN (M3) | 0.7560 | 7.3368 | 9.6200 |
| ResNet50, last three layers (M3) | 0.7567 | 7.7461 | 9.6390 |
| ResNet50 features + SVR (M3) | 0.7427 | 7.6207 | 9.8275 |

MAE and RMSE are in MOS units. These results come from one seed. The CNNs also use higher-resolution inputs than the MLPs.

## Dataset and splits

The metadata contains 10,073 labeled images. We retain the official 2,015-image
test set and split the remaining pool 80/20 with seed 42, matching Milestone 2:

| Training | Validation | Test |
| ---: | ---: | ---: |
| 6,446 | 1,612 | 2,015 |

Milestone 3 reuses the same image assignments. Preprocessing statistics are
fitted on training data; validation results select settings and checkpoints.
Test results are reported after those choices are fixed.

Download the [512×384 image archive](https://zenodo.org/records/19500652) and the
[authors' metadata CSV](https://raw.githubusercontent.com/subpic/koniq/master/metadata/koniq10k_distributions_sets.csv).
Use this local layout, or change the two dataset paths in the notebook:

```text
data/koniq/images/KonIQ-10k/
├── koniq10k_distributions_sets.csv
└── 512x384/
    └── image files
```

The dataset and generated checkpoints are ignored by Git. Extra archive images
that are absent from the metadata are excluded. A local symlink to the dataset
works on this computer; a remote Colab runtime needs its own accessible copy.

## Running Milestone 2

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/02_m2_submission.ipynb`, set `METADATA` and `IMAGES` in the setup
cell, and run the cells in order. The notebook contains its reusable functions
and classes directly and needs no local report helper. Its existing outputs
show the completed real-data run. Rerunning trains 14 models and saves
checkpoints and curves under `outputs/milestone2/`.

The modular implementation remains available for reuse:

| File | Purpose |
| --- | --- |
| `src/mlp_data.py` | Data preparation, split, normalization, and DataLoaders |
| `src/mlp_models.py` | Shallow/deep MLPs, losses, and optimizers |
| `src/mlp_training.py` | Training, evaluation, and checkpoint selection |
| `src/run_milestone2.py` | Command-line experiments and report generation |
| `src/audit_koniq.py` | Dataset inspection and Milestone 1 figures |

Run the modular study with:

```bash
python -m src.run_milestone2 --output-dir outputs/milestone2_new
```

## Running Milestone 3

Open `notebooks/03_m3_submission.ipynb` and choose a Colab A100 kernel. All
implementation code is included in its cells. Missing data is downloaded
directly from the public archive into `/content/koniq`; no Mac dataset upload
or repository clone is needed. Run the cells in order.

Results are written to `/content/milestone3` on Colab, or
`outputs/milestone3` locally. Each trial saves its validation-selected
checkpoint and per-epoch CSV log. The final export contains the selected
scratch CNN, early-stopped scratch CNN, selected transfer CNN, SVR, tables, logs, and plots. Download this
archive before the temporary Colab runtime disconnects, and save the executed
notebook with its outputs visible.

Reusable M3 code is also in `src/cnn_data.py`, `src/cnn_models.py`, and
`src/cnn_training.py`. The notebook includes these definitions directly.

Optional existing M2 checks are in `tests/`.

## Sources

- [KonIQ-10k dataset and rating description](https://database.mmsp-kn.de/koniq-10k-database.html)
- [Dataset paper](https://arxiv.org/abs/1910.06180)
- [Authors' metadata and reference code](https://github.com/subpic/koniq)

The raw dataset is not included in this repository. See the dataset sources
for access and image licensing information.
