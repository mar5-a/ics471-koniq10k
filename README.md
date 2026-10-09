<div align="center">
  <h1>koniq 10k image quality assessment</h1>
  <h2>ICS 471</h2>
  <p>deep learning course project · milestones 1 and 2</p>

  <p>
    <a href="proposal/proposal.pdf">proposal pdf</a> ·
    <a href="notebooks/01_koniq_audit.ipynb">audit notebook</a> ·
    <a href="notebooks/02_m2.ipynb">MLP experiments</a> ·
    <a href="figures/representative_samples.png">sample figure</a>
  </p>
</div>

## what this project is

we are planning a small image quality experiment using KonIQ 10k. given one real world photograph, the later model will estimate the technical quality that people assigned to it. it will not need a clean reference image, so the task is a no reference image quality problem.

this repository contains the proposal and dataset audit from milestone 1, plus the reusable PyTorch MLP pipeline and experiment notebook for milestone 2.

## why we chose it

we wanted a computer vision problem that feels realistic but can still stay manageable for a course project. KonIQ 10k is a good fit because the photographs have natural issues such as blur, noise, poor exposure, and compression. the labels also come from human ratings, which makes the project more interesting than a simple object category task.

the original proposal planned an ImageNet pretrained ResNet18 with one output value. milestone 2 instead implements shallow and deep MLPs to meet the assignment. the main question remains whether the model can rank images in roughly the same order as human quality ratings.

## dataset at a glance

| split | images |
| --- | ---: |
| training | 7,058 |
| validation | 1,000 |
| test | 2,015 |

the released metadata has 10,073 labeled images. the original ratings use a 1 to 5 scale, while the released MOS target is stored on a 0 to 100 scale. in the metadata we audited, MOS ranges from 3.91 to 88.39, with a mean of 58.73 and a median of 62.35.

the downloaded archive has 10,373 image files. 10,073 match the metadata, and the 300 extra files are ignored. the matched files used by the audit are readable 512x384 RGB images.

## what the milestone includes

- a short proposal with the problem, target, data description, split, metric, and later experiment plan
- a notebook that checks the metadata and local image archive
- a MOS distribution plot, split summary, and real sample image grid
- simple source code so the checks and figures can be regenerated

the main validation metric will be Spearman rank correlation, or SROCC. this makes sense here because we care about whether predicted quality rankings agree with human rankings. the test set stays untouched until the final evaluation stage.

## how to reproduce the audit

1. create an environment and install the small audit dependency set:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. download the 512x384 archive from Zenodo and extract the images under `data/koniq/images/KonIQ-10k/512x384/`.

3. put `koniq10k_distributions_sets.csv` under `data/koniq/images/KonIQ-10k/`. the notebook can also fetch the metadata from the authors' repository.

4. run the command line audit:

   ```bash
   python src/audit_koniq.py \
     --metadata data/koniq/images/KonIQ-10k/koniq10k_distributions_sets.csv \
     --images data/koniq/images/KonIQ-10k/512x384 \
     --output-dir figures
   ```

or open `notebooks/01_koniq_audit.ipynb` and run the cells after the local data folders are ready.

## milestone 2: reusable PyTorch MLP study

Open [`notebooks/02_m2.ipynb`](notebooks/02_m2.ipynb) and select the project
`.venv` Python kernel. The notebook explains the architecture, losses,
regularization, split policy, and all rubric decisions. Run every cell after
placing the real dataset in the paths documented in [`data/README.md`](data/README.md).

For a command-line run (Python 3.12 recommended):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run_milestone2 --output-dir outputs/milestone2
```

The study trains 14 models: a full 3-loss × 3-optimizer shallow comparison,
four learning rates for the selected pair, then a five-hidden-layer MLP.
The shallow network has hidden widths `(256, 128)`; the deeper network has
`(256, 128, 64, 32, 16)`. Both use ReLU and a scalar linear regression output.
The fixed L2 coefficient is `1e-4`, with the exact objective documented in the
notebook. Defaults are 30 epochs, batch size 64, seed 42, and CPU execution.
Use `--device cuda` or `--device mps` on supported hardware.

**Split policy:** the table above describes the original official splits.
For this assignment, the official test set stays fixed (2,015 images), while
the official train and validation sets are pooled and re-split 80/20 into
6,446 training and 1,612 validation images. This meets the requested validation
fraction. All fitted preprocessing uses only those 6,446 training images.
Without predefined splits, the reusable helper uses overall 64/16/20.
MOS is continuous: class encoding and classification stratification do not apply.
The helper supports stratification when reused with class labels.

**Selection:** maximize validation SROCC, with validation RMSE breaking ties.
MAE and RMSE are reported in original MOS units. Test evaluation happens only
after selecting the loss, optimizer, learning rate, and each model's epoch.
No numerical results are supplied until the actual dataset is run.

Each run saves under its output directory:

- `report.md`: comparison tables, settings, interpretation, and figures
- `loss_optimizer_comparison.csv`, `learning_rate_comparison.csv`, `test_comparison.csv`
- `shallow_curves.png`, `deep_curves.png`, and training sample images
- split manifest, version/configuration metadata, per-epoch histories,
  final test predictions, and checkpoints containing fitted preprocessing

The notebook creates a fresh folder under `outputs/milestone2/` for each run.
Command-line output directories must be empty to avoid mixing different experiments.
Generated outputs are ignored by Git; retain the report and executed notebook
with your submission. The notebook imports the reusable modules in `src/`;
update those modules when changing the data pipeline, models, or training.

Reusable modules:

| module | responsibility |
| --- | --- |
| `src/mlp_data.py` | metadata checks, splits, training-only normalization, tensors, DataLoaders, samples |
| `src/mlp_models.py` | configurable MLP, loss and optimizer factories |
| `src/mlp_training.py` | seeded training, metrics, validation checkpoint selection |
| `src/run_milestone2.py` | controlled comparisons, final test evaluation, plots and report |

Run correctness checks with `python -m unittest discover -s tests -v`. These
include an entire miniature 14-model study on synthetic images; synthetic
results are never reported as KonIQ results.

## sources and access

- image archive: [KonIQ 10k on Zenodo](https://zenodo.org/records/19500652), listed as CC BY 4.0
- metadata and authors' reference code: [subpic/koniq](https://github.com/subpic/koniq)
- dataset paper: [KonIQ 10k paper](https://arxiv.org/abs/1910.06180)
- authors' project files: [OSF project](https://osf.io/hcsdy/)

the raw image archive is not committed to this repository. individual source photographs may have attribution requirements in addition to the archive level license, so the notebook creates local figures after the data is downloaded.

## repository layout

```text
proposal/proposal.pdf                 final 2 to 3 page proposal
notebooks/01_koniq_audit.ipynb        reproducible inspection notebook
src/audit_koniq.py                    audit and figure generation code
figures/                               generated plots and audit tables
assets/kfupm_logo.png                 proposal PDF branding
data/README.md                        download and local data layout
notebooks/02_m2.ipynb                 milestone 2 experiments and explanation
src/mlp_*.py                         reusable PyTorch pipeline and training
src/run_milestone2.py                 full milestone 2 experiment runner
tests/test_milestone2.py              pipeline and experiment correctness checks
requirements.txt                     audit and PyTorch dependencies
```

## team members

<div align="center">

| name | student id |
| --- | ---: |
| Muhammad Ammar Sohail | 202356790 |
| Mohamed Kamaleldin | 202338790 |

</div>
