# KonIQ-10k Image Quality Assessment - Milestone 1

Proposal-supporting code and artifacts for a deep learning course project on blind/no-reference image quality assessment.

## Project in one sentence

Given one real-world RGB photograph, predict its human-perceived technical quality score without a pristine reference image.

## Why this project

KonIQ-10k is more substantial than a toy image-classification dataset because its images contain authentic, mixed quality issues such as blur, noise, exposure problems, and compression artifacts. At the same time, the course implementation can stay compact: a pretrained CNN with one scalar output is enough for the planned experimental phase.

The first milestone is intentionally limited to dataset inspection and an experimental plan. This repository does not train or fine-tune a model.

## Dataset audit

The released metadata contains 10,073 images and an official split:

| Split | Images |
| --- | ---: |
| Training | 7,058 |
| Validation | 1,000 |
| Test | 2,015 |

The original crowd ratings use a 1-5 quality scale. The released metadata stores the derived MOS target on a 0-100 scale. The audit notebook uses that released MOS consistently; lower values mean lower perceived technical quality. In the current metadata, MOS ranges from 3.91 to 88.39, with mean 58.73 and median 62.35.

The downloaded archive contains 10,373 image files, while 10,073 have metadata rows and labels. The audit uses the 10,073 matched images and ignores the 300 extra unlisted files.

The planned validation metric is Spearman rank correlation (SROCC), because image-quality assessment is primarily concerned with whether predicted quality rankings agree with human rankings.

## Sources and access

- Image archive: [KonIQ-10k on Zenodo](https://zenodo.org/records/19500652) - 512x384 archive, listed as CC BY 4.0, approximately 768 MB.
- Metadata and authors' reference code: [subpic/koniq](https://github.com/subpic/koniq).
- Dataset paper: [KonIQ-10k paper](https://arxiv.org/abs/1910.06180).
- Authors' project files: [OSF project](https://osf.io/hcsdy/).

The raw image archive is not committed to this repository. Individual source images may have attribution requirements in addition to the archive-level license, so the notebook regenerates local figures after download rather than redistributing the image collection.

## Repository layout

```text
proposal/proposal.pdf                 Final 2-3 page proposal
notebooks/01_koniq_audit.ipynb        Reproducible inspection notebook
src/audit_koniq.py                    Audit and figure-generation code
figures/                               Generated plots and audit tables
data/README.md                        Download and local data layout
requirements.txt                      Audit-only Python dependencies
```

## Reproduce the audit

1. Create an environment and install the audit dependencies:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Download the 512x384 archive from Zenodo and extract its images under `data/koniq/images/`.

3. Put `koniq10k_distributions_sets.csv` under `data/koniq/metadata/`. The notebook can also fetch the metadata from the authors' GitHub repository.

4. Run the notebook or the command-line audit:

   ```bash
   python src/audit_koniq.py \
     --metadata data/koniq/metadata/koniq10k_distributions_sets.csv \
     --images data/koniq/images \
     --output-dir figures
   ```

The generated figures include a MOS distribution, split summary, and representative low/middle/high quality image samples.

## Planned modeling scope after Milestone 1

The proposed later model is an ImageNet-pretrained ResNet18 with its final layer replaced by a one-value regression head. Candidate controlled experiments are frozen features versus limited fine-tuning and 224x224 versus 512x384 input. These are plans only; no training is included in this milestone repository.

## Team

Current project owner: Muhammad Ammar Sohail (student ID 202356790). The second teammate will be added to the proposal once their name and student ID are confirmed; their GitHub handle is `mkamaleldin7`.
