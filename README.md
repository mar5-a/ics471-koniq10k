<div align="center">
  <h1>koniq 10k image quality assessment</h1>
  <h2>ICS 471</h2>
  <p>deep learning course project · milestone 1</p>

  <p>
    <a href="proposal/proposal.pdf">proposal pdf</a> ·
    <a href="notebooks/01_koniq_audit.ipynb">audit notebook</a> ·
    <a href="figures/representative_samples.png">sample figure</a>
  </p>
</div>

## what this project is

we are planning a small image quality experiment using KonIQ 10k. given one real world photograph, the later model will estimate the technical quality that people assigned to it. it will not need a clean reference image, so the task is a no reference image quality problem.

this repository is for the first milestone of our ICS 471 course project. it contains the proposal, the dataset checks, and the figures used in the proposal. it does not train or fine tune a model yet.

## why we chose it

we wanted a computer vision problem that feels realistic but can still stay manageable for a course project. KonIQ 10k is a good fit because the photographs have natural issues such as blur, noise, poor exposure, and compression. the labels also come from human ratings, which makes the project more interesting than a simple object category task.

the planned model is intentionally simple: an ImageNet pretrained ResNet18 with one output value. the main question is whether the model can rank images in roughly the same order as human quality ratings.

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

2. download the 512x384 archive from Zenodo and extract the images under `data/koniq/images/`.

3. put `koniq10k_distributions_sets.csv` under `data/koniq/metadata/`. the notebook can also fetch the metadata from the authors' repository.

4. run the command line audit:

   ```bash
   python src/audit_koniq.py \
     --metadata data/koniq/metadata/koniq10k_distributions_sets.csv \
     --images data/koniq/images \
     --output-dir figures
   ```

or open `notebooks/01_koniq_audit.ipynb` and run the cells after the local data folders are ready.

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
requirements.txt                      audit only Python dependencies
```

## team members

<div align="center">

| name | student id |
| --- | ---: |
| Muhammad Ammar Sohail | 202356790 |
| Mohamed Kamaleldin | 202338790 |

</div>
