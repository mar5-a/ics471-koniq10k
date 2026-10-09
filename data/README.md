# Local KonIQ-10k data

The dataset is intentionally not committed to Git. Download the 512x384 archive from [Zenodo](https://zenodo.org/records/19500652), then extract the image files under:

```text
data/koniq/images/KonIQ-10k/512x384/
```

Place the authors' metadata CSV at:

```text
data/koniq/images/KonIQ-10k/koniq10k_distributions_sets.csv
```

The metadata file can be downloaded from the authors' repository:

<https://raw.githubusercontent.com/subpic/koniq/master/metadata/koniq10k_distributions_sets.csv>

Milestone 2 uses these same paths. Nested image directories are supported;
only filenames listed in the metadata are used. Missing images or duplicate
matching basenames cause an error instead of silently changing the dataset.
Keep the `set` column so the official test set remains held out. The remaining
training/validation pool is re-split 80/20 for this assignment. If `set` is absent
for another dataset, the pipeline uses an 80/20 train/test split followed by an
80/20 train/validation split.

For a dataset stored elsewhere, change `METADATA` and `IMAGES` in
`notebooks/02_m2.ipynb`, or pass `--metadata /path/to/file.csv --images /path/to/images`
to `python -m src.run_milestone2`.

Do not redistribute the raw image collection through this repository. Check the individual source-image attribution requirements before sharing example images outside the course context.
