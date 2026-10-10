"""Markdown report formatting for the Milestone 2 notebook.

The data pipeline, models, training, selection, and plots stay in the notebook.
This helper only formats the resulting tables and settings as Markdown.
"""

import numpy as np


def markdown_table(frame):
    """Render tables without an optional tabulate dependency."""
    def cell(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_report(grid, lr_table, tests, *, shallow, deep, image_size,
                 split_policy, epochs, batch_size, l2, seed, device, base_lr):
    """Format the completed study without training models or exporting data."""
    loss = shallow.settings['loss']
    optimizer = shallow.settings['optimizer']
    shallow_widths = tuple(shallow.settings['hidden_dims'])
    deep_widths = tuple(deep.settings['hidden_dims'])
    delta_srocc = deep.validation['srocc'] - shallow.validation['srocc']
    return f"""# Milestone 2: KonIQ-10k MLP regression

## Protocol
Seed {seed}; {epochs} epochs per trial; batch size {batch_size}; device {device}.
Split policy: {split_policy}.
RGB images resize to {image_size}, scale to [0,1], and use training-only
channel standardization. MOS targets use training-only mean/std normalization.
No rating-derived metadata is used as input. There are no classification labels;
stratification and class encoding are inapplicable. Samples span training MOS ranks.

Shallow hidden widths: {shallow_widths}; deep hidden widths: {deep_widths}. ReLU hidden
activations provide inexpensive nonlinearities; the linear scalar output predicts
standardized MOS without clipping. The taper limits model size. Resizing reduces
compute but can remove subtle quality artifacts, a limitation of this baseline.

Explicit L2 strength is **{l2:g}**: objective = mean data loss + lambda/2 times
the sum of squared weights (biases excluded). Optimizer weight decay is zero.
MSE emphasizes large errors; MAE reduces their influence; Huber (delta=1 in
standardized MOS) combines quadratic and linear behavior. SGD momentum=0.9;
RMSprop alpha=0.99; Adam uses PyTorch defaults. All trials use the same splits,
initialization seed, batch order, epoch budget, and regularization.

## 3 × 3 loss/optimizer comparison
Base learning rate: {base_lr:g}. Epochs and combinations are chosen by maximum
validation SROCC, then minimum RMSE for ties. Loss magnitudes across different
loss functions are not directly comparable. Undefined SROCC ranks last.

{markdown_table(grid)}

## Four learning rates
Selected pair: **{loss} + {optimizer}**.

{markdown_table(lr_table)}

Selected learning rate: **{shallow.settings['lr']:g}**.

## Held-out test comparison
The same selected loss, optimizer and learning rate are used for both networks.
Each network restores its own best validation epoch. MAE/RMSE are in original
MOS units. The constant training-mean baseline has undefined SROCC by definition.

{markdown_table(tests)}

Deep minus shallow validation SROCC: {delta_srocc:.6f}.
Deep minus shallow test SROCC: {tests.iloc[1].srocc - tests.iloc[0].srocc:.6f}.
Deep minus shallow test MAE: {tests.iloc[1].mae - tests.iloc[0].mae:.6f}.
Positive SROCC differences favor deep; negative MAE differences favor deep.
These are single-seed results, not evidence of statistical significance. Test
results are reported after selection and must not be used to retune this study.

## Required learning curves
Loss is the unpenalized data loss on standardized MOS, evaluated at each epoch's
final weights for both splits. MAE is in MOS units; SROCC is also shown. Dotted
lines identify selected checkpoints, which need not be the last epochs.

![Training samples](samples.png)
![Shallow loss and performance](shallow_curves.png)
![Deep loss and performance](deep_curves.png)
"""
