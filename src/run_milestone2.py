"""Run the complete 3x3, four-LR, and shallow/deep KonIQ regression study.

Usage: python -m src.run_milestone2 --metadata ... --images ...
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.mlp_data import prepare_data, plot_samples
from src.mlp_models import DEEP, SHALLOW, LOSSES, OPTIMIZERS, make_loss
from src.mlp_training import evaluate, regression_metrics, selection_key, train_model


def plot_history(trial, path, title):
    history = trial.history
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric, label in zip(axes, ["loss", "mae", "srocc"],
                                 ["Data loss (standardized target)", "MAE (MOS)", "SROCC"]):
        ax.plot(history.epoch, history[f"train_{metric}"], label="Training")
        ax.plot(history.epoch, history[f"val_{metric}"], label="Validation")
        ax.axvline(trial.best_epoch, color="gray", linestyle=":", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_trial(trial, bundle, directory):
    directory.mkdir(parents=True, exist_ok=True)
    trial.history.to_csv(directory / "history.csv", index=False)
    torch.save({"model_state": {k: v.detach().cpu() for k, v in trial.model.state_dict().items()},
                "settings": trial.settings, "best_epoch": trial.best_epoch,
                "preprocessor": bundle.preprocessor.state_dict(),
                "image_size": bundle.image_size}, directory / "checkpoint.pt")


def trial_row(trial, run_id):
    return {"run": run_id, "loss": trial.settings["loss"],
            "optimizer": trial.settings["optimizer"], "lr": trial.settings["lr"],
            "best_epoch": trial.best_epoch,
            **{f"val_{k}": v for k, v in trial.validation.items()},
            "parameters": sum(p.numel() for p in trial.model.parameters()), "status": "ok"}


def markdown_table(frame):
    """Render tables without an optional tabulate dependency."""
    def cell(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def run_experiments(bundle, output_dir, epochs=30, batch_size=64, l2=1e-4,
                    seed=42, device="cpu", base_lr=1e-3,
                    learning_rates=(1e-4, 3e-4, 1e-3, 3e-3), metadata_path=None):
    """Complete controlled study; final test evaluation happens after all selection.

    An output directory is single-use to prevent accidentally mixing runs.
    Diverged trials remain in the comparison tables as failures, never winners.
    """
    if len(learning_rates) != 4 or len(set(learning_rates)) != 4 or min(learning_rates) <= 0:
        raise ValueError("Provide exactly four distinct positive learning rates")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a new, empty output directory: {output}")
    config = dict(epochs=epochs, batch_size=batch_size, l2=l2, seed=seed,
                  device=str(device), base_lr=base_lr, learning_rates=list(learning_rates),
                  image_size=bundle.image_size, shallow=SHALLOW, deep=DEEP,
                  selection="maximum validation SROCC; minimum RMSE breaks ties",
                  split_policy=("official test held out; remaining pool split 80/20"
                                if "official_set" in bundle.manifest else "random 64/16/20"),
                  split_counts=bundle.manifest.split.value_counts().to_dict(),
                  python=platform.python_version(), torch=str(torch.__version__),
                  numpy=np.__version__, pandas=pd.__version__, status="running")
    if metadata_path is not None:
        config["metadata_sha256"] = hashlib.sha256(Path(metadata_path).read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2))
    bundle.manifest.to_csv(output / "split_manifest.csv", index=False)
    plot_samples(bundle, output / "samples.png")

    def fit(run_id, hidden, loss, optimizer, lr):
        print(f"Training {run_id}: {loss}, {optimizer}, lr={lr:g}", flush=True)
        trial = train_model(bundle, hidden, loss, optimizer, lr, epochs,
                            batch_size, l2, seed, device)
        save_trial(trial, bundle, output / run_id)
        print(f"  epoch={trial.best_epoch}; validation SROCC={trial.validation['srocc']:.4f}; "
              f"MAE={trial.validation['mae']:.4f}", flush=True)
        return trial

    def sweep(specifications, filename):
        rows, best = [], None
        for run_id, loss, optimizer, lr in specifications:
            try:
                trial = fit(run_id, SHALLOW, loss, optimizer, lr)
                rows.append(trial_row(trial, run_id))
                if best is None or selection_key(trial.validation) > selection_key(best.validation):
                    best = trial
            except FloatingPointError as exc:
                rows.append(dict(run=run_id, loss=loss, optimizer=optimizer, lr=lr,
                                 status=f"failed: {exc}"))
                print(f"  FAILED: {exc}", flush=True)
            pd.DataFrame(rows).to_csv(output / filename, index=False)
        if best is None:
            raise RuntimeError(f"All trials in {filename} failed; inspect results")
        return pd.DataFrame(rows), best

    grid, best_pair = sweep(
        [(f"grid_{loss}_{opt}", loss, opt, base_lr) for loss in LOSSES for opt in OPTIMIZERS],
        "loss_optimizer_comparison.csv",
    )
    loss, optimizer = best_pair.settings["loss"], best_pair.settings["optimizer"]
    lr_table, shallow = sweep(
        [(f"lr_{i}", loss, optimizer, lr) for i, lr in enumerate(learning_rates)],
        "learning_rate_comparison.csv",
    )
    deep = fit("deep", DEEP, loss, optimizer, shallow.settings["lr"])
    save_trial(shallow, bundle, output / "shallow")
    # All architecture/hyperparameter/epoch choices are now locked.
    test_loader = bundle.loaders(batch_size, seed)["test"]
    test_rows = []
    for name, trial in [("shallow", shallow), ("deep", deep)]:
        metrics, targets, predictions = evaluate(
            trial.model, test_loader, make_loss(loss), bundle.preprocessor, device
        )
        test_rows.append(dict(network=name, hidden_layers=len(trial.settings["hidden_dims"]),
                              parameters=sum(p.numel() for p in trial.model.parameters()),
                              best_epoch=trial.best_epoch, **metrics))
        pd.DataFrame({"image_name": bundle.manifest.query("split == 'test'").image_name.to_numpy(),
                      "target_mos": targets, "predicted_mos": predictions}).to_csv(
                          output / f"{name}_test_predictions.csv", index=False)
        plot_history(trial, output / f"{name}_curves.png", f"{name.title()} MLP · {loss}/{optimizer}")
    baseline = regression_metrics(targets, np.full_like(targets, bundle.preprocessor.target_mean))
    test_rows.append(dict(network="training-mean baseline", hidden_layers=0, parameters=0,
                          best_epoch=0, **baseline))
    tests = pd.DataFrame(test_rows)
    tests.to_csv(output / "test_comparison.csv", index=False)
    delta_srocc = deep.validation["srocc"] - shallow.validation["srocc"]
    report = f"""# Milestone 2: KonIQ-10k MLP regression

## Protocol
Seed {seed}; {epochs} epochs per trial; batch size {batch_size}; device {device}.
Split policy: {config['split_policy']}.
RGB images resize to {bundle.image_size}, scale to [0,1], and use training-only
channel standardization. MOS targets use training-only mean/std normalization.
No rating-derived metadata is used as input. There are no classification labels;
stratification and class encoding are inapplicable. Samples span training MOS ranks.

Shallow hidden widths: {SHALLOW}; deep hidden widths: {DEEP}. ReLU hidden
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
    (output / "report.md").write_text(report)
    config["status"] = "complete"
    config["selected"] = dict(loss=loss, optimizer=optimizer, lr=shallow.settings["lr"])
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return {"grid": grid, "learning_rates": lr_table, "test": tests, "output_dir": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/koniq/images/KonIQ-10k/koniq10k_distributions_sets.csv")
    parser.add_argument("--images", default="data/koniq/images/KonIQ-10k/512x384")
    parser.add_argument("--output-dir", default="outputs/milestone2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    bundle = prepare_data(args.metadata, args.images, seed=args.seed)
    run_experiments(bundle, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
                    l2=args.l2, seed=args.seed, device=args.device, metadata_path=args.metadata)


if __name__ == "__main__":
    main()
