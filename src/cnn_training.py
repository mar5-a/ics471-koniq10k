"""Plain training, validation, checkpointing, and plots for image-quality regression."""

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from src.cnn_models import ScratchCNN, optimizer_for, set_training_mode


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_scores(targets, predictions):
    targets, predictions = np.asarray(targets).ravel(), np.asarray(predictions).ravel()
    errors = predictions - targets
    correlation = (
        float(spearmanr(targets, predictions).statistic)
        if np.std(predictions) > 0
        else float("nan")
    )
    return {
        "srocc": correlation,
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
    }


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std):
    model.eval()
    predictions, targets, loss_sum = [], [], 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)
        loss_sum += nn.functional.mse_loss(output.float(), labels, reduction="sum").item()
        predictions.append(output.float().cpu().numpy())
        targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions).ravel() * target_std + target_mean
    targets = np.concatenate(targets).ravel() * target_std + target_mean
    scores = regression_scores(targets, predictions)
    scores["loss"] = loss_sum / len(loader.dataset)
    return scores, targets, predictions


def train_model(
    model,
    loaders,
    settings,
    output_dir,
    device,
    target_mean,
    target_std,
    channel_mean,
    channel_std,
    epochs=15,
    patience=None,
):
    """Train on training images; restore the checkpoint with best validation SROCC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = optimizer_for(model, settings["learning_rate"])
    evaluation_transform = loaders["validation"].dataset.transform
    image_mean, image_std = channel_mean, channel_std
    if not isinstance(model, ScratchCNN):
        image_mean, image_std = evaluation_transform.mean, evaluation_transform.std
    history, best_scores, best_epoch, stale_epochs = [], None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_training_mode(model)
        for images, labels in loaders["training"]:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predictions = model(images)
                loss = nn.functional.mse_loss(predictions.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {output_dir.name}")
            loss.backward()
            optimizer.step()
        # Measure both splits without augmentation or dropout for comparable curves.
        train_scores, _, _ = evaluate(model, loaders["train_eval"], device, target_mean, target_std)
        val_scores, _, _ = evaluate(model, loaders["validation"], device, target_mean, target_std)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_scores.items()},
                **{f"val_{k}": v for k, v in val_scores.items()},
                "seconds": time.perf_counter() - start,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        current_score = np.nan_to_num(val_scores["srocc"], nan=-np.inf)
        previous_score = (
            -np.inf if best_scores is None else np.nan_to_num(best_scores["srocc"], nan=-np.inf)
        )
        improved = best_scores is None or (current_score, -val_scores["rmse"]) > (
            previous_score,
            -best_scores["rmse"],
        )
        # Save by validation ranking, using RMSE only to break an exact tie.
        if improved:
            best_scores, best_epoch, stale_epochs = val_scores.copy(), epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "settings": settings,
                    "best_epoch": best_epoch,
                    "validation": best_scores,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "channel_mean": image_mean,
                    "channel_std": image_std,
                    "evaluation_preprocessing": repr(evaluation_transform),
                    "seed": 42,
                    "crop_area": [0.8, 1.0],
                    "weight_decay": 1e-4,
                    "epoch_budget": epochs,
                    "patience": patience,
                    "batch_size": loaders["training"].batch_size,
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        print(
            f"{output_dir.name} | {epoch:02d}/{epochs} | "
            f"train MAE {train_scores['mae']:.3f} | val MAE {val_scores['mae']:.3f} | "
            f"val SROCC {val_scores['srocc']:.4f} | {history[-1]['seconds']:.0f}s",
            flush=True,
        )
        if patience is not None and stale_epochs >= patience:
            print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.cpu()
    result = {
        **settings,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        **{f"val_{k}": v for k, v in best_scores.items()},
        "seconds": time.perf_counter() - start,
        "checkpoint": str(output_dir / "best.pt"),
    }
    pd.DataFrame([result]).to_csv(output_dir / "result.csv", index=False)
    return result, pd.DataFrame(history)


def plot_history(history, best_epoch, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, metric, label in zip(
        axes, ["loss", "mae", "srocc"], ["MSE (standardized MOS)", "MAE (MOS units)", "SROCC"]
    ):
        ax.plot(history.epoch, history[f"train_{metric}"], label="Training")
        ax.plot(history.epoch, history[f"val_{metric}"], label="Validation")
        ax.axvline(best_epoch, color="0.4", linestyle=":", label="Best checkpoint")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def extract_features(model, loader, device, target_mean, target_std):
    """Read the input to the saved model's final scalar prediction layer."""
    model.eval().to(device)
    final_layer = [layer for layer in model.modules() if isinstance(layer, nn.Linear)][-1]
    captured = []
    handle = final_layer.register_forward_pre_hook(
        lambda layer, inputs: captured.append(inputs[0].detach().float().cpu())
    )
    labels = []
    try:
        for images, targets in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(images.to(device))
            labels.append(targets.numpy())
    finally:
        handle.remove()
        model.cpu()
    return torch.cat(captured).numpy(), np.concatenate(labels).ravel() * target_std + target_mean
