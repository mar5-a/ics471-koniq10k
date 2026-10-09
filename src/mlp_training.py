"""Reusable training and evaluation with validation-only checkpoint selection."""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.mlp_models import MLP, make_loss, make_optimizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def regression_metrics(targets, predictions):
    """Metrics in original MOS units. Constant predictions have undefined SROCC."""
    targets = np.asarray(targets, dtype=np.float64).ravel()
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite regression target/prediction")
    error = predictions - targets
    ranks_y = pd.Series(targets).rank().to_numpy()
    ranks_p = pd.Series(predictions).rank().to_numpy()
    srocc = float("nan")
    if np.std(ranks_y) > 0 and np.std(ranks_p) > 0:
        srocc = float(np.corrcoef(ranks_y, ranks_p)[0, 1])
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "srocc": srocc}


def selection_key(metrics):
    """Higher SROCC wins; lower RMSE breaks ties. Undefined SROCC ranks last."""
    score = metrics["srocc"]
    return (score if np.isfinite(score) else -float("inf"), -metrics["rmse"])


@torch.no_grad()
def evaluate(model, loader, criterion, preprocessor, device):
    model.eval()
    predictions, targets = [], []
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predicted = model(x)
        loss_sum += float(criterion(predicted, y)) * len(x)
        predictions.append(predicted.cpu())
        targets.append(y.cpu())
    predictions = preprocessor.inverse_target(torch.cat(predictions)).numpy().ravel()
    targets = preprocessor.inverse_target(torch.cat(targets)).numpy().ravel()
    metrics = regression_metrics(targets, predictions)
    metrics["loss"] = loss_sum / len(loader.dataset)
    if not np.isfinite(metrics["loss"]):
        raise FloatingPointError("Nonfinite evaluation loss")
    return metrics, targets, predictions


@dataclass
class Trial:
    model: MLP
    history: pd.DataFrame
    best_epoch: int
    validation: dict
    settings: dict


def train_model(bundle, hidden_dims, loss_name, optimizer_name, lr,
                epochs=30, batch_size=64, l2=1e-4, seed=42, device="cpu"):
    """Fit only training data; select epoch by validation SROCC, never test.

    Objective = mean data loss + (l2 / 2) * sum(weight**2).
    Biases are excluded. Optimizer weight_decay stays zero so this exact penalty
    is consistent across SGD, RMSprop, and Adam. Plots show unpenalized data loss.
    Every epoch's train/validation metrics use the same end-of-epoch model.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0 or l2 < 0:
        raise ValueError("epochs/batch_size/lr must be positive and l2 nonnegative")
    seed_everything(seed)
    loaders = bundle.loaders(batch_size, seed)
    # Evaluation must not advance the training shuffle generator.
    train_evaluation = DataLoader(bundle.datasets["training"], batch_size=batch_size)
    input_dim = bundle.datasets["training"].tensors[0].shape[1]
    model = MLP(input_dim, hidden_dims).to(device)
    criterion = make_loss(loss_name)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    history, best_state, best_metrics, best_epoch = [], None, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loaders["training"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            data_loss = criterion(model(x), y)
            penalty = 0.5 * l2 * sum(p.square().sum() for p in model.parameters() if p.ndim > 1)
            objective = data_loss + penalty
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Training diverged at epoch {epoch}")
            objective.backward()
            optimizer.step()
        train_metrics, _, _ = evaluate(model, train_evaluation, criterion, bundle.preprocessor, device)
        validation, _, _ = evaluate(model, loaders["validation"], criterion, bundle.preprocessor, device)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in validation.items()}})
        if best_metrics is None or selection_key(validation) > selection_key(best_metrics):
            best_metrics, best_epoch = validation, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    settings = dict(input_dim=input_dim, hidden_dims=list(hidden_dims), loss=loss_name,
                    optimizer=optimizer_name, lr=lr, epochs=epochs, batch_size=batch_size,
                    l2=l2, seed=seed, device=str(device))
    return Trial(model, pd.DataFrame(history), best_epoch, best_metrics, settings)
