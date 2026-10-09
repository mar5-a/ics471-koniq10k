"""Configurable fully connected networks, regression losses, and optimizers."""

from torch import nn
from torch.optim import SGD, Adam, RMSprop

SHALLOW = (256, 128)
DEEP = (256, 128, 64, 32, 16)
LOSSES = ("mse", "mae", "huber")
OPTIMIZERS = ("sgd", "rmsprop", "adam")


class MLP(nn.Module):
    """Any number of hidden layers; linear outputs support regression.

    ReLU avoids sigmoid/tanh saturation and is inexpensive. A tapering width
    controls parameters for flattened images. The deep model retains the first
    two widths so extra depth is the main architecture change.
    """

    def __init__(self, input_dim, hidden_dims=SHALLOW, output_dim=1, activation=nn.ReLU):
        super().__init__()
        if not hidden_dims or any(width < 1 for width in (input_dim, *hidden_dims, output_dim)):
            raise ValueError("All layer widths must be positive; provide hidden layers")
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), activation()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.flatten(start_dim=1))


def make_loss(name):
    """Huber delta=1 is one training-target standard deviation."""
    factories = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": lambda: nn.HuberLoss(delta=1.0)}
    if name not in factories:
        raise ValueError(f"Unknown loss {name}; choose {tuple(factories)}")
    return factories[name]()


def make_optimizer(name, parameters, lr):
    """Explicit L2 is added to the objective in training, uniformly for all three."""
    factories = {
        "sgd": lambda: SGD(parameters, lr=lr, momentum=0.9),
        "rmsprop": lambda: RMSprop(parameters, lr=lr, alpha=0.99),
        "adam": lambda: Adam(parameters, lr=lr),
    }
    if name not in factories:
        raise ValueError(f"Unknown optimizer {name}; choose {tuple(factories)}")
    return factories[name]()
