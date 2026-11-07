"""CNN regressors for KonIQ-10k: a small custom network and three pretrained models."""

import torch
from torch import nn
from torchvision import models


class ScratchCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        # Each convolution has its own BatchNorm and ReLU; pooling ends the block.
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, images):
        x = self.block1(images)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(self.pool(x))


def build_pretrained(name, mode, pretrained=True):
    """mode is 'head_only', 'last_three', or 'expanded_head'."""
    if name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(4096, 1)
        head_name = "classifier.6"
        last_three = ["classifier.0", "classifier.3", "classifier.6"]
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(2048, 1)
        head_name = "fc"
        last_three = ["layer4.2.conv2", "layer4.2.conv3", "fc"]
    elif name == "inception_v3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            init_weights=False,
            transform_input=True,
        )
        model.AuxLogits = None
        model.aux_logits = False
        model.fc = nn.Linear(2048, 1)
        head_name = "fc"
        # These are the two terminal convolutions on the deepest final-block branch.
        # They are parallel sibling layers; Inception has no single sequential tail.
        last_three = ["Mixed_7c.branch3x3dbl_3a.conv", "Mixed_7c.branch3x3dbl_3b.conv", "fc"]
    else:
        raise ValueError(f"Unknown model: {name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "expanded_head":
        input_features = 25088 if name == "vgg19" else 2048
        head = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        if name == "vgg19":
            model.classifier = head
            head_name = "classifier"
        else:
            model.fc = head
        trainable_layers = [head_name]
    else:
        trainable_layers = last_three if mode == "last_three" else [head_name]
        for layer_name in trainable_layers:
            for parameter in model.get_submodule(layer_name).parameters():
                parameter.requires_grad = True
    model.model_name = name
    model.transfer_mode = mode
    model.trainable_layers = trainable_layers
    model.head_name = "classifier" if name == "vgg19" else "fc"
    return model


def set_training_mode(model):
    if isinstance(model, ScratchCNN):
        model.train()
    else:
        # Eval mode freezes pretrained BatchNorm statistics. Conv/Linear gradients
        # still work in eval mode; requires_grad controls which weights are fitted.
        model.eval()
        head = model.get_submodule(model.head_name)
        if model.transfer_mode == "head_only" and model.model_name == "vgg19":
            head[-1].train()
        else:
            head.train()


def optimizer_for(model, learning_rate, weight_decay=1e-4):
    if isinstance(model, ScratchCNN):
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    head_parameters, pretrained_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            is_new = (
                name.startswith(model.head_name)
                if model.transfer_mode == "expanded_head"
                else name.startswith("classifier.6" if model.model_name == "vgg19" else "fc")
            )
            (head_parameters if is_new else pretrained_parameters).append(parameter)
    # Use a smaller step for pretrained weights than for the new prediction head.
    groups = [{"params": head_parameters, "lr": learning_rate}]
    if pretrained_parameters:
        groups.append({"params": pretrained_parameters, "lr": learning_rate * 0.1})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def trainable_layer_table(model):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "layer": name,
                "type": type(layer).__name__,
                "trainable_parameters": sum(
                    p.numel() for p in layer.parameters(recurse=False) if p.requires_grad
                ),
            }
            for name, layer in model.named_modules()
            if any(p.requires_grad for p in layer.parameters(recurse=False))
        ]
    )
