from __future__ import annotations

import torch
from torchvision import models


def create_resnet18(num_classes: int = 2, pretrained: bool = False) -> torch.nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
