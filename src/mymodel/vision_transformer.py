# https://pytorch.org/vision/stable/models.html
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights
)
import torch.nn as nn

class ViTMultiLabel(nn.Module):
    def __init__(self, num_classes, weight_v='base'):
        super(ViTMultiLabel, self).__init__()
        if weight_v == 'base':
            weights = ViT_B_16_Weights.DEFAULT
        elif weight_v == 'base_swag':
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.vit = vit_b_16(weights=weights)
        self.vit.heads[-1] = nn.Linear(self.vit.heads[-1].in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
