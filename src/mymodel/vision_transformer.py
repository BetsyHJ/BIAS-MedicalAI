from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
)
import torch.nn as nn

class ViTMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(ViTMultiLabel, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads[-1] = nn.Linear(self.vit.heads[-1].in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
