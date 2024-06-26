import torchvision.models as models
import torch.nn as nn

class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiLabel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
