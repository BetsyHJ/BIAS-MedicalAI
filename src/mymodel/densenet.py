import torchvision.models as models
import torch.nn as nn

class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetMultiLabel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)
