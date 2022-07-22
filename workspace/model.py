from torch import nn
import torch
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

class CustomModel(nn.Module):
    def __init__(self, num_classes, model_name, pretrained=True):
        super(CustomModel, self).__init__()
        if model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x