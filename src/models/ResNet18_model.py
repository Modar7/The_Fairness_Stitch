"""
This Python script contains some code adapted from a GitHub repository.

Original source: [https://github.com/yuzhenmao/Fairness-Finetuning/tree/main]

"""

import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import Normalize


class MyResNet(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(MyResNet, self).__init__()

        self.resnet18 = models.resnet18(weights=None)  # models.resnet18(pretrained=pretrain)
        # Replace last fc layer
        self.num_feats = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.num_feats, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def set_grad(self, val):
        for param in self.resnet18.parameters():
            param.requires_grad = val

    def get_feature_extractor(self):
        return nn.Sequential(*list(self.resnet18.children())[:-1])

    def get_features(self, x, norm=False):
        if norm:
            x = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x)
        features = self.get_feature_extractor()(x)
        return torch.reshape(features, (features.shape[0], -1))

    def append_last_layer(self, num_classes=2):
        num_out_features = self.num_feats
        self.out_fc = nn.Linear(num_out_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x

