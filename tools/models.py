import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BACKBONES


def create_encoder(backbone):
    model = BACKBONES[backbone](pretrained=True)
    features_dim = model.fc.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, features_dim


class SupConModel(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=128, second_stage=False, num_classes=None, only_features=False):
        super(SupConModel, self).__init__()
        self.encoder, features_dim = create_encoder(backbone)
        self.second_stage = second_stage
        self.only_features = only_features
        self.embedding_dim = embedding_dim

        if self.second_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.classifier = nn.Linear(features_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.ReLU(inplace=True),
                nn.Linear(features_dim, self.embedding_dim))

    def forward(self, x):
        if self.second_stage:
            feat = self.encoder(x).squeeze()
            return self.classifier(feat)
        else:
            feat = self.encoder(x).squeeze()
            return F.normalize(self.head(feat), dim=1)

