import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbones import BACKBONES


def create_encoder(backbone):
    try:
        if 'timm_' in backbone:
            backbone = backbone.split('_')[-1]
            timm.create_model(model_name=backbone, pretrained=True)
        else:
            model = BACKBONES[backbone](pretrained=True)
    except RuntimeError or KeyError:
        raise RuntimeError('Specify the correct backbone name. Either one of torchvision backbones, or a timm backbone.'
                           'For timm - add prefix \'timm_\'. For instance, timm_resnet18')

    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Can\'t find the linear layer of the model')

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, features_dim


class SupConModel(nn.Module):
    def __init__(self, backbone='resnet50', projection_dim=128, second_stage=False, num_classes=None):
        super(SupConModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(backbone)
        self.second_stage = second_stage
        self.projection_head = True
        self.projection_dim = projection_dim
        self.embed_dim = projection_dim

        if self.second_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.classifier = nn.Linear(self.features_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.features_dim, self.projection_dim))

    def use_projection_head(self, mode):
        self.projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x):
        if self.second_stage:
            feat = self.encoder(x).squeeze()
            return self.classifier(feat)
        else:
            feat = self.encoder(x).squeeze()
            if self.projection_head:
                return F.normalize(self.head(feat), dim=1)
            else:
                return F.normalize(feat, dim=1)
