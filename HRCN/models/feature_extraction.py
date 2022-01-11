# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/24 10:01
# @File-name       : feature_extraction.py
# @Description     :

import torch
import torch.nn as nn
import torchvision
from models.networks import sequential, set_requires_grad


class VGG_featureX(nn.Module):
    def __init__(self, fea_c = 64, norm=None, use_input_norm=False):
        super(VGG_featureX, self).__init__()
        if not norm:
            model = torchvision.models.vgg19(pretrained=True)
        else:
            model = torchvision.models.vgg19_bn(pretrained=True)
        self.use_input_norm = use_input_norm
        self.features = sequential(*list(model.features.children()))[0:(fea_c + 1)]
        set_requires_grad(self.features, requires_grad=False)

    def forward(self, x):
        if x.size()[1] < 3:
            # VGG for the net with in_channels = 3
            x = torch.cat([x, x, x], dim=1)
        if self.use_input_norm:
            x = (x - torch.mean(x))/ torch.std(x)
        else:
            x = x
        x = self.features(x)

        return x

