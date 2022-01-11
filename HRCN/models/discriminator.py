# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/23 11:07
# @File-name       : discriminator.py
# @Description     :
from models.networks import ConvBlock, sequential
import torch.nn as nn

class Discriminator_VGG_64(nn.Module):
    def __init__(self, in_c=3, mid_c=128, ks=3, stride=1, padding=1, norm='batch', act='mish'):
        super(Discriminator_VGG_64, self).__init__()
        ks2 = 2 * ks
        padding2 = padding * 2
        # output: N, C, 32, 32
        conv_0 = ConvBlock(in_c=in_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_1 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 16 16
        conv_2 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_3 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 8 8
        conv_4 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_5 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 4 4
        conv_6 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_7 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # sequential, output:
        self.features = sequential(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7)
        # classifier
        self.classifier = sequential(nn.Linear(mid_c * 4 * 4, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        # keep N constant
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_c=3, mid_c=128, ks=3, stride=1, padding=1, norm='batch', act='mish'):
        super(Discriminator_VGG_96, self).__init__()
        ks2 = 2 * ks
        padding2 = padding * 2
        # output: N, C, 48, 48
        conv_0 = ConvBlock(in_c=in_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_1 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 24 24
        conv_2 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_3 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 12 12
        conv_4 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_5 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 6 6
        conv_6 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_7 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 3 3
        conv_8 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_9 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # sequential, output:
        self.features = sequential(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9)
        # classifier
        self.classifier = sequential(nn.Linear(mid_c * 3 * 3, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        # keep N constant
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_c=3, mid_c=128, ks=3, stride=1, padding=1, norm='batch', act='prelu'):
        super(Discriminator_VGG_128, self).__init__()
        ks2 = 2 * ks
        padding2 = padding * 2
        # output: N, C, 64, 64
        conv_0 = ConvBlock(in_c=in_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_1 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 32 32
        conv_2 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_3 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 16 16
        conv_4 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_5 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 8 8
        conv_6 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_7 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 4 4
        conv_8 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_9 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # sequential, output:
        self.features = sequential(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9)
        # classifier
        self.classifier = sequential(nn.Linear(mid_c * 4 * 4, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_c=3, mid_c=128, ks=3, stride=1, padding=1, norm='batch', act='mish'):
        super(Discriminator_VGG_192, self).__init__()
        ks2 = 2 * ks
        padding2 = padding * 2
        # output: N, C, 96, 96
        conv_0 = ConvBlock(in_c=in_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_1 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 48 48
        conv_2 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_3 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 24 24
        conv_4 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_5 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 12 12
        conv_6 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_7 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 6 6
        conv_8 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_9 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 3 3
        conv_10 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_11 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)
        # sequential, output:
        self.features = sequential(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9,
                                   conv_10, conv_11)
        # classifier
        self.classifier = sequential(nn.Linear(mid_c * 3 * 3, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_256(nn.Module):
    def __init__(self, in_c=3, mid_c=128, ks=3, stride=1, padding=1, norm='batch', act='mish'):
        super(Discriminator_VGG_256, self).__init__()
        ks2 = 2 * ks
        padding2 = padding * 2
        # output: N, C, 128, 128
        conv_0 = ConvBlock(in_c=in_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_1 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 64 64
        conv_2 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_3 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 32 32
        conv_4 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_5 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 16 16
        conv_6 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_7 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 8 8
        conv_8 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_9 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)

        # output: mid_c 4 4
        conv_10 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv_11 = ConvBlock(in_c=mid_c, out_c=mid_c, ks=ks2, stride=2 * stride, padding=padding2, act=act, norm=norm)
        # sequential, output:
        self.features = sequential(conv_0, conv_1, conv_2, conv_3, conv_4, conv_5, conv_6, conv_7, conv_8, conv_9,
                                   conv_10, conv_11)
        # classifier
        self.classifier = sequential(nn.Linear(mid_c * 4 * 4, 128), nn.Mish(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
