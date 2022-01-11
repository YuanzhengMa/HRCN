# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/23 10:58
# @File-name       : res_net.py
# @Description     :
from models.networks import ResBlock, ConvBlock, upconv, sequential, ShortcutBlock
import torch.nn as nn
from math import ceil, log

class ResNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, mid_c=128, numb=10, scale=4, norm='batch', act='relu'):
        super(ResNet, self).__init__()
        fea_conv = ConvBlock(in_c=in_c, out_c=mid_c, act=None, norm=None)
        res_block = [ResBlock(in_c=mid_c, out_c=mid_c, norm=norm, act=act) for _ in range(numb)]
        lr_conv = ConvBlock(in_c=mid_c, out_c=mid_c, act=None, norm=norm)
        hr_conv0 = ConvBlock(in_c=mid_c, out_c=mid_c, act=None, norm=norm)
        hr_conv1 = ConvBlock(in_c=mid_c, out_c=out_c, act=None, norm=None)
        if scale == 0:
            print("Please input the right scale factor!")
            exit(0)
        elif scale == 1:
            upsampler = [upconv(in_c=mid_c, out_c=mid_c, scale=1, act=act, norm=norm)]
        elif scale % 2 == 0:
            upsampler = [upconv(in_c=mid_c, out_c=mid_c, scale=2, act=act, norm=norm) for _ in range(scale // 2)]
        else:
            upsampler0 = [upconv(in_c=mid_c, out_c=mid_c, scale=2, act=act, norm=norm) for _ in range(ceil(scale / 2) - 1)]
            upsampler1 = [upconv(in_c=mid_c, out_c=mid_c, scale=3, act=act, norm=norm)]
            upsampler = sequential(upsampler0, upsampler1)
        self.model = sequential(fea_conv, ShortcutBlock(sequential(*res_block, lr_conv)), *upsampler, hr_conv0, hr_conv1)

    def forward(self, x):
        x = self.model(x)
        return x