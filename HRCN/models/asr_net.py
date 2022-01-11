# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 10:16
# @File-name       : asr_net.py
# @Description     :
import torch.nn as nn
from .networks import ConvBlock, sequential, upconv, ShortcutBlock, RungeKuttaResBlock
from .base_model import BaseModel
import torch
class ASRNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, mid_c=128, numb=10, scale=4, norm='batch', act='relu', nz = 8):
        super(ASRNet, self).__init__()
        self.nz = nz
        fea_conv = ConvBlock(in_c=in_c, out_c=mid_c, act=None, norm=None)
        rk_res_block = [RungeKuttaResBlock(in_c=mid_c, out_c=mid_c, norm=norm, act=act) for _ in range(numb)]
        lr_conv = ConvBlock(in_c=mid_c, out_c=mid_c, act=act, norm=norm)
        hr_conv0 = ConvBlock(in_c=mid_c, out_c=mid_c, act=None, norm=norm)
        hr_conv1 = ConvBlock(in_c=mid_c, out_c=out_c, act=None, norm=None)
        if scale % 2 == 0:
            upsampler = [upconv(in_c=mid_c, out_c=mid_c, scale=2, act=act, norm=norm) for _ in range(scale // 2)]
        else:
            upsampler0 = [upconv(in_c=mid_c, out_c=mid_c, scale=2, act=act, norm=norm) for _ in range(scale // 2 - 1)]
            upsampler1 = [upconv(in_c=mid_c, out_c=mid_c, scale=3, act=act, norm=norm)]
            upsampler = sequential(upsampler0, upsampler1)
        self.model = sequential(fea_conv, ShortcutBlock(sequential(*rk_res_block, lr_conv)), *upsampler, hr_conv0, hr_conv1)

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)#size(z) ==> N C=8, W, H
        else:
            x_with_z = x  # no z
        x = self.model(x_with_z)
        return x
