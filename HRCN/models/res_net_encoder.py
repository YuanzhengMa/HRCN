# -*- coding: UTF-8 -*-
# @Start-time      : 2021/7/13 21:49
# @File-name       : res_net_encoder.py
# @Description     :
import torch.nn as nn
import functools
import torch
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)



def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, mid_c=64, numb=4,
                 norm='batch', nl_layer=functools.partial(nn.ReLU, inplace=True)):
        super(E_ResNet, self).__init__()

        max_ndf = 4
        conv_layers = [nn.Conv2d(in_c, mid_c, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, numb):
            input_ndf = mid_c * min(max_ndf, n)
            output_ndf = mid_c * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, out_c)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, out_c)])
        self.conv = nn.Sequential(*conv_layers)

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to('cuda')

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        mu = self.fc(conv_flat)
        logvar = self.fcVar(conv_flat)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar