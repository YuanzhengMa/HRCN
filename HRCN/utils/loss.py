# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/24 17:43
# @File-name       : loss.py
# @Description     :
import torch.nn as nn
import torch
from torch.nn import MSELoss, L1Loss
from models.networks import Regularization


class LapSNR(nn.Module):
    '''
    LapSNR Carbonnier loss
    '''
    def __init__(self):
        super(LapSNR, self).__init__()
        self.eps = 1e-4

    def forward(self, X, Y):
        diff = 1 / X.size()[0] * torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class GANLoss(nn.Module):
    '''
    Generative Loss
    '''
    def __init__(self, gan_type, real_label=1.0, fake_label=0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label = real_label
        self.fake_label = fake_label

        if self.gan_type == 'l1gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'l2gan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label)
        else:
            return torch.empty_like(input).fill_(self.fake_label)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

def init_loss(net, opt):

    #init MSE lapSNR and GAN Loss
    mse_loss = MSELoss()
    l1_loss = LapSNR()#L1Loss()
    gan_loss = GANLoss(gan_type='l1gan')

    # regularization factor
    if opt['base_setting']['scale'] == 2:
        w = opt['net_setting']["loss_weight_2x"][-1]
    elif opt['base_setting']['scale'] == 3:
        w = opt['net_setting']["loss_weight_3x"][-1]
    elif opt['base_setting']['scale'] == 4:
        w = opt['net_setting']["loss_weight_4x"][-1]  # [G, mse, perceptual, lap]
    else:
        w = 0
    reg_loss = Regularization(net=net, weight_decay=w, p=opt["net_setting"]["regularization_p"])
    if opt["base_setting"]["cuda"] and torch.cuda.is_available():
        mse_loss = mse_loss.cuda()
        l1_loss = l1_loss.cuda()
        gan_loss = gan_loss.cuda()
        reg_loss = reg_loss.cuda()

    return gan_loss, mse_loss, l1_loss, reg_loss