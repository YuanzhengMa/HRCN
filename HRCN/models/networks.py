# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 20:43
# @File-name       : networks.py
# @Description     :
import torch
import torch.nn as nn
import torchvision.models
from torch.nn import init
from collections import OrderedDict
from torch.optim import lr_scheduler

# initializing
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# Blocks be used in Network
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=(3, 3), stride=(1, 1), padding=1, act='mish', norm='batch'):
        super(ConvBlock, self).__init__()
        # define convlution
        self.conv = torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride,
                                    padding=padding, bias=True)
        # choose norm type
        if norm is not None and norm != 'None':
            self.norm = norm
        else:
            self.norm = None
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(out_c)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(out_c)
        else:
            self.bn = None
        # choose activation type among relu...
        if act:
            if act.lower() == 'relu':
                self.act = torch.nn.ReLU(True)
            elif act.lower() == 'prelu':
                self.act = torch.nn.PReLU()
            elif act.lower() == 'lrelu':
                self.act = torch.nn.LeakyReLU(0.2, True)
            elif act.lower() == 'elu':
                self.act = torch.nn.ELU()
            elif act.lower() == 'mish':
                self.act = torch.nn.Mish()
        else:
            self.act = None

    def forward(self, x):
        # y = [Conv-Norm-Act]x
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.act is not None:
            return self.act(out)
        else:
            return out


class RungeKuttaResBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=(3, 3), stride=(1, 1), padding=1, norm=None, act='mish'):
        super().__init__()
        for idx in range(0, 4, 1):
            exec("self.res" + str(idx) + " = ConvBlock(in_c=" + str(in_c) + ",out_c=" + str(
                in_c) + ",ks=" + str(ks) + ",stride=" + str(stride) + ",padding=" + str(padding) + ",norm='" + str(
                norm) + "',act='" + str(act) + "')")
        exec("self.attention" + " = ConvBlock(in_c=" + str(2*in_c) + ",out_c=" + str(
            out_c) + ",ks=" + str(ks) + ",stride=" + str(stride) + ",padding=" + str(padding) + ",norm='" + str(
            norm) + "',act='" + str(act) + "')")

    def forward(self, x):
        weight = [1, 1 / 2, 1 / 2, 1]
        y = x
        res = x
        for idx in range(0, 4, 1):
            res = eval("self.res" + str(idx) + "(weight[idx]*res) + x")
            y = y + (1 / 6) * res * (1 / weight[idx])
        x = eval("self.attention" + "(torch.cat([2 * x - y, y],dim=1))")
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=(3, 3), stride=(1, 1), padding=1, norm='batch', act='relu'):
        super().__init__()
        conv0 = ConvBlock(in_c=in_c, out_c=out_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        conv1 = ConvBlock(in_c=in_c, out_c=out_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)
        self.res = sequential(conv0, conv1)
    def forward(self, x):
        res = self.res(x)
        x = x + res
        return x


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            submodule: nn.Module
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def upconv(in_c, out_c, scale=2, ks=(3, 3), stride=(1, 1), padding=1, norm=None, act='relu', mode='nearest'):
    upsample = nn.Upsample(scale_factor=scale, mode=mode)
    conv = ConvBlock(in_c=in_c, out_c=out_c, ks=ks, stride=stride, padding=padding, act=act, norm=norm)

    return sequential(upsample, conv)


class ShortcutBlock(nn.Module):
    """Elementwise sum the output of a submodule to its input"""

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        temp_str = 'Identify + \n|'
        mod_str = self.sub.__repr__().replace('\n', '\n|')
        temp_str = temp_str + mod_str
        return temp_str

class Regularization(nn.Module):
    def __init__(self, net, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0 or p == 0:
            exit(0)
        else:
            self.model = net
            self.weight_decay = weight_decay
            self.p = p
            self.weight_list = self.get_weight_list

    def forward(self, net):
        self.weight_list = self.get_weight_list(net=net)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss


    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for k, w in weight_list:
            reg_L = torch.norm(w, p=p)
            reg_loss += reg_L
        # return value need plus the weight decay param
        return weight_decay * reg_loss


    def get_weight_list(self, net):
        weight_list = []
        for k, w in net.named_parameters():
            if 'weight' in k:
                weight = (k, w)
                weight_list.append(weight)
            return weight_list


def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def log_net_params_num(net, logger):
    sum_params = 0
    for param in net.parameters():
        sum_params += param.numel()
    logger.info(f"Total number of parameters in {net._get_name()} is: {sum_params}")
