# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/24 18:14
# @File-name       : fit.py
# @Description     :
import random
import time
from torch.nn.functional import interpolate
import torch.cuda
from torch.autograd import Variable
from models.networks import set_requires_grad
from utils.loss import init_loss
from models.feature_extraction import VGG_featureX
from torch import optim
from math import ceil
from utils.utils import imshowpair, save_model
import numpy as np

class BP2():
    def __init__(self, opt, encoder, decoder, discriminator_encoder, discriminator_decoder, logger):
        super(BP2, self).__init__()
        # loss function = G + MSE + Perceptual + l1
        self.gan_lossE, self.mse_lossE, self.l1_lossE, self.reg_lossE = init_loss(encoder, opt)
        self.gan_lossDeE, self.mse_lossDeE, self.l1_lossDeE, self.reg_lossDeE = init_loss(decoder, opt)
        # load features extraction:=VGG19 for perceptual Loss
        self.Fea_net = VGG_featureX(fea_c=49, norm=opt["net_setting"]["feature_net_use_norm"], use_input_norm=opt["net_setting"]["feature_net_input_norm"])
        self.Fea_net = self.Fea_net.cuda() if opt["base_setting"]["cuda"] else self.Fea_net
        self.opt = opt
        self.encoder = encoder.cuda() if opt["base_setting"]["cuda"] else self.encoder
        self.decoder = decoder.cuda() if opt["base_setting"]["cuda"] else self.decoder
        self.discriminator_encoder = discriminator_encoder.cuda() if opt["base_setting"]["cuda"] else self.discriminator_encoder
        self.discriminator_decoder = discriminator_decoder.cuda() if opt["base_setting"]["cuda"] else self.discriminator_decoder
        if opt['base_setting']['scale'] == 2:
            self.w = opt['net_setting']["loss_weight_2x"]
        elif opt['base_setting']['scale'] == 3:
            self.w = opt['net_setting']["loss_weight_3x"]
        elif opt['base_setting']['scale'] == 4:
            self.w = opt['net_setting']["loss_weight_4x"] #[G, mse, perceptual, l1]
        else:
            w = 0
        # init optimizer
        self.DeE_optimizer = optim.Adam(self.decoder.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        self.E_optimizer = optim.Adam(self.encoder.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        self.Dis_optimizer_encoder = optim.Adam(self.discriminator_encoder.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        self.Dis_optimizer_decoder = optim.Adam(self.discriminator_decoder.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        # init scheduler
        self.DeE_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.DeE_optimizer, T_0=3, T_mult=5)
        self.Dis_scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.Dis_optimizer_encoder, T_0=3, T_mult=5)
        self.E_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.E_optimizer, T_0=3, T_mult=5)
        self.Dis_scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.Dis_optimizer_decoder, T_0=3, T_mult=5)
        #
        self.logger = logger
    def fit(self, epoch, data_loader, data_loader_discriminator, phase='train'):
        if phase == 'train':
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        since = time.time()
        running_lossE = 0
        running_ganE = 0
        running_l1E = 0
        running_gan_fitDeE = 0
        running_lossDeE = 0
        running_ganDeE = 0
        running_mseDeE = 0
        running_perceptualDeE = 0
        running_l1DeE = 0
        running_discriminatorDeE = 0
        running_kl  = 0
        for iteration, batch in enumerate(data_loader):
            input, target = Variable(batch['img_LR']),Variable(batch['img_HR'])
            if self.opt['base_setting']['cuda'] and torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()
            # random choose from data_loader_discriminator for training encoder
            x = np.random.randint(0, len(data_loader_discriminator)-1)
            for iterationD, batchD in enumerate(data_loader_discriminator):
                if iterationD == x:
                    encoded_target = Variable(batchD['img_LR']).cuda() if self.opt['base_setting']['cuda'] and torch.cuda.is_available() else Variable(batchD['img_LR'])
                    break
            # generating bicubic
            # targetLR = interpolate(input=target, scale_factor=1/self.opt['base_setting']['scale'], mode='bicubic', align_corners=True, recompute_scale_factor=False)

            # # -------1-1. training encoder
            set_requires_grad(nets=self.decoder, requires_grad=False)
            set_requires_grad(nets=self.encoder, requires_grad=True)
            set_requires_grad(nets=self.discriminator_decoder, requires_grad=False)
            set_requires_grad(nets=self.discriminator_encoder, requires_grad=False)
            self.E_optimizer.zero_grad()
            forgedLR = self.encoder(input)
            #  calculate Loss
            # gan loss 4 Encodder
            g1 = self.gan_lossE(self.discriminator_encoder(forgedLR) - torch.mean(self.discriminator_encoder(input)), True)
            g2 = self.gan_lossE(self.discriminator_encoder(input) - torch.mean(self.discriminator_encoder(forgedLR)), False)
            gan_fitE = self.w[0] / 5 * (g1 + g2) / 2
            l1_fitE = self.w[1] * self.l1_lossE(forgedLR, input)
            # calculate kl_loss
            kl_loss = 1e-5 * torch.nn.functional.kl_div(forgedLR.softmax(dim=-1).log(), encoded_target.softmax(dim=-1), reduction='sum')
            # perceptual loss
            # forged_features = self.Fea_net(forgedLR).detach()
            # target_features = self.Fea_net(input).detach()
            # perceptual_fitE = self.w[2] * sum([self.mse_lossE(forged_features[idx], target_features[idx]) for idx in range(len(forged_features))])
            reg_fitE = self.reg_lossE(self.encoder)

            lossE = gan_fitE + l1_fitE + kl_loss + reg_fitE
            lossE.backward()
            self.E_optimizer.step()
            self.E_scheduler.step()
            running_lossE += lossE
            running_ganE += gan_fitE
            running_l1E += l1_fitE
            running_kl += kl_loss
            # -------1-2. training discriminator_encoder
            set_requires_grad(nets=self.decoder, requires_grad=False)
            set_requires_grad(nets=self.encoder, requires_grad=False)
            set_requires_grad(nets=self.discriminator_decoder, requires_grad=False)
            set_requires_grad(nets=self.discriminator_encoder, requires_grad=True)
            #  calculate Loss
            self.Dis_optimizer_encoder.zero_grad()
            # gan loss
            g1 = self.gan_lossE(self.discriminator_encoder(forgedLR.detach()) - torch.mean(self.discriminator_encoder(input)), False)
            g2 = self.gan_lossE(self.discriminator_encoder(input) - torch.mean(self.discriminator_encoder(forgedLR.detach())), True)
            gan_fitE = (g1 + g2) / 2
            torch.autograd.backward(gan_fitE)
            self.Dis_optimizer_encoder.step()
            self.Dis_scheduler_encoder.step()

            # -------2-1. deducing model from forged to target
            set_requires_grad(nets=self.encoder, requires_grad=False)
            set_requires_grad(nets=self.decoder, requires_grad=True)
            set_requires_grad(nets=self.discriminator_decoder, requires_grad=False)
            set_requires_grad(nets=self.discriminator_encoder, requires_grad=False)
            set_requires_grad(nets=self.Fea_net, requires_grad=False)
            self.DeE_optimizer.zero_grad()
            forged = self.decoder(input*0.5+forgedLR.detach()*0.5)
            # calculate Loss
            mse_fitDeE = self.w[1] * self.mse_lossDeE(forged, target)
            # perceptual loss
            forged_features = self.Fea_net(forged).detach()
            target_features = self.Fea_net(target).detach()
            perceptual_fitDeE = self.w[2] * sum([self.mse_lossDeE(forged_features[idx], target_features[idx]) for idx in range(len(forged_features))])
            l1_fitDeE = self.w[3] * self.l1_lossDeE(forged, target)
            # gan loss
            g1 = self.gan_lossDeE(self.discriminator_decoder(forged) - torch.mean(self.discriminator_decoder(target)), True)
            g2 = self.gan_lossDeE(self.discriminator_decoder(target) - torch.mean(self.discriminator_decoder(forged)), False)
            gan_fitDeE = self.w[0] * (g1 + g2) / 2
            reg_lossDeE = self.reg_lossDeE(self.decoder)
            lossDeE = gan_fitDeE + mse_fitDeE + perceptual_fitDeE + l1_fitDeE + reg_lossDeE
            torch.autograd.set_detect_anomaly(True)
            lossDeE.backward()
            self.DeE_optimizer.step()
            self.DeE_scheduler.step()
            running_lossDeE += lossDeE
            running_ganDeE += gan_fitDeE
            running_mseDeE += mse_fitDeE
            running_perceptualDeE += perceptual_fitDeE
            running_l1DeE += l1_fitDeE

            # -------2-2. train discriminator for decoder
            set_requires_grad(nets=self.encoder, requires_grad=False)
            set_requires_grad(nets=self.decoder, requires_grad=False)
            set_requires_grad(nets=self.discriminator_decoder, requires_grad=True)
            set_requires_grad(nets=self.discriminator_encoder, requires_grad=False)
            self.Dis_optimizer_decoder.zero_grad()
            # gan loss
            g1 = self.gan_lossDeE(self.discriminator_decoder(forged.detach()) - torch.mean(self.discriminator_decoder(target)), False)
            g2 = self.gan_lossDeE(self.discriminator_decoder(target) - torch.mean(self.discriminator_decoder(forged.detach())), True)
            gan_fitDeE1 = (g1 + g2) / 2
            running_discriminatorDeE += gan_fitDeE1
            torch.autograd.backward(gan_fitDeE1)
            self.Dis_optimizer_decoder.step()
            self.Dis_scheduler_decoder.step()

            if iteration % len(batch) == 0:
                self.logger.info(phase + " Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, (iteration +1), ceil(len(data_loader.dataset) / data_loader.batch_size), lossDeE.item()))
            if iteration >= self.opt['base_setting']['max_iteration_per_epoch']:
                break
        running_lossE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_ganE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_l1E /= self.opt['base_setting']['max_iteration_per_epoch']
        running_gan_fitDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_kl /= self.opt['base_setting']['max_iteration_per_epoch']
        running_lossDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_ganDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_mseDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_perceptualDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_l1DeE /= self.opt['base_setting']['max_iteration_per_epoch']
        running_discriminatorDeE /= self.opt['base_setting']['max_iteration_per_epoch']
        end_since = time.time()
        delta_time = end_since - since
        self.logger.info(phase + " Encoder:: Total_loss: {:.4f}, gan_lossE: {:.4f}, gan_lossDeE: {:.4f}, l1_loss: {:.4f}, kl_loss: {:.4f}".format(running_lossE, running_ganE, running_lossDeE, running_l1E, running_kl))
        self.logger.info(phase + " Decoder:: Total_loss: {:.4f}, gan_loss: {:.4f}, mse_loss: {:.4f}, perceptual_loss: {:.4f}, l1_loss: {:.4f}".format(running_lossDeE, running_ganDeE, running_mseDeE, running_perceptualDeE, running_l1DeE, running_discriminatorDeE))
        self.logger.info(phase + " Time spent in {:.0f} epoch is {:0.0f}minutes:{:0.0f}seconds.".format(epoch, round(delta_time / 60), delta_time % 60))
        self.logger.info(phase + " Epoch in pretraining {} complete: Average Loss: {:.4f}".format(epoch, running_lossDeE))

        if phase == 'validation':
            # imshowpair(target, input, forgedLR, maxPlotPairs=4, enlargeScale=self.opt["base_setting"]["scale"], denormParams=[self.opt['m0'], self.opt['m1'], self.opt['m2'], self.opt['s0'], self.opt['s1'], self.opt['s2']], save_path=self.opt["experiment"]["val_results_path"], name=["Target", "InputLR", "BicubicLR", "ForgedLR"])
            imshowpair(target, input, forged, maxPlotPairs=4, enlargeScale=self.opt["base_setting"]["scale"], denormParams=[self.opt['m0'], self.opt['m1'], self.opt['m2'], self.opt['s0'], self.opt['s1'], self.opt['s2']], save_path=self.opt["experiment"]["val_results_path"])
            save_model(epoch=epoch, model=self.encoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="trained_model_DeE_"+str(epoch)+".pth")
            save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="trained_model_E_" + str(epoch) + ".pth")
            if running_lossDeE < self.opt["net_setting"]["min_loss"]:
                self.opt["net_setting"]["min_loss"] = running_lossDeE
                save_model(epoch=epoch, model=self.encoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="best_model_E.pth")
                save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="best_model_DeE.pth")

        return running_lossE, running_lossDeE


class BP1():
    def __init__(self, opt, decoder, discriminator, logger):
        super(BP1, self).__init__()
        # loss function = G + MSE + Perceptual + l1
        self.gan_loss1, self.mse_loss1, self.l1_loss1, self.reg_loss1 = init_loss(decoder, opt)
        # load features extraction:=VGG19 for perceptual Loss
        self.Fea_net = VGG_featureX(fea_c=49, norm=opt["net_setting"]["feature_net_use_norm"], use_input_norm=opt["net_setting"]["feature_net_input_norm"])
        self.Fea_net = self.Fea_net.cuda() if opt["base_setting"]["cuda"] else self.Fea_net
        self.opt = opt
        self.decoder = decoder.cuda() if opt["base_setting"]["cuda"] else self.decoder
        self.discriminator = discriminator.cuda() if opt["base_setting"]["cuda"] else self.discriminator
        if opt['base_setting']['scale'] == 2:
            self.w = opt['net_setting']["loss_weight_2x"]
        elif opt['base_setting']['scale'] == 3:
            self.w = opt['net_setting']["loss_weight_3x"]
        elif opt['base_setting']['scale'] == 4:
            self.w = opt['net_setting']["loss_weight_4x"] # [GAN_loss, mse, perceptual, l1]
        else:
            w = 0
        # init optimizer
        self.DeE_optimizer = optim.Adam(self.decoder.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        self.Dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=opt['base_setting']['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
        # init scheduler
        self.DeE_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.DeE_optimizer, T_0=3, T_mult=5)
        self.Dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.Dis_optimizer, T_0=3, T_mult=5)
        # init logger
        self.logger = logger
    def fit(self, epoch, data_loader, phase='train'):
        if phase == 'train':
            self.decoder.train()
        else:
            self.decoder.eval()
        since = time.time()
        running_loss1 = 0
        running_mse = 0
        running_l1 = 0
        for iteration, batch in enumerate(data_loader):
            input, target = Variable(batch['img_LR']),Variable(batch['img_HR'])
            if self.opt['base_setting']['cuda'] and torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()
            # generating bicubic'
            # -------1. deducing model from input to forged
            set_requires_grad(nets=self.decoder, requires_grad=True)
            self.DeE_optimizer.zero_grad()
            forged = self.decoder(input)
            #  calculate Loss
            mse_fit1 = self.w[1] * self.mse_loss1(forged, target)
            l1_fit1 = self.w[3] * self.l1_loss1(forged, target)
            reg_loss1 = self.reg_loss1(self.decoder)
            loss1 = mse_fit1 + l1_fit1 + reg_loss1
            loss1.backward()
            self.DeE_optimizer.step()
            self.DeE_scheduler.step()
            running_loss1 += loss1
            running_mse += mse_fit1
            running_l1 += l1_fit1
            if iteration % len(batch) == 0:
                self.logger.info(phase + " Epoch[{}]({}/{}): Total Loss of Decoder: {:.6f}".format(epoch, (iteration + 1), ceil(len(data_loader.dataset) / data_loader.batch_size), loss1.item()))
            if iteration >= self.opt['base_setting']['max_iteration_per_epoch']:
                break
        running_loss1 /= self.opt['base_setting']['max_iteration_per_epoch']
        running_l1 /= self.opt['base_setting']['max_iteration_per_epoch']
        running_mse /= self.opt['base_setting']['max_iteration_per_epoch']
        end_since = time.time()
        delta_time = end_since - since

        self.logger.info(phase + " Average Loss of Decoder:: {:.4f}, mse_loss: {:.4f}, l1_loss: {:.4f}".format(running_loss1, running_mse, running_l1))
        self.logger.info(phase + " Time spent in {:.0f} epoch:: {:0.0f}minutes:{:0.0f}seconds.".format(epoch, round(delta_time / 60), delta_time % 60))

        if phase == 'validation':
            imshowpair(target, input, forged, maxPlotPairs=4, enlargeScale=self.opt["base_setting"]["scale"], denormParams=[self.opt['m0'], self.opt['m1'], self.opt['m2'], self.opt['s0'], self.opt['s1'], self.opt['s2']], save_path=self.opt["experiment"]["val_results_path"])
            save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="trained_model_DeE_" + str(epoch) + ".pth")
            if running_loss1 < self.opt["net_setting"]["min_loss"]:
                self.opt["net_setting"]["min_loss"] = running_loss1
                save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="best_model_DeE.pth")

        return running_loss1

    def fit_gan(self, epoch, data_loader, phase='train'):

        if phase == 'train':
            self.decoder.train()
        else:
            self.decoder.eval()
        since = time.time()
        running_loss1 = 0
        running_mse = 0
        running_l1 = 0
        running_perceptual = 0
        running_gan = 0
        running_regularization = 0
        running_discriminator = 0
        for iteration, batch in enumerate(data_loader):

            input, target = Variable(batch['img_LR']), Variable(batch['img_HR'])
            if self.opt['base_setting']['cuda'] and torch.cuda.is_available():
                input, target = input.cuda(), target.cuda()
            # generating bicubic'
            # -------1. deducing model from input to forged
            # ------- training decoder
            self.DeE_optimizer.zero_grad()
            set_requires_grad(nets=self.decoder, requires_grad=True)
            set_requires_grad(nets=self.discriminator, requires_grad=False)
            set_requires_grad(nets=self.Fea_net, requires_grad=False)
            forged = self.decoder(input)
            #  calculate Loss
            mse_fit1 = self.w[1] * self.mse_loss1(forged, target)
            l1_fit1 = self.w[3] * self.l1_loss1(forged, target)
            reg_loss1 = self.reg_loss1(self.decoder)
            # perceptual loss
            forged_features = self.Fea_net(forged).detach()
            target_features = self.Fea_net(target).detach()
            perceptual_fit1 = self.w[2] * sum([self.mse_loss1(forged_features[idx], target_features[idx]) for idx in range(len(forged_features))])
            # gan loss
            g1 = self.gan_loss1(self.discriminator(forged) - torch.mean(self.discriminator(target)), True)
            g2 = self.gan_loss1(self.discriminator(target) - torch.mean(self.discriminator(forged)), False)
            gan_fit1 = self.w[0] * (g1 + g2)/2
            # sum loss
            loss1 = gan_fit1 + mse_fit1 + perceptual_fit1 + l1_fit1 + reg_loss1
            # train decoder
            torch.autograd.set_detect_anomaly(True)
            loss1.backward()#autograd.backward(loss1, grad_tensors=torch.ones_like(loss1), retain_graph=False,create_graph=False)
            self.DeE_optimizer.step()
            self.DeE_scheduler.step()
            running_loss1 += loss1
            running_gan += gan_fit1
            running_mse += mse_fit1
            running_perceptual += perceptual_fit1
            running_l1 += l1_fit1
            running_regularization += reg_loss1

            # ------- training discriminator
            if iteration % self.opt['base_setting']['frequency_train_discriminator']:
                self.Dis_optimizer.zero_grad()
                set_requires_grad(nets=self.decoder, requires_grad=False)
                set_requires_grad(nets=self.discriminator, requires_grad=True)
                set_requires_grad(nets=self.Fea_net, requires_grad=False)
                # gan loss
                # detach() should be used on "forged" to avoid loss backward to decoder
                d1 = self.gan_loss1(self.discriminator(forged.detach()) - torch.mean(self.discriminator(target)), False)
                d2 = self.gan_loss1(self.discriminator(target) - torch.mean(self.discriminator(forged.detach())), True)
                gan_fit2 = (d1 + d2) / 2
                running_discriminator += gan_fit2
                gan_fit2.backward()
                self.Dis_optimizer.step()
                self.Dis_scheduler.step()

            if iteration % len(batch) == 0:
                self.logger.info(phase + " Epoch[{}]({}/{}): Total Loss of Decoder: {:.6f}".format(epoch, (iteration + 1), ceil(len(data_loader.dataset) / data_loader.batch_size), loss1.item()))
            if iteration >= self.opt['base_setting']['max_iteration_per_epoch']:
                break
        running_loss1 /= self.opt['base_setting']['max_iteration_per_epoch']
        running_gan /= self.opt['base_setting']['max_iteration_per_epoch']
        running_mse /= self.opt['base_setting']['max_iteration_per_epoch']
        running_perceptual /= self.opt['base_setting']['max_iteration_per_epoch']
        running_l1 /= self.opt['base_setting']['max_iteration_per_epoch']
        running_discriminator /= self.opt['base_setting']['max_iteration_per_epoch']
        end_since = time.time()
        delta_time = end_since - since

        self.logger.info(phase + " Average Loss of Decoder:: total_loss: {:.4f}, gan_loss: {:.4f}, mse_loss: {:.4f}, perceptual_loss: {:.4f}, l1_loss: {:.4f}, regularization_loss: {:.4f}".format(running_loss1, running_gan, running_mse, running_perceptual, running_l1, running_regularization))
        self.logger.info(phase + " Average Loss of Discriminator:: total_loss: {:.4f},".format(running_discriminator))
        self.logger.info(phase + " Time spent in {:.0f} epoch:: {:0.0f}minutes:{:0.0f}seconds.".format(epoch, round(delta_time / 60), delta_time % 60))

        if phase == 'validation':
            imshowpair(target, input, forged, maxPlotPairs=4, enlargeScale=self.opt["base_setting"]["scale"], denormParams=[self.opt['m0'], self.opt['m1'], self.opt['m2'], self.opt['s0'], self.opt['s1'], self.opt['s2']], save_path=self.opt["experiment"]["val_results_path"])
            save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="trained_model_DeE_" + str(epoch) + ".pth")
            if running_loss1 < self.opt["net_setting"]["min_loss"]:
                self.opt["net_setting"]["min_loss"] = running_loss1
                save_model(epoch=epoch, model=self.decoder, only_save_checkpoint=True, root_path=self.opt["experiment"]["models_path"], model_name="best_model_DeE.pth")

        return running_loss1

class test_model():
    def __init__(self, opt, decoder, logger):
        super(test_model, self).__init__()
        self.opt = opt
        self.decoder = decoder.cuda() if opt["base_setting"]["cuda"] else self.decoder
        # init logger
        self.logger = logger
    def forge(self, data_loader):
        self.decoder.eval()
        since = time.time()

        for iteration, batch in enumerate(data_loader):
            input = Variable(batch['img_LR'])
            if self.opt['base_setting']['cuda'] and torch.cuda.is_available():
                input = input.cuda()
            # generating bicubic'
            # -------1. deducing model from input to forged
            forged = self.decoder(input)
            imshowpair(input, input, forged, maxPlotPairs=4, enlargeScale=self.opt["base_setting"]["scale"],
                       denormParams=[self.opt['m0'], self.opt['m1'], self.opt['m2'], self.opt['s0'], self.opt['s1'],
                                     self.opt['s2']], save_path=self.opt["experiment"]["test_results_path"])
        end_since = time.time()
        delta_time = end_since - since
        self.logger.info(" Time spent :: {:0.0f}minutes:{:0.0f}seconds.".format(round(delta_time / 60), delta_time % 60))

        return 0