# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 15:57
# @File-name       : HR_dataset.py
# @Description     :
import os.path
from datasets.base_dataset import BaseDataset, get_transform, get_crop_params, make_dataset_k_fold
from PIL import Image
import numpy as np
import torch
import skimage.util.noise as noise
import random


def saltpepper_noise(img, prob):
    output = np.zeros_like(img)
    for idx in range(img.shape[0]):
        for idy in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[idx][idy][:] = 0
            elif rdn > 1 - prob:
                output[idx][idy][:] = 1
            else:
                output[idx][idy][:] = img[idx][idy][:]

    return output
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

class HRDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # obtain path of HR images
        self.paths_HR_train, self.paths_HR_validation = make_dataset_k_fold(opt, os.path.join(opt['datasets']['train']['dir_HR']))
        self.opt = opt
    def __getitem__(self, index):
        # dataset for train
        path_HR = eval("self.paths_HR_" + self.opt["base_setting"]["phase"] + "[index]")
        img_HR = Image.open(path_HR)
        # if img_HR.layers == 1:
        if img_HR.mode != 'RGB':
            img_HR = img_HR.convert('RGB')
        # apply the transform to img_HR
        transform_params = get_crop_params(self.opt, img_HR.size)
        img_transform = get_transform(self.opt, params=transform_params, grayscale=self.opt['datasets']['train']['grayscale'], convert=True)

        img_HR = img_transform(img_HR)
        # DownSampling
        # img_LR = Image.fromarray(np.asarray(img_HR)[0::self.opt['base_setting']['scale'], 0::self.opt['base_setting']['scale']])
        img_LR = img_HR[:, 0::self.opt['base_setting']['scale'], 0::self.opt['base_setting']['scale']]

        # noise = np.random.normal(0, 5e-2, img_LR.shape)
        # img_LR += torch.from_numpy(noise).cuda().type(torch.FloatTensor)

        # img_LR = torch.from_numpy(img_LR.numpy()).cuda().type(torch.FloatTensor)
        # img_LR = torch.from_numpy(saltpepper_noise(img_LR, 0.005)).cuda().type(torch.FloatTensor)
        # reset self.opt['datasets']['train']['crop_HR']
        return {'img_HR': img_HR, 'img_LR':img_LR, 'path_HR':path_HR, 'path_LR':path_HR}


    def __len__(self):
        if self.opt["base_setting"]["phase"] == "train":
            self.paths_HR = self.paths_HR_train
        else:
            self.paths_HR = self.paths_HR_validation
        return len(self.paths_HR)

