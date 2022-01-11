# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 15:58
# @File-name       : HRLR_dataset.py
# @Description     :
import os.path
from datasets.base_dataset import BaseDataset, get_transform, make_dataset, get_crop_params
from PIL import Image
import random


class LRDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # obtain path of HR images
        if opt["base_setting"]["phase"] == "discriminator_train":
            self.paths_LR = make_dataset(os.path.join(opt['datasets']['train']['dir_realLR']))
        elif opt["base_setting"]["phase"] == "test":
            self.paths_LR = make_dataset(os.path.join(opt['datasets']['train']['dir_testLR']))
        self.opt = opt
    def __getitem__(self, index):
        # dataset for train
        path = self.paths_LR[index]
        img_LR = Image.open(path)
        # if img_HR.layers == 1:
        if img_LR.mode != 'RGB':
            img_LR = img_LR.convert('RGB')
        # LR images do no need of  crop and flip
        temp = self.opt['datasets']['train']['crop_HR']
        if not self.opt["base_setting"]["phase"] == "test":
            self.opt['datasets']['train']['crop_HR'] = self.opt['datasets']['train']['crop_HR'] // self.opt['base_setting']['scale']
            transform_params = get_crop_params(self.opt, img_LR.size)
        else:
            transform_params = {'crop_pos': (0, 0), 'flip': 0}
        img_transform = get_transform(self.opt, params=transform_params, grayscale=self.opt['datasets']['train']['grayscale'], convert=True)
        img_LR = img_transform(img_LR)
        self.opt['datasets']['train']['crop_HR'] = temp
        return {'img_LR':img_LR, 'path_LR':path, }


    def __len__(self):

        return len(self.paths_LR)