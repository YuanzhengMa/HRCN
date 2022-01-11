# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 15:53
# @File-name       : base_dataset.py
# @Description     :
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import os
import os.path
import torch.utils.data as data
import cv2
from torchvision.transforms import InterpolationMode

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt['dataroot']

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, item):
        pass

def get_crop_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['datasets']['train']['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['datasets']['train']['size_HR']
    elif opt['datasets']['train']['preprocess'] == 'scale_width_and_crop':
        new_w = opt['datasets']['train']['size_HR']
        new_h = opt['datasets']['train']['size_HR'] * h // w
    else:
        # using scale _width_and_crop as default
        new_w = opt['datasets']['train']['size_HR']
        new_h = opt['datasets']['train']['size_HR'] * h // w
    # the start point of cropping
    x = random.randint(0, np.maximum(0, new_w - opt['datasets']['train']['crop_HR']))
    y = random.randint(0, np.maximum(0, new_h - opt['datasets']['train']['crop_HR']))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    #transform list for transforming
    transform_list = []
    preprocess = eval("opt['datasets']['train']['preprocess_"+opt["base_setting"]["phase"]+"']")
    if grayscale:
        # one channel
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preprocess:
        out_size = [opt['datasets']['train']['size_HR'], opt['datasets']['train']['size_HR']]
        transform_list.append(transforms.Resize(out_size, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt['datasets']['train']['size_HR'], method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt['datasets']['train']['crop_HR']))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['datasets']['train']['crop_HR'])))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt['datasets']['train']['no_flip']:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((opt['m0'],), (opt['s0'],))]
        else:
            transform_list += [transforms.Normalize((opt['m0'], opt['m1'], opt['m2']), (opt['s0'], opt['s1'], opt['s2']))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base=2, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    else:
        return cv2.resize(np.copy(img), (tw, th), interpolation=cv2.INTER_CUBIC)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

# judge any file is image or not by extension
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_k_fold(opt, dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, sub_dirs, _ in sorted(os.walk(dir)):
        for sub_dir in sub_dirs:
            for sub_root, _, fnames in sorted(os.walk(os.path.join(dir,sub_dir))):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(dir, sub_dir, fname)
                        images.append(path)
    # k_fold
    # sort images in order
    sorted(images)
    train_images = []
    val_images = []
    per_slice = k_fold4img(images, k=opt["datasets"]["k_fold_num"])
    # choose one slice as validation dataset
    x = opt["datasets"]["k"]
    for idx in range(opt["datasets"]["k_fold_num"]):
        exec("images_"+str(idx)+"=images[" + str(idx) + "*per_slice:per_slice*"+str(idx+1)+"]")

    for idx in range(opt["datasets"]["k_fold_num"]):
        if idx != x:
            exec("train_images.extend(images_" + str(idx)+")")
        else:
            exec("val_images.extend(images_" + str(idx) + ")")

    return train_images, val_images


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, sub_dirs, _ in sorted(os.walk(dir)):
        for sub_dir in sub_dirs:
            for sub_root, _, fnames in sorted(os.walk(os.path.join(dir,sub_dir))):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(dir, sub_dir, fname)
                        images.append(path)

    return sorted(images)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset_k_fold(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def k_fold4img(dataset, k=5):
    L = len(dataset)
    per_fold = L//k
    return per_fold