# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 22:54
# @File-name       : utils.py
# @Description     :
import os
from datetime import datetime
import logging
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
def get_timestamp():
    return datetime.now().strftime("%y_%m_%d-%H_%M_%S")


# make directions for many paths
def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths)
    else:
        for path in paths:
            os.makedirs(path)

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    # write mode, to file on disk
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        # to steam standout in python console
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

#output format for options
def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

#seed
def seed(opt, init_seed=123, manual=False):
    if manual:
        nut = init_seed
    else:
        nut = np.random.randint(0, 1e8)
    if opt["cuda"] and torch.cuda.is_available():
        # random seed for GPU
        torch.cuda.manual_seed_all(nut)
    else:
        # random seed for CPU
        torch.manual_seed(nut)


def calculate_mean_and_std(data_path, has_sub_dir=True):
    image_filenames = list()
    if has_sub_dir:
        for sub_dir in os.listdir(data_path):
            file_path = os.path.join(data_path, sub_dir)
            temp = sorted([os.path.join(file_path, x) for x in os.listdir(file_path) if
                 any(x.endswith(extension) for extension in [".png", ".jpg", "jpeg", ".bmp"])])
            image_filenames.extend(temp)
    else:
        temp = sorted([os.path.join(data_path, x) for x in os.listdir(data_path) if
             any(x.endswith(extension) for extension in [".png", ".jpg", "jpeg", ".bmp"])])
        image_filenames.extend(temp)

    dlen = len(image_filenames)
    index_ary = range(0, dlen)  # np.random.randint(0, dlen, 100)
    m0, m1, m2, v0, v1, v2 = 0, 0, 0, 0, 0, 0
    temp = cv2.imread(image_filenames[0])
    if len(temp.shape) == 3:
        for index in index_ary:
            temp = cv2.imread(image_filenames[index])
            m0, m1, m2 = temp[:, :, 0].mean() / 255, temp[:, :, 1].mean() / 255, temp[:, :, 2].mean() / 255
            v0, v1, v2 = temp[:, :, 0].std() / 255, temp[:, :, 1].std() / 255, temp[:, :, 2].std() / 255
            print("RGB images, and the mean and variation are:[{:.4f},{:.4f},{:.4f}],[{:.4f},{:.4f},{:.4f}]".format(m0, m1, m2, v0, v1, v2))
            return m0, m1, m2, v0, v1, v2
    else:
        for index in index_ary:
            temp = cv2.imread(image_filenames[index])
            m0 = temp[:, :, 0].mean() / 255
            v0 = temp[:, :, 0].std() / 255
            print("Grayscale images, and the mean and variation are:[{:.4f},{:.4f}]".format(m0, v0))
            return m0, v0
def write_loss(opt, loss_list, txt_name="loss_list"):
    txt_name = txt_name+"_{}.txt".format(get_timestamp())
    with open(os.path.join(opt['experiment']['Loss_curve_path'], txt_name), 'w') as f:
        for p in loss_list:
            f.write(str(p))
            f.write("\n")

def imshowpair(target, input, forged, maxPlotPairs=3, enlargeScale=4, denormParams=[0.5,0.5,0.5,0.5,0.5,0.5], save_img=False, save_path=r'.\results', img_name="img", draw_gap=1, name=["Target image", "Input image", "Bicubic image", "Forged image"]):
    m0, m1, m2, s0, s1, s2 = denormParams
    if target.size()[1] == 1:
        if torch.is_tensor(target):
            target = target.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            target = np.asarray([target[:,:,:,0]*s0]).transpose(1,2,3,0) + np.asarray([np.ones(target.shape)[:,:,:,0]*m0]).transpose(1,2,3,0)
        if torch.is_tensor(input):
            input = input.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            input = np.asarray([input[:,:,:,0]*s0]).transpose(1,2,3,0) + np.asarray([np.ones(input.shape)[:,:,:,0]*m0]).transpose(1,2,3,0)
        if torch.is_tensor(forged):
            forged = forged.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            forged = np.asarray([forged[:,:,:,0]*s0]).transpose(1,2,3,0) + np.asarray([np.ones(forged.shape)[:,:,:,0]*m0]).transpose(1,2,3,0)
        # init figure
        fig = plt.figure(1)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
        intervel = min(target.shape[0], maxPlotPairs)
        for idx in range(0, intervel, draw_gap):
            # target
            img = np.squeeze(target[idx])
            img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax  = fig.add_subplot(intervel, 4, 4 * idx + 1, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[0])
            ax.imshow(img,cmap='gray')
            plt.imsave(os.path.join(save_path, img_name +"_"+name[0]+"_"+ get_timestamp() + ".jpg"), img, cmap=cm.gray)
            # input
            img = np.squeeze(input[idx])
            img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax = fig.add_subplot(intervel, 4, 4 * idx + 2, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[1])
            ax.imshow(img,cmap='gray')
            plt.imsave(os.path.join(save_path, img_name + "_"+name[1]+"_" + get_timestamp() + ".jpg"), img, cmap=cm.gray)
            # bicubic
            img = cv2.resize(img, (0, 0), fx = enlargeScale, fy=enlargeScale, interpolation=cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)
            ax = fig.add_subplot(intervel, 4, 4 * idx + 3, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[2])
            ax.imshow(img,cmap='gray')
            plt.imsave(os.path.join(save_path, img_name + "_"+name[2]+"_" + get_timestamp() + ".jpg"), img, cmap=cm.gray)
            # forged
            img = np.squeeze(forged[idx])
            # img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax  = fig.add_subplot(intervel, 4, 4 * idx + 4, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[3])
            ax.imshow(img,cmap='gray')
            plt.imsave(os.path.join(save_path, img_name + "_"+name[3]+"_" + get_timestamp() + ".jpg"), img, cmap=cm.gray)
    else: # == 3
        if torch.is_tensor(target):
            target = target.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            target = np.asarray([target[:,:,:,0]*s0, target[:,:,:,1]*s1, target[:,:,:,2]*s2]).transpose(1,2,3,0) + np.asarray([np.ones(target.shape)[:,:,:,0]*m0, np.ones(target.shape)[:,:,:,1]*m1, np.ones(target.shape)[:,:,:,2]*m2]).transpose(1,2,3,0)
        if torch.is_tensor(input):
            input = input.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            input = np.asarray([input[:,:,:,0]*s0, input[:,:,:,1]*s1, input[:,:,:,2]*s2]).transpose(1,2,3,0) + np.asarray([np.ones(input.shape)[:,:,:,0]*m0, np.ones(input.shape)[:,:,:,1]*m1, np.ones(input.shape)[:,:,:,2]*m2]).transpose(1,2,3,0)
        if torch.is_tensor(forged):
            forged = forged.cuda().data.cpu().numpy().transpose((0, 2, 3, 1))# N, W, H, C
            forged = np.asarray([forged[:,:,:,0]*s0, forged[:,:,:,1]*s1, forged[:,:,:,2]*s2]).transpose(1,2,3,0) + np.asarray([np.ones(forged.shape)[:,:,:,0]*m0, np.ones(forged.shape)[:,:,:,1]*m1, np.ones(forged.shape)[:,:,:,2]*m2]).transpose(1,2,3,0)
        # init figure
        fig = plt.figure(1)
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
        intervel = min(target.shape[0], maxPlotPairs)
        for idx in range(0, intervel, draw_gap):
            # target
            img = np.squeeze(target[idx])
            img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax  = fig.add_subplot(intervel, 4, 4 * idx + 1, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[0])
            ax.imshow(img)
            plt.imsave(os.path.join(save_path, img_name +"_"+name[0]+"_"+ get_timestamp() + ".jpg"), img)
            # input
            img = np.squeeze(input[idx])
            img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax = fig.add_subplot(intervel, 4, 4 * idx + 2, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[1])
            ax.imshow(img)
            plt.imsave(os.path.join(save_path, img_name + "_"+name[1]+"_" + get_timestamp() + ".jpg"), img)
            # bicubic
            img = cv2.resize(img, (0, 0), fx = enlargeScale, fy=enlargeScale, interpolation=cv2.INTER_CUBIC)
            img = np.clip(img, 0, 1)
            ax = fig.add_subplot(intervel, 4, 4 * idx + 3, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[2])
            ax.imshow(img)
            plt.imsave(os.path.join(save_path, img_name + "_"+name[2]+"_" + get_timestamp() + ".jpg"), img)
            # forged
            img = np.squeeze(forged[idx])
            # img = (img - img.min()*np.ones(img.shape)) / (img.max() - img.min())
            img = np.clip(img, 0, 1)
            ax  = fig.add_subplot(intervel, 4, 4 * idx + 4, xticks=[], yticks=[])
            if idx == 0:
                ax.set_title(name[3])
            ax.imshow(img)
            plt.imsave(os.path.join(save_path, img_name + "_"+name[3]+"_" + get_timestamp() + ".jpg"), img)
    plt.draw()
    plt.pause(3)
    plt.close(fig)
    if save_img:
        fig.savefig(os.path.join(save_path,img_name+get_timestamp()+".jpg"), dpi=300)


def save_model(epoch, model, only_save_checkpoint, root_path, model_name):
    if only_save_checkpoint:
        torch.save(model.state_dict(),os.path.join(root_path, model_name))
    else:
        state = {"epoch": epoch, "model": model, "checkPoints": model.state_dict()}
        torch.save(state, f=os.path.join(root_path, "plus_"+model_name))

def plot_loss_curve(loss_list):
    plt.plot(range(0, len(loss_list)), loss_list)
    plt.ylabel('Loss (Validation)'), plt.xlabel('Epochs (Validation)'), plt.title(
        'Epochs vs. Loss  (Validation)')
    plt.draw()
    plt.pause(3)
    plt.close('all')