# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/30 20:11
# @File-name       : test.py
# @Description     :
import argparse
import logging
import options.options
from datasets import create_dataset
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.utils import setup_logger, dict2str, seed, calculate_mean_and_std
from models.networks import log_net_params_num
from fit.fit import test_model

if __name__ == '__main__':
    # --------0. load options from json------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=r'options\params_train.json')
    opt = options.options.parse(parser.parse_args().opt, is_train=False)
    # convert miss key to none dict
    opt = options.options.dict_to_nonedict(opt)
    # --------2. set up logger------------#
    setup_logger(logger_name=opt['base_setting']['net_name'], phase='test', root=opt['experiment']['log_path'],  screen=True)
    logger = logging.getLogger(name=opt['base_setting']['net_name'])
    # writing options information
    logger.info(dict2str(opt))
    # --------3. seed------------#
    seed(opt, init_seed=100, manual=False)
    # --------4. calculate mean and std of the dataset---------#
    opt['m0'], opt['m1'], opt['m2'], opt['s0'], opt['s1'], opt['s2'] = calculate_mean_and_std(data_path=opt["datasets"]["train"]["dir_HR"], has_sub_dir=True)  # 0.471, 0.448, 0.408, 0.234, 0.239, 0.242  #
    opt["base_setting"]["phase"] = "discriminator_train"
    train_set_discriminator_train = create_dataset(opt)
    logger.info("Total num of images used for training is : {:,d}".format(len(train_set_discriminator_train.dataset)))
    # --------5. resume training, load model---------#
    # load encoder network
    if opt["base_setting"]["net_name"].lower() == 'asrnet':
        from models.asr_net import ASRNet
        from models.res_net import ResNet
        DeE_net = ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])
        #DeE_net = ASRNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])
    elif opt["base_setting"]["net_name"].lower() == 'srresnet':
        from models.res_net import ResNet
        from models.asr_net import ASRNet
        # DeE_net = ASRNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"],
        #                  mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"],
        #                  scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"],
        #                  act=opt["net_setting"]["act"], nz=0)
        DeE_net = ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])
    else:
        logger.info("Please input network class among {'asrnet', 'resnet', 'dbpn'}")
        exit(0)
    # --------6. instantiation---------#
    # print the sum num of network paramater
    log_net_params_num(net=DeE_net, logger=logger)

    # --------7. load dataset---------#
    opt["base_setting"]["phase"] = "test"
    test_set = create_dataset(opt)

    # --------8. load weight--------- #
    DeE_net.load_state_dict(state_dict=torch.load(os.path.join(opt["experiment"]["models_path"], "best_model_DeE.pth")))
    test_model = test_model(opt=opt, decoder=DeE_net, logger=logger)
    rlt = test_model.forge(data_loader=test_set.dataloader)
