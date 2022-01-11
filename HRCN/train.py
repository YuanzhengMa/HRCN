import argparse
import logging

import torchvision.models

import options.options
from datasets import create_dataset
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.utils import setup_logger, dict2str, seed, calculate_mean_and_std, plot_loss_curve
from models.networks import log_net_params_num
from fit.fit import BP2, BP1
from utils.utils import write_loss

if __name__ == '__main__':
    # --------0. load options from json------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=r'options\params_train.json')
    opt = options.options.parse(parser.parse_args().opt, is_train=True)
    # convert miss key to none dict
    opt = options.options.dict_to_nonedict(opt)

    # --------1. set up logger------------#
    setup_logger(logger_name=opt['base_setting']['net_name'], phase='train', root=opt['experiment']['log_path'],  screen=True)
    logger = logging.getLogger(name=opt['base_setting']['net_name'])
    # writing options information
    logger.info(dict2str(opt))
    # --------2. seed------------#
    seed(opt, init_seed=100, manual=False)
    # --------3. calculate mean and std of the dataset---------#
    opt['m0'], opt['m1'], opt['m2'], opt['s0'], opt['s1'], opt['s2'] = 0.471, 0.448, 0.408, 0.234, 0.239, 0.242#calculate_mean_and_std(data_path=opt["datasets"]["train"]["dir_HR"], has_sub_dir=True)  # 0.471, 0.448, 0.408, 0.234, 0.239, 0.242  #
    opt["base_setting"]["phase"] = "discriminator_train"
    train_set_discriminator_train = create_dataset(opt)
    logger.info("Total num of images used for training is : {:,d}".format(len(train_set_discriminator_train.dataset)))
    # --------4. resume training, load model---------#
    #load encoder network
    if opt["base_setting"]["net_name"].lower() == 'asrnet':
        from models.asr_net import ASRNet
        from models.res_net import ResNet
        from models.res_net_encoder import E_ResNet

        E_net = ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=1, norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])#E_ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"],  norm=opt["net_setting"]["norm"])
        DeE_net = ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])#ASRNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])

    elif opt["base_setting"]["net_name"].lower() == 'srresnet':
        from models.res_net import ResNet
        from models.asr_net import ASRNet
        DeE_net = ASRNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"], nz=0)# ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])
        # DeE_net = ResNet(in_c=opt["net_setting"]["in_channels"], out_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"], numb=opt["net_setting"]["num_resblock"], scale=opt["base_setting"]["scale"], norm=opt["net_setting"]["norm"], act=opt["net_setting"]["act"])
    else:
        logger.info("Please input network class among {'asrnet', 'resnet', 'dbpn'}")
        exit(0)
    # load discriminator
    if opt['datasets']['train']['crop_HR'] == 128:
        from models.discriminator import Discriminator_VGG_128
        Dis_net1 = Discriminator_VGG_128(in_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"])
        if opt["base_setting"]["net_name"].lower() == 'asrnet':
            from models.discriminator import Discriminator_VGG_64
            Dis_net0 = Discriminator_VGG_64(in_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"])
    elif opt['datasets']['train']['crop_HR'] == 192:
        from models.discriminator import Discriminator_VGG_192
        Dis_net1 = Discriminator_VGG_192(in_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"])
    elif opt['datasets']['train']['crop_HR'] == 256:
        from models.discriminator import Discriminator_VGG_256
        Dis_net1 = Discriminator_VGG_256(in_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"])
        if opt["base_setting"]["net_name"].lower() == 'asrnet':
            from models.discriminator import Discriminator_VGG_128
            Dis_net0 = Discriminator_VGG_128(in_c=opt["net_setting"]["in_channels"], mid_c=opt["net_setting"]["mid_channels"])
    else:
        logger.info("Only support input size among {128, 192, 256}")
        exit(0)
    # --------5. judge whether this is a new train------------#
    if opt['base_setting']['resume_train']:
        # load best_model from experimental model path
        DeE_net.load_state_dict(torch.load(os.path.join(opt['experiment']['models_path'], "best_model_DeE.pth")))
        if opt["base_setting"]["net_name"].lower() == 'asrnet':
            E_net.load_state_dict(torch.load(os.path.join(opt['experiment']['models_path'], "best_model_E.pth")))
    # --------6. instantiation---------#
    # print the sum num of network paramater
    log_net_params_num(net=DeE_net,logger=logger)
    log_net_params_num(net=Dis_net1, logger=logger)
    if opt["base_setting"]["net_name"].lower() == 'asrnet':
        log_net_params_num(net=E_net, logger=logger)
        # log_net_params_num(net=Dis_net0, logger=logger)

    # --------7. training with k-fold--------- #
    if opt["base_setting"]["net_name"].lower() == 'asrnet':
        # BP2 instantiation, BP2 := encder-decoder structure
        bp2 = BP2(opt=opt, encoder=E_net, decoder=DeE_net, discriminator_encoder=Dis_net0, discriminator_decoder=Dis_net1, logger=logger)
        train_loss_list_encoder, train_loss_list_decoder, val_loss_list_encoder, val_loss_list_decoder = [], [], [], []
        # k-fold train for encoder-decoder
        for idx in range(opt["datasets"]["k"], opt["datasets"]["k_fold_num"]):
            # load datasets accordinf different k
            opt["datasets"]["k"] = idx
            opt["base_setting"]["phase"] = "train"
            train_set = create_dataset(opt)
            logger.info("Total num of images used for training is : {:,d}".format(len(train_set.dataset)))
            opt["base_setting"]["phase"] = "validation"
            val_set = create_dataset(opt)
            logger.info("Total num of images used for validation is : {:,d}".format(len(val_set.dataset)))
            # train
            for epoch in range(1, opt["base_setting"]["epochs"]+1):
                opt["base_setting"]["phase"] = "train"
                running_loss_encoder, running_loss_decoder =bp2.fit(epoch=epoch, data_loader=train_set.dataloader, data_loader_discriminator=train_set_discriminator_train.dataloader, phase='train')
                # train_loss_list_encoder.append(running_loss_encoder.cuda().data.cpu().numpy()), \
                train_loss_list_decoder.append(running_loss_decoder.cuda().data.cpu().numpy())
                if epoch % opt["base_setting"]["validation_frequency"] == 0:
                    opt["base_setting"]["phase"] = "validation"
                    running_loss_encoder, running_loss_decoder = bp2.fit(epoch=epoch, data_loader=train_set.dataloader, data_loader_discriminator=train_set_discriminator_train.dataloader, phase='validation')
                    # val_loss_list_encoder.append(running_loss_encoder.cuda().data.cpu().numpy()), \
                    val_loss_list_decoder.append(running_loss_decoder.cuda().data.cpu().numpy())
                    plot_loss_curve(val_loss_list_encoder), plot_loss_curve(val_loss_list_decoder)
            write_loss(opt, train_loss_list_encoder, txt_name="train_loss_list0"), write_loss(opt, train_loss_list_decoder, txt_name="train_loss_list1")
            write_loss(opt, val_loss_list_encoder, txt_name="val_loss_list0"), write_loss(opt, val_loss_list_decoder, txt_name="val_loss_list1")
    else: # SRResNet and DBPNet
        # BP1 instantiation, BP1:= single decoder structure
        bp1 = BP1(opt=opt, decoder=DeE_net, discriminator=Dis_net1, logger=logger)
        train_loss_list_decoder, val_loss_list_decoder = [], []
        # k-fold train for decoder
        for idx in range(opt["datasets"]["k"], opt["datasets"]["k_fold_num"]):
            # load datasets accordinf different k
            opt["datasets"]["k"] = idx
            opt["base_setting"]["phase"] = "train"
            train_set = create_dataset(opt)
            logger.info("Total num of images used for training is : {:,d}".format(len(train_set.dataset)))
            opt["base_setting"]["phase"] = "validation"
            val_set = create_dataset(opt)
            logger.info("Total num of images used for validation is : {:,d}".format(len(val_set.dataset)))
            for epoch in range(1, opt["base_setting"]["epochs"]+1):
                # train with train dataset
                opt["base_setting"]["phase"] = "train"
                running_loss_decoder = bp1.fit_gan(epoch=epoch, data_loader=train_set.dataloader, phase='train')
                train_loss_list_decoder.append(running_loss_decoder.cuda().data.cpu().numpy())
                if epoch % opt["base_setting"]["validation_frequency"] == 0:
                    # validating with validation dataset
                    opt["base_setting"]["phase"] = "validation"
                    running_loss_decoder = bp1.fit_gan(epoch=epoch, data_loader=val_set.dataloader, phase='validation')
                    val_loss_list_decoder.append(running_loss_decoder.cuda().data.cpu().numpy())
                    plot_loss_curve(val_loss_list_decoder)
            write_loss(opt, train_loss_list_decoder, txt_name="train_loss_list1"), write_loss(opt, val_loss_list_decoder, txt_name="val_loss_list1")
