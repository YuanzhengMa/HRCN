# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 22:13
# @File-name       : options.py
# @Description     :
import os
import json
import os.path
from utils.utils import mkdirs

def parse(opt_path, is_train=True):
    # remove comments starting with //
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str)

    if is_train:
        # create path under root path using the subpath exp[eriments//opt['name']
        experiments_root = os.path.join(opt['base_setting']['root_path'], 'experiments', opt['base_setting']['net_name'])
        opt['experiment']['models_path'] = os.path.join(experiments_root, 'models')
        opt['experiment']['Loss_curve_path'] = os.path.join(experiments_root, 'Loss_curve')
        opt['experiment']['log_path'] = os.path.join(experiments_root, 'logs')
        opt['experiment']['val_results_path'] = os.path.join(experiments_root, 'val_results')
        opt["net_setting"]["min_loss"] = 100
        opt["datasets"]["k"] = 1
        opt["experiment"]["test_results_path"] = os.path.join(experiments_root, 'test_results')

    else:  # test
        experiments_root = os.path.join(opt['base_setting']['root_path'], 'experiments', opt['base_setting']['net_name'])
        opt['experiment']['models_path'] = os.path.join(experiments_root, 'models')
        opt['experiment']['log_path'] = os.path.join(experiments_root, 'logs')
        opt["experiment"]["test_results_path"] = os.path.join(experiments_root, 'test_results')

    # create file if conditions is satisfied and os.path.isfile() is True
    for key, path in opt['experiment'].items():
        print(key + ":" + str(path))
        if not os.path.exists(path):
            mkdirs(path)


    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt