# -*- coding: UTF-8 -*-
# @Start-time      : 2021/6/21 10:14
# @File-name       : __init__.py.py
# @Description     :
import importlib
import torch.utils.data
from datasets.HR_dataset import HRDataset


def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()

    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        if opt["base_setting"]["phase"] == "discriminator_train" or opt["base_setting"]["phase"] == "test":
            dataset_class = find_dataset_using_name("LR_dataset")
        else:
            dataset_class = find_dataset_using_name(opt['datasets']['train']['dataset_mode'])

        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        if self.opt["base_setting"]["phase"] == 'train':
            self.dataloader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=opt['datasets']['train']['batch_size'],
                shuffle=True,
                num_workers=int(opt["base_setting"]["num_threads"])
            )
        elif self.opt["base_setting"]["phase"] == 'validation' or opt["base_setting"]["phase"] == "discriminator_train":
            self.dataloader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=opt['datasets']['train']['batch_size'],
                shuffle=True,
                num_workers=int(opt["base_setting"]["num_threads"])
            )
        elif self.opt["base_setting"]["phase"] == 'test':
            self.dataloader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=1,
                shuffle=True,
                num_workers=int(opt["base_setting"]["num_threads"])
            )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def find_dataset_using_name(dataset_name):
    dataset_filename = "datasets." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_','')
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


