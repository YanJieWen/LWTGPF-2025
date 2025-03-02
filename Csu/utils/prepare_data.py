'''
@File: prepare_data.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import os.path as osp

from Csu import datasets
from .data import Preprocessor


import torchvision.transforms as T
from torch.utils.data import DataLoader
from .data.sampler import (ClassUniformlySampler, RandomMultipleGallerySampler,
                           ClusterProxyBalancedSampler)
from .data import IterLoader
from .data.preprocessor import CameraAwarePreprocessor


def get_data(name,root_dir):
    root = osp.join(root_dir,name)
    print(f'root path= {root}')
    dataset = datasets.create(name,root,verbose=True)
    return dataset

def get_test_loader(cfg,dataset,batch_size,workers,testset=None):
    height = cfg['height']
    width = cfg['width']
    normalizer = T.Normalize(mean=cfg['mean'],std=cfg['std'])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader


def get_train_loader(cfg, dataset, height, width, batch_size, num_worker,
                    num_instances, iters, trainset=None):
    # Preprocessing
    normalizer = T.Normalize(mean=cfg['mean'],
                             std=cfg['std'])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(p=0.5)
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sample_type = cfg['type']
    # Choose sampler type
    # class_position [1: cluster_label, 4: proxy_label]
    if sample_type == 'proxy_balance':
        sampler = ClassUniformlySampler(train_set, class_position=4, k=num_instances)
    elif sample_type == 'cluster_balance':
        sampler = ClassUniformlySampler(train_set, class_position=1, k=num_instances)
    elif sample_type == 'cam_cluster_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=1, num_instances=num_instances)
    elif sample_type == 'cam_proxy_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=4, num_instances=num_instances)
    elif sample_type == 'cluster_proxy_balance':
        sampler = ClusterProxyBalancedSampler(train_set, k=num_instances)
    else:
        raise ValueError('Invalid sampler type name!')

    # Create dataloader
    train_loader = IterLoader(
                DataLoader(CameraAwarePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=num_worker, sampler=sampler,
                            shuffle=False, pin_memory=True, drop_last=True), length=iters)
    return train_loader
