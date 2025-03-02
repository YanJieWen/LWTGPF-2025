'''
@File: preprocessor.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''


from torch.utils.data import DataLoader, Dataset
import os
import os.path as osp
import torchvision.transforms as T
import numpy as np
import random
import math
from PIL import Image
import torchvision.transforms as T

class Preprocessor(Dataset):
    def __init__(self,dataset,root=None,transform=None,height=384,width=128):
        super().__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        fname,pid,camid = self.dataset[idx]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root,fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, idx


class CameraAwarePreprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(CameraAwarePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pseudo_label, camid, img_index, accum_label = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pseudo_label, camid, img_index, accum_label