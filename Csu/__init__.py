'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 21, 2024
@HomePage: https://github.com/YanJieWen
'''

# from .utils import yaml_load,Logger,get_data,get_model,get_test_loader,Evaluator
# from .models import MultiPartMemory
# from .metrics import *
# from .datasets import *
#
# __all__ = ('yaml_load','Logger','get_data','get_model',
#            'MultiPartMemory','get_test_loader','Evaluator'
#
#
# )

import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray