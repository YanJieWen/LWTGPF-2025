'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''



import importlib
import os
import os.path as osp

from .mb import MultiPartMemory

__all__ = (
    'MultiPartMemory',
)


model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(x))[0] for x in os.listdir(model_folder) if x.endswith('.py')]
_model_modules = [importlib.import_module(f'Csu.models.{x}') for x in model_filenames]

def create(name,*args,**kwargs):
    assert len(_model_modules)!=0,'None model found'
    model = None
    for m in _model_modules:
        model = getattr(m,name,None)
        if model is not None:
            break
    if model is None:
        raise ValueError(f'{name} is not found.')
    model = model(*args,**kwargs)
    return model