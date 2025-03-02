'''
@File: prepare_model.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

from Csu import models
import copy


def get_model(cfg):
    try:
        model = models.create(cfg['arch'],**cfg)
        # print(model)
    except: #if arch is not needed to pass
        _cfg = cfg.copy()
        _cfg.pop('arch')
        model = models.create(cfg['arch'],**_cfg)

    model.cuda()
    return model


