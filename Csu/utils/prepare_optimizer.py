'''
@File: prepare_optimizer.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch

def get_optimizer(cfg,model):
    params = []
    for name,value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['lr0']
        weight_decay = cfg['weight_decay']
        if 'bias' in name:
            lr = cfg['lr0']*cfg['bias_lr_factor']
            weight_decay = cfg['weight_decay_bias']
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    opti_name = cfg['name']
    if opti_name == 'SGD':
        optimizer = getattr(torch.optim,opti_name)(params,momentum=cfg['momentum'])
    elif opti_name == 'AdamW':
        optimizer = getattr(torch.optim,opti_name)(params,lr=cfg['lr0'],weight_decay=cfg['weight_decay'])
    elif opti_name in ['Adamax','ASGD','Rprop','RMSprop']:
        optimizer = getattr(torch.optim,opti_name)(params)
    else:
        raise ValueError(f'==={opti_name} is not found===')
    return optimizer