'''
@File: prepare_schdular.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

from .schedular import CosineLRScheduler,WarmupMultiStepLR

def get_schedular(cfg,optimizer):
    scheduler_type = cfg['scheduler_type']
    if scheduler_type=='cosine':
        num_epochs = cfg['num_epochs']
        lr_min = 0.002*cfg['lr0']
        warmup_lr_init =0.01*cfg['lr0']
        warmup_t = cfg['warmup_epochs']
        noise_range = None
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul=1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=0.67,
            noise_std=1.,
            noise_seed=42,
        )
    elif scheduler_type == 'warmup':
        lr_scheduler = WarmupMultiStepLR(optimizer, cfg['milestones'], gamma=cfg['gamma'],
                                         warmup_factor=cfg['warmup_factor'],
                                         warmup_iters=cfg['warmup_epochs'])
    else:
        raise ValueError(f'Invalid scheduler type {scheduler_type}!')
    return lr_scheduler