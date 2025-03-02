'''
@File: checkpoint_io.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import os
import os.path as osp
import torch


def save_checkpoint(model, optimizer, scheduler, ckpt_save_dir, epoch):
    if not osp.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir, exist_ok=True)
    save_files = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':scheduler.state_dict(),
        'epoch':epoch
    }
    torch.save(save_files,osp.join(ckpt_save_dir,f'{int(epoch)}.pt'))
    # torch.save(model.state_dict(), osp.join(ckpt_save_dir, 'weight_{}.pth'.format(epoch)))
    # torch.save(optimizer.state_dict(), osp.join(ckpt_save_dir, 'optim_{}.pth'.format(epoch)))
    # torch.save(scheduler.state_dict(), osp.join(ckpt_save_dir, 'scheduler_{}.pth'.format(epoch)))


def load_checkpoint(model, optimizer, scheduler, ckpt_load_dir, name):
    resume = os.path.join(ckpt_load_dir,name)
    if not osp.isfile(resume):
        raise TypeError(f'{resume} is not a file')
    ckpt = torch.load(resume,map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch = ckpt['epoch'] + 1

    # weight = torch.load(osp.join(ckpt_load_dir, 'weight_{}.pth').format(ckpt_load_ep))
    # opt_params = torch.load(osp.join(ckpt_load_dir, 'optim_{}.pth'.format(ckpt_load_ep)))
    # sch_params = torch.load(osp.join(ckpt_load_dir, 'scheduler_{}.pth'.format(ckpt_load_ep)))
    #
    # model.load_state_dict(weight)
    # optimizer.load_state_dict(opt_params)
    # scheduler.load_state_dict(sch_params)

    return model, optimizer, scheduler,start_epoch