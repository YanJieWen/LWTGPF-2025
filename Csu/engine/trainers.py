'''
@File: trainers.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''


import time
from torch.cuda import amp
from ..utils import AverageMeter
from abc import ABC,abstractmethod

class BaseTrainer(ABC):
    def __init__(self,encoder,memory):
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    @abstractmethod
    def train(self,epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        pass

    @abstractmethod
    def _parse_data(self,inputs):
        pass

    @abstractmethod
    def _forward(self,inputs):
        pass

class Vittrainerfp16(BaseTrainer):
    def __init__(self,encoder,memory):
        super().__init__(encoder,memory)
    def _parse_data(self, inputs):
        imgs, _, _, cams, index_target, _ = inputs   # img, fname, pseudo_label, camid, img_index, accum_label
        return imgs.cuda(), cams.cuda(), index_target.cuda()
    def _forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
    def train(self,epoch, data_loader, optimizer, print_freq=10, train_iters=400,fp16=False):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses =AverageMeter()
        end = time.time()
        scaler = amp.GradScaler() if fp16 else None
        for i in range(train_iters): #train one epochs
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            inputs, cams, index_target = self._parse_data(inputs)
            with amp.autocast(enabled=fp16):
                f_out = self._forward(inputs, cam_label=cams) #transformer encoder
                loss_dict = self.memory(f_out, index_target, cams, epoch) #bank memory get loss,index get which samples are trained
                loss = loss_dict['loss']
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] '
                      'Time: {:.3f} ({:.3f}), '
                      'Data: {:.3f} ({:.3f}), '
                      'Loss: {:.3f} ({:.3f}), '
                      '{}'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              ', '.join(['{}: {:.3f}'.format(k, v) for k, v in loss_dict.items()])))
