'''
@File: train.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 21, 2024
@HomePage: https://github.com/YanJieWen
'''

import argparse
import time
import os.path as osp
import sys
from datetime import timedelta
import random
import numpy as np
import torch


from Csu.utils import (yaml_load,Logger,get_data,get_model,get_test_loader,Evaluator,
                       get_schedular,get_optimizer,load_checkpoint,save_benchmark)
from Csu.models import MultiPartMemory
from Csu.engine import Vittrainerfp16,train_pipeline

def main():

    start_time =time.monotonic()
    #parser config
    parser = argparse.ArgumentParser('--ReID Network Building--')
    parser.add_argument('--config',type=str,default='./configs/lwtgpf.yaml',help='Config as yaml form')
    args = parser.parse_args()
    cfg = yaml_load(args.config)
    #init env
    if 'seed' in cfg.keys():
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    #build task folder
    log_cfg = cfg['train_setting']['log']
    task_name = time.strftime('%Y%m%d')+'_'+cfg['task_name']
    log_file_name = osp.join(log_cfg.get('log_dir','logs'),task_name+'.txt')
    ckpt_save_dir = osp.join(log_cfg.get('ckpt_save_dir','ckpt'),task_name)

    # Print settings
    sys.stdout = Logger(log_file_name)
    print("==========\n{}\n==========".format(cfg))
    print('=> Task name:', task_name)

    # Create datasets
    base_config = cfg['train_setting']['base']
    iters = base_config['iters'] if base_config['iters'] > 0 else None
    print("=> Load unlabeled dataset")
    benchmark_config = cfg['benchmark_setting']
    #train+query+gallery->save to [(path,pid,cid)]
    dataset = get_data(benchmark_config['name'],benchmark_config['root_dir'])

    #create model
    model_cfg = cfg['model_setting']
    model = get_model(model_cfg)

    #create memory bank
    mb_bank = cfg['membank_setting']
    memory = MultiPartMemory(mb_bank).cuda()

    #get dataloader
    data_cfg = cfg['input_setting']
    #train for cluster
    cluster_loader = get_test_loader(data_cfg,dataset,base_config['batch_size'],
                                     base_config['num_worker'],testset=sorted(dataset.train))

    test_loader = get_test_loader(data_cfg,dataset,base_config['batch_size'],
                                     base_config['num_worker'])
    #evaluator-->NOT IMPLMENT
    evaluator = Evaluator(cfg, model)

    #create optimizer & schedular
    opti_cfg = cfg['train_setting']['optimizer']
    optimizer = get_optimizer(opti_cfg,model)
    lr_scheduler = get_schedular(opti_cfg,optimizer)

    #ckpt setting
    if len(log_cfg['ckpt_load_dir'])==0:
        start_ep = 0
        print('=> Training from scratch')
    else:
        model, optimizer, lr_scheduler,start_ep = load_checkpoint(model, optimizer, lr_scheduler,
                                                                 log_cfg['ckpt_load_dir'],log_cfg['ckpt_name'])
        if start_ep>base_config['epochs']:
            raise ValueError('model has reached best state')
        else:
            print(f'=> Continue training from epoch={start_ep}, load checkpoint from {log_cfg["ckpt_load_dir"]}')
    trainer = Vittrainerfp16(model, memory)
    #train pipeline
    sp_cfg = cfg['sample_setting']
    test_cfg = cfg['test_setting']
    cluster_cfg = cfg['cluster_setting']
    #kernel framework
    cmc,mAP = train_pipeline(start_ep,opti_cfg['num_epochs'],model,memory,trainer,evaluator,lr_scheduler,optimizer,
                   cluster_loader,test_loader,dataset,scheduler_type=opti_cfg['scheduler_type'],log_cfg=log_cfg,
                   num_parts=model_cfg['num_parts'],sp_cfg=sp_cfg ,base_cfg=base_config,
                   cluster_cfg=cluster_cfg,test_cfg=test_cfg,data_cfg=data_cfg)
    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time - start_time)
    print('=> Task finished: {}'.format(task_name))
    print('Total running time: {}'.format(dtime))
    # Save benchmark
    #1123-->还有评估和benchmark未能实现,即利用testloader
    if log_cfg['save_benchmark']:
        if cmc is not None and mAP is not None:
            save_benchmark(log_cfg, mAP, cmc, task_name, benchmark_config['name'],dtime)
        else:
            pass










if __name__ == '__main__':
    main()