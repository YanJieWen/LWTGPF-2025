'''
@File: trainpipline.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''
import sys
import os
from tqdm import tqdm
import pandas as pd

import torch
import numpy as np

from .trainers import Vittrainerfp16
from ..utils.evaluators import extract_multipart_vit_features
from ..utils.clustering import dbscan_clustering,cam_label_split,get_centers
from .. utils.prepare_data import get_train_loader
from ..utils.checkpoint_io import save_checkpoint


def train_pipeline(start_ep,num_epochs,model,memory,trainer,evaluator,lr_scheduler,optimizer,
                   cluster_loader,test_loader,dataset,scheduler_type='warmup',log_cfg=None,
                   num_parts=5,sp_cfg=None,base_cfg=None,cluster_cfg=None,test_cfg=None,data_cfg=None):
    print(f'{"="*30}Train pipeline{"="*30}')
    pbar = tqdm(range(start_ep,num_epochs),desc=f'{"="*30}Train pipeline{"="*30}',file=sys.stdout)
    for epoch in pbar:
        pbar.desc = f'{"="*30}EPOCH num {int(epoch+1)}{"="*30}'
        pbar.desc = f'{"=" * 30}Extract features{"=" * 30}'
        #get features from train domain-->提取所有数据
        features,part_feats,_ = extract_multipart_vit_features(model,cluster_loader,num_parts)
        #NXD
        features = torch.cat([features[f].unsqueeze(0) for f,_,_ in sorted(dataset.train)],dim=0)
        #[NXD]X5
        part_feats = [torch.cat([pf[f].unsqueeze(0) for f,_,_ in sorted(dataset.train)],dim=0) for pf in part_feats]
        #clustring for pseido labels->N
        cluster_labels = dbscan_clustering(cluster_cfg, features)
        # Camera proxy generation
        pbar.desc = f'{"="*30}cam-split with global features{"="*30}'
        all_img_cams = np.array([c for _, _, c in sorted(dataset.train)])
        #这一行代码将聚类按照相机的索引进行连续编号，同一个相机被分到同一个聚类。即同一类特征又属于统一个相机被分为一个标签
        # 例如聚类结果[0,0,1,2,0]&相机[1,3,0,2,3]->[0,1,2,3,1](感觉可以用scatter实现)
        #TODO: cluster vs cluster&cam-aware
        proxy_labels = cam_label_split(cluster_labels, all_img_cams)
        #TODO:save epochs
        if (epoch+1)%10==0:
            cluster_save = {'cluster_labels':cluster_labels,'proxy_labels':proxy_labels,'filename':[],
                            'pid':[],'camid':[]}
            for img_path,pid,camid in sorted(dataset.train):
                cluster_save['filename'].append(img_path)
                cluster_save['pid'].append(pid)
                cluster_save['camid'].append(camid)
            df = pd.DataFrame(cluster_save)
            df.to_csv(os.path.join(log_cfg['ckpt_save_dir'],f'{int(epoch)}.csv'))


        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_proxies = len(set(proxy_labels)) - (1 if -1 in proxy_labels else 0)
        num_outliers = len(np.where(proxy_labels == -1)[0])
        pbar.desc = (f'{"=" * 30}Global feature clusters:{int(num_clusters)}\n'
                     f'Generated proxies:{int(num_proxies)}\n'
                     f'Outliers:{int(num_outliers)}{"=" * 30}')
        #add pseudo labels into training set
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), gcl, pl) in enumerate(zip(sorted(dataset.train), cluster_labels, proxy_labels)):
            if gcl != -1 and pl != -1:
                pseudo_labeled_dataset.append((fname, gcl, cid, i, pl))
        #Cluster-proxy mappings
        proxy_labels = torch.from_numpy(proxy_labels).long() #N
        cluster_labels = torch.from_numpy(cluster_labels).long() #N
        cluster2proxy = {}
        proxy2cluster = {}
        cam2proxy = {}
        for p in range(0, int(proxy_labels.max() + 1)):
            #{0:0,1:0,2:1,3:2}-->每个代理属于的簇
            proxy2cluster[p] = torch.unique(cluster_labels[proxy_labels == p])#one2one
        for c in range(0, int(cluster_labels.max() + 1)):
            #{0:[0,1],1:[2],2:[3]}->每个簇包含多少个代理,融入了相机感知，因此标签即使属于同一个簇由于相机不一样可能被划分到不同代理中
            cluster2proxy[c] = torch.unique(proxy_labels[cluster_labels == c])#one2many
        for cc in range(0, int(all_img_cams.max() + 1)):
            #{0:[2],1:[0],2:[3],3:[1]}-->相机与代理之间的关系
            cam2proxy[cc] = torch.unique(proxy_labels[all_img_cams == cc])#one2many
            cam2proxy[cc] = cam2proxy[cc][cam2proxy[cc] != -1] # remove outliers

        # Set memory attributes
        memory.all_proxy_labels = proxy_labels
        memory.proxy2cluster = proxy2cluster
        memory.cluster2proxy = cluster2proxy

        # Stack into a single memory
        #获取代理的特征-->[global(m,d),part(m,d)],m为基于代理的聚类-->求代理的平均特征
        proxy_memory = [get_centers(features.numpy(), proxy_labels.numpy()).cuda()] + \
                       [get_centers(f.numpy(), proxy_labels.numpy()).cuda() for f in part_feats]
        memory.proxy_memory = torch.stack(proxy_memory, dim=0) #[n_part+1,n_proxy,d]

        # camera-proxy mapping
        memory.unique_cams = torch.unique(torch.from_numpy(all_img_cams))
        memory.cam2proxy = cam2proxy
        #开始训练
        # Train one epoch
        _base_config = base_cfg.copy()
        _base_config.pop('epochs')
        _base_config.pop('fp16')
        train_loader = get_train_loader(sp_cfg, dataset, data_cfg['height'],data_cfg['width'],**_base_config,
                                        trainset=pseudo_labeled_dataset)
        # Train one epoch
        curr_lr = lr_scheduler._get_lr(epoch + 1)[0] if scheduler_type == 'cosine' else lr_scheduler.get_lr()[
            0]
        pbar.desc=f'{"="*30}Current Lr: {curr_lr}{"="*30}'
        train_loader.new_epoch()
        trainer.train(epoch + 1, train_loader, optimizer, print_freq=log_cfg['print_freq'], train_iters=len(train_loader),
                      fp16=base_cfg['fp16'])

        if scheduler_type == 'cosine':
            lr_scheduler.step(epoch + 1)
        else:
            lr_scheduler.step()
        #save checkpoints
        cmc,mAP = None,None
        if (epoch + 1) % log_cfg['save_interval'] == 0:
            save_checkpoint(model, optimizer, lr_scheduler, log_cfg['ckpt_save_dir'], epoch + 1)
        #evaluation
        if (epoch+1)%test_cfg['eval_step']==0 or epoch == base_cfg['epochs'] - 1:
            cmc,mAP = evaluator.evaluate_vit(test_loader,dataset.query,dataset.gallery,cmc_flag=True,
                                            rerank=False)
        torch.cuda.empty_cache()
        pbar.desc = '=> CUDA cache is released.'
        print('')
    return cmc,mAP




