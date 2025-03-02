'''
@File: predict.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 28, 2024
@HomePage: https://github.com/YanJieWen
'''

#!/usr/bin/env Python
# coding=utf-8


import argparse
import re

import copy
import pandas as pd

import torch
import tqdm
from Csu.utils import yaml_load,get_model
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict

from Csu.utils import get_test_loader,yaml_load,get_data,get_model
from Csu import to_torch

import pandas as pd

config = './configs/occduke.yaml'
ckpt_path = './runs/occduke/ckpt/50.pt'

cfg = yaml_load(config)
data_cfg = cfg['input_setting']
benchmark_config = cfg['benchmark_setting']
base_config = cfg['train_setting']['base']
model_cfg = cfg['model_setting']

ckpt = torch.load(ckpt_path)['model']
dataset = get_data(benchmark_config['name'],benchmark_config['root_dir'])
model = get_model(model_cfg)
expct_keys,miss_keys = model.load_state_dict(ckpt,strict=False)
model.to('cuda:0')
#query&gallery
test_loader = get_test_loader(data_cfg,dataset,base_config['batch_size'],
                                     base_config['num_worker'])
#step1: run prediction features
model.eval()
features = OrderedDict()
labels = OrderedDict()
with torch.no_grad():
    for i, (imgs, fnames, pids, cams, _) in enumerate(tqdm.tqdm(test_loader)):
        imgs = to_torch(imgs).cuda()
        cams = to_torch(cams).cuda()
        outputs = model(imgs, cam_label=cams)
        if isinstance(outputs, dict):
            outputs = outputs['global']
        outputs = outputs.data.cpu()
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
#step2: calmetrics
query = dataset.query
gallery = dataset.gallery
x = torch.cat([features[f].unsqueeze(0) for f,_,_ in query],dim=0)#mxd
y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], dim=0)#nxd
m,n = x.size(0), y.size(0)
x = x.view(m, -1)
y = y.view(n, -1)
dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
dist_m.addmm_(1, -2, x, y.t())
#step3:get pariwise results
results = {'q_id':[],'q_f':[]}
for i in range(10):
    id_name = f'g{int(i)}_id'
    path_name = f'g{int(i)}_f'
    if id_name not in results.keys():
        results[id_name] = []
        results[path_name] = []
    else:
        pass
# false_pariwise = {'q_id':[],'q_f':[],'g_ids':[],'g_files':[]}
# suceess_pariwise = copy.deepcopy(false_pariwise)
query_ids = [pid for _, pid, _ in query]
gallery_ids = [pid for _, pid, _ in gallery]
query_cams = [cam for _, _, cam in query]
gallery_cams = [cam for _, _, cam in gallery]
distmat = dist_m.cpu().numpy()
m,n = distmat.shape
indices = np.argsort(distmat,axis=1)#mxn,基于画廊的样本排序索引,从小到大排序，值越小值越近
#TODO: save prediction results
_indices = indices[:,:10]
for i in tqdm.tqdm(range(m),desc='save prediction'):
    results['q_id'].append(query[i][1])
    results['q_f'].append(query[i][0])
    for k in range(len(_indices[i])):
        k_id = _indices[i][k]
        results[f'g{int(k)}_id'].append(gallery[k_id][1])
        results[f'g{int(k)}_f'].append(gallery[k_id][0])
res_df = pd.DataFrame(results)
res_df.to_csv('./res.csv')

query_ids = np.asarray(query_ids)#m
gallery_ids = np.asarray(gallery_ids)#n
query_cams = np.asarray(query_cams)#m
gallery_cams = np.asarray(gallery_cams)#n
#基于代价矩阵-->命中为同一个pid的0/1,仅为pid相同为1
matches = (gallery_ids[indices] == query_ids[:, np.newaxis])#是否为相同的id的人->[m,n]画廊中有多少个和查询中行人id匹配的
ret = np.zeros(100) #value
num_valid_queries = 0
for i in tqdm.tqdm(range(m), desc='CMC Eval.'):#遍历每一个样本
    #n->personid与cameraid有一个不满足则为True，都满足为false->0/1->1为无效，0为有效
    valid = ((gallery_ids[indices[i]] != query_ids[i]) |
             (gallery_cams[indices[i]] != query_cams[i]))#
    #如果不存在任何一个1值，全部为0则抛弃
    if not np.any(matches[i, valid]): continue
    index = np.nonzero(matches[i, valid])[0]#非0的索引-->跨摄像头相同行人ID
    for j, k in enumerate(index):
        if k - j >= 100: break
        ret[k - j] += 1
        break
    num_valid_queries += 1
res = ret.cumsum() / num_valid_queries
for k in (1,5,10):
    print('  top-{:<4}{:12.1%}'.format(k, res[k-1]))








