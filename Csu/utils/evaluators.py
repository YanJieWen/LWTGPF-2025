'''
@File: evaluators.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

from collections import OrderedDict
import os.path as osp

import torch
import csv
import tqdm
from Csu import to_numpy,to_torch
from .rerank import re_ranking
from ..metrics import mean_ap,cmc

def save_benchmark(cfg, mAP, cmc, task_name, dataset_name,time_cost):
    if not osp.exists(cfg['benchmark_path']):
        with open(cfg['benchmark_path'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Training time', 'Dataset', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10'])
    with open(cfg['benchmark_path'], 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([task_name, '{}'.format(time_cost), dataset_name, '{:.1%}'.format(mAP), '{:.1%}'.format(cmc[0]),
                        '{:.1%}'.format(cmc[4]), '{:.1%}'.format(cmc[9])])
    print('=> Benchmark is updated.')


def extract_vit_features(model,data_loader):
    '''

    :param model: nn.Module
    :param data_loader: iter
    :return: 每个图片的编码信息以及pid为标签
    '''
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    with torch.no_grad():
        for i,(imgs,fnames,pids,cams,_) in enumerate(tqdm.tqdm(data_loader)):
            imgs = to_torch(imgs).cuda()
            cams = to_torch(cams).cuda()
            outputs = model(imgs,cam_label=cams)
            if isinstance(outputs,dict):
                outputs = outputs['global']
            outputs = outputs.data.cpu()
            for fname,output,pid in zip(fnames,outputs,pids):
                features[fname] = output
                labels[fname] = pid
    return features,labels

def pairwise_distance(features,query=None,gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2#对高维特征维度进行汇总
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())#计算代理之间的成对关系
        return dist_m
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m,n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(distmat,query,gallery,query_ids=None,gallery_ids=None,
                 query_cams=None,gallery_cams=None,cmc_topk=(1, 5, 10),cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else: #必须提供查询和画廊
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    if (not cmc_flag):
        return mAP
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))
    return cmc_scores['market1501'], mAP

class Evaluator(object):
    def __init__(self,cfg,model):
        super().__init__()
        self.model = model
        self.cfg = cfg
    def evaluate_vit(self,data_loader,query,gallery,cmc_flag=False,rerank=False,is_concat=False):
        features,_ = extract_vit_features(self.model,data_loader)
        distmat,_,_ = pairwise_distance(features,query,gallery)
        results = evaluate_all(distmat,query=query,gallery=gallery,cmc_flag=cmc_flag)
        if (not rerank):
            return results
        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query)
        distmat_gg = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


def extract_multipart_vit_features(model,data_loader,num_parts):
    model.eval()
    global_feats = OrderedDict()
    labels = OrderedDict()
    part_feats = [OrderedDict() for _ in range(num_parts)]
    with torch.no_grad():
        for i,(imgs,fnames,pids,cams,_) in enumerate(tqdm.tqdm(data_loader)):
            imgs = to_torch(imgs).cuda()
            cams = to_torch(cams).cuda()
            out_dict = model(imgs, cam_label=cams) #receive img and cams number
            for k,v in out_dict.items():
                if k=='global':
                    out_dict[k] = v.data.cpu()
                elif k=='part':
                    out_dict[k] = [x.data.cpu() for x in v]
            #obtain global
            for fname,output,pid in zip(fnames,out_dict['global'],pids):
                global_feats[fname] = output
                labels[fname] = pid
            #obtain local
            for part,pf in zip(out_dict['part'],part_feats):
                for fname, output, pid in zip(fnames, part, pids):
                    pf[fname] = output
            # OD, list, OD
    return global_feats,part_feats,labels#{fname:feature}xb,[{fname:feature}]x5