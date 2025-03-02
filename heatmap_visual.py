'''
@File: heatmap_visual.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 05, 2024
@HomePage: https://github.com/YanJieWen
'''

import re

import torch
from Csu.utils import yaml_load,get_model
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image


from Csu.tools import SwinGradCam,AttentionRollout

# ckpt_path = './runs/tmgf-occduke/ckpt/50.pt'
ckpt_path = './runs/occduke/ckpt/50.pt'
yaml_path = './configs/occduke.yaml'
# yaml_path = './configs/tmgfocc.yaml'
img_path = './benchmarks/Occ_Duke/query/2772_c4_f0172169.jpg'
H,W = 384, 128
visual_type = 'grad'

ckpt = torch.load(ckpt_path)['model']

cfg = yaml_load(yaml_path)
model_cfg = cfg['model_setting']
model = get_model(model_cfg)
model.to('cuda:0')
model.eval()
expct_keys,miss_keys = model.load_state_dict(ckpt,strict=False)
pattern = re.compile(r'([-\d]+)_c(\d)')
_, camid = map(int, pattern.search(img_path).groups())
img = cv2.imread(img_path)
_img = cv2.resize(img,(W,H))
img_inp = Image.fromarray(img)
transform = transforms.Compose([
        transforms.Resize((H,W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5 , 0.5, 0.5]),
    ])
input_tensor = transform(img_inp).unsqueeze(0).cuda()
#base.stages.2.blocks.16.attn.w_msa.attn_drop
#b1.blocks.0.attn.w_msa.attn_drop
#b1.blocks.1,b2.blocks.1,base.stages.2.blocks.16,base.stages.2.blocks.17
# model.base.blocks[10]
# model.base.stages[2].blocks[17]
# print(model.base.blocks[11])
# [print(name) for name,m in model.named_modules()]
# x = torch.randn((1,3,384,128)).cuda()
# handle = []
# def save_activation(m,input,output):
#     handle.append(output.cpu().detach())
# model.base.blocks[11].register_forward_hook(hook=save_activation)
# out = model(x,cam_label=torch.as_tensor([3]))
# print(handle)


_vtype = {
    'attn': AttentionRollout(model,'base.stages.2.blocks.16.attn.w_msa.attn_drop',
                        'max',0.9),
    'grad': SwinGradCam(model,[model.base.stages[2].blocks[17]],'GradCAM',
                             (H//16,W//16),camid,model_type='swin'),
}

if visual_type=='attn':
    _vtype['attn'](input_tensor,img,camid)
elif visual_type=='grad':
    _vtype['grad'](input_tensor,_img)
else:
    raise TypeError(f'{visual_type} is not supported')