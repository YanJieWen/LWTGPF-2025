'''
@File: swin_rollout.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 05, 2024
@HomePage: https://github.com/YanJieWen
'''

import cv2

import torch
import numpy as np


def rollout(attentions, discard_ratio, head_fusion):
    maps = attentions[-1]
    N,H,T2,_ = maps.shape
    maps = maps.view(1,N,H,T2,T2)
    maps_list = [maps[:,n,:,:,:] for n in range(N)]
    result = torch.eye(maps_list[0].size(-1))
    with torch.no_grad():
        for attention in maps_list:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    # print(result.shape)
    mask = result[0, 0, :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class AttentionRollout:
    def __init__(self,model,attention_layer_name:str=None,head_fusion:str='mean',discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name,m in self.model.named_modules():
            if attention_layer_name==name:
                m.register_forward_hook(self.get_attention)
        self.attentions = []
    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())


    def __call__(self, input_tensor,img:np.array=None,cam_label=None):#bgr
        h,w = img.shape[:-1]
        with torch.no_grad():
            output = self.model(input_tensor,cam_label=torch.as_tensor([cam_label]) if cam_label is not None else None)
            mask = rollout(self.attentions, self.discard_ratio, self.head_fusion)
        mask = cv2.resize(mask, (w, h))
        mask = show_mask_on_image(img, mask)
        cv2.imshow("roll_out", mask)
        cv2.imwrite("heatmap_attn.png", mask)
        cv2.waitKey(-1)
        # return rollout(self.attentions, self.discard_ratio, self.head_fusion)
