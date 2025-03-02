'''
@File: grad_cam.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 12月 05, 2024
@HomePage: https://github.com/YanJieWen
'''

#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import torch
import cv2
import torch.nn.functional as F

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image

class ActivationsAndGradients:
    def __init__(self,model,target_layers,reshape_transform,cam_label):
        self.model = model
        self.cam_label = cam_label
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform#for VIT
        self.handles = []
        for tgt_l in target_layers:
            self.handles.append(tgt_l.register_forward_hook(hook=self.save_activation))
            #不适用backward_hook:https://github.com/pytorch/pytorch/issues/61519
            self.handles.append(tgt_l.register_forward_hook(hook=self.save_gradient))
    def save_activation(self,m,input,output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x,cam_label=torch.as_tensor([self.cam_label]) if self.cam_label is not None else None)
        return [[model_output['global'],model_output['part']]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class Swin_target(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        glb_feature, _ = data
        results = F.normalize(glb_feature,dim=1).squeeze(dim=0)
        results,_ = torch.sort(results,descending=True)
        # results = []
        # soft_glb = F.sigmoid(glb_feature).squeeze(dim=0)
        # for idx in range(int(soft_glb.size(-1)*0.02)):
        #     if soft_glb[idx]>0.2:
        #         results.append(soft_glb[idx])
        # return max(results)
        return sum(results[:int(results.size(-1)*0.02)])
        # return torch.max(glb_feature)

_methods={
    'GradCAMPlusPlus':GradCAMPlusPlus,
    'GradCAM':GradCAM,
    'XGradCAM':XGradCAM,
    'EigenCAM':EigenCAM,
    'HiResCAM':HiResCAM,
    'LayerCAM':LayerCAM,
    'RandomCAM':RandomCAM,
    'EigenGradCAM':EigenGradCAM,
}

class SwinGradCam:
    def __init__(self,model,target_layer:list=None,method='GradCAM',input_size:tuple=None,cam_label:int=1,model_type:str='swin'):
        height,width = input_size
        model = model
        for p in model.parameters():
            p.requires_grad_(True)
        target = Swin_target()
        method = _methods[method](model,target_layer,use_cuda=True)
        method.activations_and_grads = ActivationsAndGradients(model, target_layer,
                                                               self.reshaoe_transform_swin if model_type=='swin' else self.reshape_transform(),cam_label)
        self.__dict__.update(locals())

    def reshape_transform(self,tensor):
        B, L, C = tensor[:,1:,:].shape#对于vit，第一个是clstoken
        results = tensor[:,1:,:].view(B, self.height, self.width, C).permute(0, 3, 1, 2).contiguous()
        return results
    def reshaoe_transform_swin(self,tensor):
        B, L, C = tensor.shape
        results = tensor.view(B, self.height, self.width, C).permute(0, 3, 1, 2).contiguous()
        return results

    def __call__(self,input_tensor,img:np.array=None):
        img = np.float32(img) / 255.0
        cam_img = self.method(input_tensor,[self.target])
        grayscale_cam = cam_img[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False)
        cv2.imshow("grad_cam", cam_image)
        cv2.imwrite('heatmap_grad.png',cam_image)
        cv2.waitKey(-1)

