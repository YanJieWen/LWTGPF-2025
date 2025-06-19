'''
@File: swin.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 27, 2024
@HomePage: https://github.com/YanJieWen
'''
import warnings
from copy import deepcopy
from collections import OrderedDict


from functools import partial
from itertools import repeat

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
to_2tuple = _ntuple(2)

def swin_converted_lu(ckpt):
    new_ckpt = OrderedDict()
    # print(ckpt.keys())
    for k,v in ckpt.items():
        if 'head' in k or 'semantic_embed_b' in k or 'semantic_embed_w' in k or 'backbone.norm' in k:
            continue
        new_v = deepcopy(v)
        new_k = '.'.join(k.split('.')[1:])
        new_ckpt[new_k] = new_v
    return new_ckpt

def swin_converted_imagenet(ckpt):
    new_ckpt = OrderedDict()
    for k,v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.','attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    out_channel, in_channel = v.shape
                    v = v.reshape(out_channel, 4, in_channel // 4)
                    new_v = v[:, [0, 2, 1, 3], :].transpose(1,
                                                        2).reshape(out_channel, in_channel)
                elif 'norm.' in k:
                    in_channel = v.shape[0]
                    v = v.reshape(4, in_channel // 4)
                    new_v = v[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k
        new_ckpt[new_k] = new_v
    # print(new_ckpt.keys())
    return new_ckpt


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def build_dropout(drop_cfg):
    drop_layer = DropPath(drop_cfg['drop_prob'])
    return drop_layer
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

def build_norm_layer(norm_cfg):
    type = norm_cfg['type']
    assert type  in ('LN','BN','IN','IBN','GN')
    if type=='LN':
        norm_layer = partial(nn.LayerNorm,eps=1e-6,)
    elif type=='BN':
        norm_layer = partial(nn.BatchNorm2d,)
    elif type=='IN':
        norm_layer = partial(nn.InstanceNorm2d,affine=True)
    elif type=='IBN':
        norm_layer = partial(IBN,)
    elif type=='GN':
        norm_layer = partial(nn.GroupNorm,num_groups=16)
    else:
        raise ValueError(f'{type}-norm is not found!')
    return norm_layer

def build_activation_layer(act_cfg):
    if act_cfg['type'] == 'ReLU':
        act_layer = nn.ReLU(inplace=act_cfg['inplace'])
    elif act_cfg['type'] == 'GELU':
        act_layer = nn.GELU()
    return act_layer

class AdaptivePadding(nn.Module):
    def __init__(self,kernel_size=1,stride=1,dilation=1,padding='corner'):
        super().__init__()
        assert padding in ('same','corner'), f'{padding} is not satisfy'
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
    def get_pad_shape(self,input_shape):
        input_h,input_w = input_shape
        kernel_h,kernel_w = self.kernel_size
        stride_h,stride_w = self.stride
        output_h = math.ceil(input_h/stride_h)
        output_w = math.ceil(input_w/stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w
    def forward(self,x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner': #右下角填充
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same': #左右上下填充 默认0值
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x

class PatchEmbed(nn.Module): #将图像制作为补丁大小为4的令牌，padding必须为patch_size的整数倍
    def __init__(self,in_channels,embed_dims,kernel_size,stride,padding='corner',dilation=1,
                 bias=True,norm_cfg=None,input_size=None):
        super().__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding,str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.projection = nn.Conv2d(
            in_channels=in_channels,out_channels=embed_dims,kernel_size=kernel_size,
            stride=stride,padding=padding,dilation=dilation,bias=bias
        )
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg)(embed_dims)
        else:
            self.norm = None
        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)
                # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None
    def forward(self,x):
        if self.adap_padding:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

class PatchMerging(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=None,padding='corner',
                 dilation=1,bias=False,norm_cfg=dict(type='LN')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg)(sample_dim)
        else:
            self.norm = None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)
    def forward(self,x,input_size):
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])
        if self.adap_padding:
            x = self.adap_padding(x) #保证x能被stride整除
            H, W = x.shape[-2:]
        x = self.sampler(x)
        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1
        output_size = (out_h, out_w)
        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


def window_partition(x,window_size):
    '''
    将特征图划分为具有wins大小的多个斑块
    :param x:
    :param window_size:
    :return: [b*w/win*h/win,wins,wins,c]
    '''
    b,h,w,c =x.shape
    x = x.view(b,h//window_size,window_size,w//window_size,window_size,c)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,c)
    return windows

def window_reverse(x,window_size,H,W):
    B = int(x.shape[0]/(H*W/window_size/window_size))
    x = x.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x

class WindowMSA(nn.Module):
    def __init__(self,embed_dims,num_heads,window_size,qkv_bias=True,qk_scale=None,
                 attn_drop_rate=0.,proj_drop_rate=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        #define relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h,coords_w],indexing='ij'))
        coords_flat =  torch.flatten(coords,1)
        relative_coords = coords_flat[:,:,None]-coords_flat[:,None,:]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0]+=self.window_size[0]-1
        relative_coords[:,:,1]+=self.window_size[1]-1
        relative_coords[:,:,0]*=2*self.window_size[1]-1
        rel_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index',rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        nn.init.trunc_normal_(self.relative_position_bias_table,std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask=None):
        B,L,C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q,k,v = qkv.unbind(0)#[b,h,l,c/h]
        q = q*self.scale
        attn = (q @ k.transpose(-2, -1)) #[b,h,l,l]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, L,
                             L) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class ShiftWindowMSA(nn.Module):
    def __init__(self,embed_dims,num_heads,window_size,shift_size=0,qkv_bias=True,qk_scale=None,
                 attn_drop_rate=0,proj_drop_rate=0,dropout_layer=dict(type='DropPath', drop_prob=0.)):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate
        )
        self.drop = build_dropout(dropout_layer)
    def create_mask(self,x,H,W):
        Hp = int(np.ceil(H/self.window_size))*self.window_size
        Wp = int(np.ceil(W/self.window_size))*self.window_size
        img_mask = torch.zeros((1,Hp,Wp,1),device=x.device)
        h_slices = (slice(0,-self.window_size),slice(-self.window_size,-self.shift_size),
                    slice(-self.shift_size,None))
        w_slices = (slice(0,-self.window_size),slice(-self.window_size,-self.shift_size),
                    slice(-self.shift_size,None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:,h,w,:] = cnt
                cnt+=1
        mask_windows = window_partition(img_mask,self.window_size)
        mask_windows = mask_windows.view(-1,self.window_size*self.window_size)
        attn_mask = mask_windows.unsqueeze(1)-mask_windows.unsqueeze(2)#boradcast-->[1,ws2,ws2]
        attn_mask = attn_mask.masked_fill(attn_mask!=0,float(-100.)).masked_fill(attn_mask==0,float(0.))
        return attn_mask

    def forward(self,query,hw_shape):
        B,L,C = query.shape
        H,W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        #将特征图pad到整数倍
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b)) #corner填充
        H_pad, W_pad = query.shape[1], query.shape[2]
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            attn_mask = self.create_mask(shifted_query,H_pad, W_pad)
        else:
            attn_mask = None
            shifted_query = query
        query_windows = window_partition(shifted_query,self.window_size)
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows,self.window_size,H_pad,W_pad)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        if pad_r>0 or pad_b>0:
            x = x[:,:H,:W,:].contiguous()
        x = x.view(B,H*W,C)
        x = self.drop(x)
        return x

class FFN(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

class SwinBlock(nn.Module):
    def __init__(self,embed_dims,num_heads,feedforward_channels,window_size=7,shift=False,
                 qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),norm_cfg=dict(type='LN')):
        super().__init__()
        #norm first
        self.norm1 = build_norm_layer(norm_cfg)(embed_dims)
        self.attn = ShiftWindowMSA(embed_dims=embed_dims,num_heads=num_heads,window_size=window_size,
                                   shift_size=window_size//2 if shift else 0,qkv_bias=qkv_bias,qk_scale=qk_scale,
                                   attn_drop_rate=attn_drop_rate,proj_drop_rate=drop_rate,
                                   dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))
        self.norm2 = build_norm_layer(norm_cfg)(embed_dims)
        self.ffn = FFN(embed_dims=embed_dims,feedforward_channels=feedforward_channels,
                       num_fcs=2,ffn_drop=drop_rate,dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                       act_cfg=act_cfg,add_identity=True)
    def forward(self,x ,hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x



class SwinBlockSequence(nn.Module):
    def __init__(self,embed_dims,num_heads,feedforward_channels,depth,window_size=7,
                 qkv_bias=True,qk_scale=None,drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.,
                 downsample=None,act_cfg=dict(type='GELU'),norm_cfg=dict(type='LN')):
        super().__init__()
        if isinstance(drop_path_rate,list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates)==depth, f'{len(drop_path_rates)}!={depth}'
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        self.blocks = nn.ModuleList([SwinBlock(embed_dims=embed_dims,
                                               num_heads=num_heads,feedforward_channels=feedforward_channels,
                                               window_size=window_size,shift=False if i%2==0 else True,
                                               qkv_bias=qkv_bias,qk_scale=qk_scale,
                                               drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,
                                               drop_path_rate=drop_path_rates[i],act_cfg=act_cfg,
                                               norm_cfg=norm_cfg)
                                     for i in range(depth)])
        self.downsample = downsample
    def forward(self,x,hw_shape):
        for block in self.blocks:
            x = block(x,hw_shape)
        if self.downsample:
            x_down,down_hw_shape = self.downsample(x,hw_shape)
            return x_down,down_hw_shape,x,hw_shape
        else:
            return x,hw_shape,x,hw_shape


class SwinTransformer(nn.Module):
    def __init__(self,img_size:int=224,in_channels:int=3,
                 embed_dim:int=96,patch_size:int=4,window_sizes:int=7,mlp_ratio:int=4,
                 depths:tuple=(2,2,6,2),num_heads:tuple=(3,6,12,24),strides:tuple=(4,2,2,2),
                 out_indices:tuple=(1,2,3),qkv_bias:bool=True,qk_scale=None,patch_norm:bool=True,
                 drop_rate:float=0.,attn_drop_rate:float=0.,drop_path_rate:float=0.1,
                 use_abs_pos_embed:bool=False,act_cfg:dict={'type':'GELU'},norm_cfg:dict={'type':'LN'},
                 camera:int=0,view:int=0,sie_coef:float=1.0):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        self.patch_embed = PatchEmbed(in_channels,embed_dim,kernel_size=patch_size,stride=strides[0],
                                      norm_cfg=norm_cfg if patch_norm else None)
        self.cam_num = camera
        self.view_num = view
        self.sie_coef = sie_coef
        self.out_indices = out_indices
        num_layers = len(depths)
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            nn.init.trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_coef))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            nn.init.trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_coef))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            nn.init.trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_coef))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))
        self.use_abs_pos_embed = use_abs_pos_embed #如果使用绝对位置嵌入则需要重塑
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        if self.use_abs_pos_embed:
            patch_row = img_size[0] // patch_size
            patch_col = img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dim)))
            nn.init.trunc_normal_(self.absolute_pos_embed,std=.02)
            print('using absolute position embedding')
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,total_depth)]
        self.stages = nn.ModuleList()
        in_channels = embed_dim
        for i in range(num_layers):#0,1,2,3
            if i<num_layers-1:
                downsample = PatchMerging(in_channels=in_channels,
                                          out_channels=2*in_channels,
                                          stride=strides[i+1],
                                          norm_cfg=norm_cfg if patch_norm else None)
            else:
                downsample = None
            stage = SwinBlockSequence(in_channels,num_heads[i],feedforward_channels=mlp_ratio*in_channels,depth=depths[i],window_size=window_sizes,
                 qkv_bias=qkv_bias,qk_scale=qk_scale,drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                 downsample=downsample,act_cfg=act_cfg,norm_cfg=norm_cfg)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        self.num_features = [int(embed_dim * 2 ** i) for i in range(num_layers)]
        for i in out_indices:
            layer = build_norm_layer(norm_cfg)(self.num_features[i])
            self.add_module(f'norm{i}',layer)

    def load_param(self,model_path=None,convert_type='luperson'):
        print(f'=>loading parameters from {convert_type} model')
        if model_path is None:
            print('No pretrained model, training from scratch')
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.trunc_normal_(m,std=.02)
                elif isinstance(m,nn.LayerNorm):
                    nn.init.constant_(m.bias,0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            ckpt = torch.load(model_path,map_location='cpu')

            if 'teacher' in ckpt:
                ckpt = ckpt['teacher']
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if convert_type=='luperson':
                _state_dict = swin_converted_lu(_state_dict)
            elif convert_type == 'imagenet':
                _state_dict = swin_converted_imagenet(_state_dict)
            else:
                raise ValueError(f'{convert_type} is not supported--luperson,imagenet')
        cnt = 0
        # print('conver:',_state_dict.keys())
        # print('ourmodel:',self.state_dict().keys())
        for k,v in _state_dict.items():
            try:
                self.state_dict()[k].copy_(v)
                cnt+=1
            except:
                print('===========================Warning=========================')
                if convert_type=='luperson':
                    print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,self.state_dict()[
                                                                                                    k].shape))
                else:
                    print(f'k:{k} is not found in model with shape of {v.shape}')
        print('Load %d / %d layers.' % (cnt, len(self.state_dict().keys())))





    def forward(self,x,camera_id=None,view_id=None):
        x,hw_shape= self.patch_embed(x)#下采用4-ratio
        if self.use_abs_pos_embed:
            x = x+self.absolute_pos_embed
        if self.cam_num>0 and self.view_num>0:
            x = x+self.sie_coef*self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num>0:#camera embed
            # print(x.shape,self.sie_embed[camera_id].shape,self.sie_embed.shape,camera_id)
            x = x+self.sie_coef*self.sie_embed[camera_id]
        elif self.view_num>0:
            x = x+self.sie_coef*self.sie_embed[view_id]
        x = self.drop_after_pos(x)
        # print(x.shape)
        outs = OrderedDict()
        outs['hwshapes'] = []
        for i,stage in enumerate(self.stages[:-1]):#最后一个block被用于共享特征
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self,f'norm{i}')
                out = norm_layer(out) #对下采样前的out进行输出
                outs[f'stage{i}']=out
                outs['hwshapes'].append(out_hw_shape)
        return x,outs




from thop import clever_format
from thop import profile

def get_model_info(model,inputs):
    flops,params = profile(model,inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    print(f'=>Swin-Transformer: flops:{flops}\t params:{params}')


def swin_base_patch4_window7_224(img_size=224,drop_rate=0.0, attn_drop_rate=0.0,
                                 drop_path_rate=0.,verbose=True,pretrained_path=None,convert_type='imagenet',**kwargs):

    model = SwinTransformer(img_size = img_size, patch_size=4, window_sizes=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    model.load_param(pretrained_path,convert_type)
    inputs = (torch.randn(1, 3, img_size[0], img_size[1]), torch.as_tensor(1) if model.cam_num > 1 else None, None,)
    get_model_info(model,inputs) if verbose else None
    return model

def swin_small_patch4_window7_224(img_size=224,drop_rate=0.0, attn_drop_rate=0.0,
                                  drop_path_rate=0., verbose=True,pretrained_path=None,convert_type='imagenet',**kwargs):
    model = SwinTransformer(img_size = img_size, patch_size=4, window_sizes=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    model.load_param(pretrained_path, convert_type)
    inputs = (torch.randn(1, 3, img_size[0], img_size[1]), torch.as_tensor(1) if model.cam_num > 1 else None, None,)
    get_model_info(model,inputs) if verbose else None
    return model

def swin_tiny_patch4_window7_224(img_size=224,drop_rate=0.0, attn_drop_rate=0.0,
                                 drop_path_rate=0.,verbose=True,pretrained_path=None,convert_type='imagenet',**kwargs):
    model = SwinTransformer(img_size = img_size, patch_size=4, window_sizes=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    model.load_param(pretrained_path, convert_type)
    inputs = (torch.randn(1, 3, img_size[0], img_size[1]), torch.as_tensor(1) if model.cam_num > 1 else None, None,)
    get_model_info(model,inputs) if verbose else None
    return model


# if __name__ == '__main__':
#     model = swin_small_patch4_window7_224(img_size=480,verbose=True,pretrained_path='../../pretrain/imagenet/swin/swin_small_patch4_window7_224.pth')
#     model.load_param('../../pretrain/imagenet/swin/swin_small_patch4_window7_224.pth',convert_type='imagenet')
    # param_dict = torch.load('../../pretrain/luperson/swin/swin_tiny_teacher.pth',map_location='cpu')
    #
    # mis,exp = model.load_state_dict(param_dict,strict=False)
    # print(mis)
    # print(exp)
    # [print(k) for k in param_dict.keys()]
    # x = torch.rand(2,3,480,512)
    # [print(x.shape) for k,x in model(x).items() if k!='hwshapes']
    # [print(x) for k, x in model(x).items() if k == 'hwshapes']
