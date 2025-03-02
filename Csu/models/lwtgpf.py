'''
@File: lwtgpf.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 11月 29, 2024
@HomePage: https://github.com/YanJieWen
'''

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmengine.model import constant_init,normal_init

from .swinlu import swin_base_patch4_window7_224,build_norm_layer,get_model_info,swin_small_patch4_window7_224


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class FPN(nn.Module):
    def __init__(self,in_channels,out_channels,start_level=0,end_level=-1,):
        super().__init__()
        assert isinstance(in_channels,list), f'{in_channels} must be mutil levels'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(start_level,end_level):
            l_conv = nn.Conv2d(in_channels[i],out_channels,1)
            fpn_conv = nn.Conv2d(out_channels,out_channels,3,1,1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    def forward(self,x):
        assert len(x)==len(self.in_channels)
        laterals = [lateral_conv(x[i]) for i,lateral_conv in enumerate(self.lateral_convs)]

        #build top-down path
        use_backbone_levels = len(laterals)
        for i in range(use_backbone_levels-1,0,-1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i-1] = laterals[i-1]+F.interpolate(laterals[i],size=prev_shape,mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(use_backbone_levels)]
        return outs


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DyRelu(nn.Module):
    def __init__(self,cardinality,ratio=4):
        super().__init__()
        self.channels = cardinality
        self.expansion = 4
        self.global_avgpool =  nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=cardinality,out_channels=int(cardinality/ratio),
                               kernel_size=1,stride=1)
        self.act_0 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=int(cardinality/ratio),out_channels=cardinality*self.expansion,
                               kernel_size=1,stride=1)
        self.act_1 = Hsigmoid(inplace=True)
    def forward(self,x):
        coeffs = self.global_avgpool(x)
        coeffs = self.act_0(self.conv1(coeffs))
        coeffs = self.act_1(self.conv2(coeffs))-0.5
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
        a1 = a1 * 2.0 + 1.0
        a2 = a2 * 2.0
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out

class DCNv2(nn.Module):
    def __init__(self,in_channelse,out_channels,stride=1,norm_cfg=dict(type='GN')):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(in_channelse,out_channels,3,stride=stride,
                                          padding=1,bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg)(num_channels=out_channels)
    def forward(self,x,offset,mask):
        x = self.conv(x.contiguous(),offset,mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class MultiAttentionBlock(nn.Module):
    def __init__(self,in_channels,out_channels,zeri_init_offset=True):
        super().__init__()
        self.zero_init_offset = zeri_init_offset
        self.offset_and_mask_dim = 3*3*3 #kernel size
        self.offset_dim = 2*3*3
        self.spatial_conv_high = DCNv2(in_channels,out_channels)
        self.spatial_conv_mid = DCNv2(in_channels,out_channels)
        self.spatial_conv_low = DCNv2(in_channels,out_channels,stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), Hsigmoid())
        self.task_attn_module = DyRelu(out_channels)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)
    def forward(self,x):
        '''
        高分辨率图与中等水平结合，中等水平同时融合高和低分辨率图，低分辨率与中等水平图融合
        :param x:
        :return:[C,H,W]-->[D,H,W]
        '''
        outs = []
        for l in range(len(x)):
            offset_mask = self.spatial_conv_offset(x[l])
            offset = offset_mask[:, :self.offset_dim, :, :]
            mask = offset_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[l], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if l > 0:
                low_feat = self.spatial_conv_low(x[l - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if l < len(x) - 1:
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[l + 1], offset, mask),
                    size=x[l].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs

class MTAhead(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks=6,zero_init_offset=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.zero_init_offset = zero_init_offset
        mta = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i==0 else self.out_channels
            mta.append(
                MultiAttentionBlock(
                    in_channels,out_channels,zero_init_offset
                )
            )
        self.layers = nn.Sequential(*mta)

    def forward(self,x):
        outs = self.layers(x)
        return outs




def patch_reverse(features,sizes):
    '''

    :param features: list[Tensor]
    :param sizes: list[()]
    :return:list[(b,c,h,w)]
    '''
    new_features = []
    for feature,s in zip(features,sizes):
        B, _, C = feature.size()
        _feature = feature.view(B,s[0],s[1],C).permute(0,3,1,2).contiguous()
        new_features.append(_feature)
    return new_features

def patch_merge(x):
    B,C,H,W = x.shape
    _feature = x.view(B,C,-1)
    return _feature

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LWTGPF(nn.Module):
    '''
    Local Enhanced Window-based Transformer Global-Part Features encoder
    '''
    __factory = {'lwtgpf': swin_small_patch4_window7_224}

    def __init__(self,arch,img_size,sie_coef,camera_num,view_num,out_indices,drop_path_rate,
                 drop_rate,attn_drop_rate,pretrain_path,num_parts,has_head,
                 granularities,branch,convert_type,swin_verbose,global_feature_type,**kwargs):
        super().__init__()
        print(f'using Swin Transformer_type: {arch} as a backbone')
        assert sum(granularities) == num_parts
        assert branch in ('all', 'b1', 'b2')
        if camera_num:
            camera_num = camera_num
        else:
            camera_num = 0
        if view_num:
            view_num = view_num
        else:
            view_num = 0
        self.base = LWTGPF.__factory[arch](img_size=img_size,drop_rate=drop_rate,attn_drop_rate=attn_drop_rate,sie_coef=sie_coef,
                                           drop_path_rate=drop_path_rate,verbose=swin_verbose,pretrained_path=pretrain_path,
                                           convert_type=convert_type,out_indices=out_indices,camera=camera_num,view=view_num,**kwargs)
        self.num_features = [self.base.num_features[id] for id in out_indices]
        self.has_head = has_head
        self.granularities = granularities
        self.branch = branch
        self.global_feature_type = global_feature_type

        self.num_parts = num_parts
        self.fmap_h = img_size[0]//32
        self.fmap_w = img_size[1]//32
        self.hw_shapes = [self.fmap_h, self.fmap_w]
        # self.fpn = FPN(in_channels=self.num_features,out_channels=fpn_channels,end_level=len(self.num_features))
        if self.has_head:
            # self.head = MTAhead(in_channels=fpn_channels,out_channels=head_channels,
            #                     num_blocks=head_nums,zero_init_offset=True)
            block = self.base.stages[-1]
            self.b1 = copy.deepcopy(block)
            self.b2 = copy.deepcopy(block)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        #bottom-top--global-->downsample&patial pooling
        # self.bt_pool = nn.AvgPool2d(2,stride=2)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        #top-down--part
        # self.td_pool = nn.Upsample(scale_factor=2,mode='nearest')
        for i,g in enumerate(self.granularities):
            setattr(self,f'b{int(i+1)}_pool',nn.AdaptiveAvgPool2d((g,1)))

        #global bottleneck
        in_fans = self.base.num_features[-1]
        self.bottleneck = self.make_bnneck(in_fans,weights_init_kaiming)
        #part bottleneck
        self.part_bns = nn.ModuleList(
            [self.make_bnneck(in_fans,weights_init_kaiming) for _ in range(self.num_parts)]
        )

    # def scale2ml(self,x):#将层次化特征缩放到中间规模
    #
    #     assert len(x)==3 and len(x[0].shape)==4,f'input feature mode is not meet'
    #     high,medium,low = x[0],x[1],x[2]
    #     b2t = medium + self.td_pool(low) #b,c,h,w
    #     t2d = medium+self.bt_pool(high)#b,c,h,w
    #     return b2t,t2d

    def forward_single_branch(self,x,branch,label=None,cam_label=None,view_label=None):
        x,_ = self.base(x,camera_id=cam_label,view_id=view_label)
        B = x.size(0)
        x_glb = self.global_pool(x.permute(0,2,1).contiguous()).view(B,-1)
        x_patch = x.permute(0,2,1).reshape((B,-1,self.fmap_h,self.fmap_w))
        x_part = getattr(self,f'{branch}_pool')(x_patch).squeeze(-1)
        # features = [v for k,v in x.items() if 'stage' in k]
        # hw_shapes = x['hwshapes']
        # x = patch_reverse(features,hw_shapes)
        # x = self.fpn(x)
        # #scale2midium level
        # global_f,part_f = self.scale2ml(x)
        # global_f = patch_merge(global_f)
        # x_glb = self.global_pool(global_f).squeeze(dim=-1)#b,c
        # x_part = getattr(self,f'{branch}_pool')(part_f).squeeze(dim=-1)#b,c
        return x_glb,x_part

    def forward_multi_branch(self,x,label=None,cam_label=None,view_label=None):
        x,_= self.base(x, camera_id=cam_label, view_id=view_label)
        B = x.size(0)

        #branch-1-->BXLXD
        x_b1,_,_,_ = self.b1(x,self.hw_shapes)
        x_b1_glb = self.global_pool(x_b1.permute(0,2,1).contiguous()).view(B, -1)
        x_b1_patch = x_b1.permute(0,2,1).reshape((B,-1,self.fmap_h,self.fmap_w))
        x_b1_patch = self.b1_pool(x_b1_patch).squeeze(-1)
        #brachn-2
        x_b2,_,_,_ = self.b2(x,self.hw_shapes)
        x_b2_glb = self.global_pool(x_b2.permute(0,2,1).contiguous()).view(B,-1)
        x_b2_patch = x_b2.permute(0,2,1).reshape((B,-1,self.fmap_h,self.fmap_w))
        x_b2_patch = self.b2_pool(x_b2_patch).squeeze(-1)
        if self.global_feature_type == 'mean':
            x_glb = 0.5 * (x_b1_glb + x_b2_glb) # (B, C)
        elif self.global_feature_type == 'b1':
            x_glb = x_b1_glb
        elif self.global_feature_type == 'b2':
            x_glb = x_b2_glb
        else:
            raise ValueError('Invalid global feature type: {}'.format(self.global_feature_type))
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2)
        # features = [v for k, v in x.items() if 'stage' in k]
        # hw_shapes = x['hwshapes']
        # x = patch_reverse(features, hw_shapes)
        # x = self.fpn(x)
        # x = self.head(x)
        #
        # global_f, part_f = self.scale2ml(x)
        # global_f = patch_merge(global_f)#b,c,l
        # x_glb = self.global_pool(global_f).squeeze(dim=-1)
        # x_b1_patch = self.b1_pool(part_f).squeeze(dim=-1)
        # x_b2_patch = self.b2_pool(part_f).squeeze(dim=-1)
        # x_part = torch.cat([x_b1_patch,x_b2_patch],dim=2)
        return x_glb,x_part
    def forward_multi_branch_no_head(self,x,label=None,cam_label=None,view_label=None):
        x,_ = self.base(x, camera_id=cam_label, view_id=view_label)
        B = x.size(0)
        x_glb = self.global_pool(x.permute(0,2,1).contiguous()).view(B, -1)
        x_patch = x.permute(0, 2, 1).reshape((B, -1, self.fmap_h, self.fmap_w))
        x_b1_patch = self.b1_pool(x_patch).squeeze(-1)
        x_b2_patch = self.b2_pool(x_patch).squeeze(-1)
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2)
        # features = [v for k, v in x.items() if 'stage' in k]
        # hw_shapes = x['hwshapes']
        # x = patch_reverse(features, hw_shapes)
        # x = self.fpn(x)
        # global_f, part_f = self.scale2ml(x)
        # global_f = patch_merge(global_f)
        # x_glb = self.global_pool(global_f).squeeze(dim=-1)
        # x_b1_patch = self.b1_pool(part_f).squeeze(dim=-1)
        # x_b2_patch = self.b2_pool(part_f).squeeze(dim=-1)
        # x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2)
        return x_glb, x_part



    def forward(self,x,label=None,cam_label=None,view_label=None):
        B = x.size(0)
        if self.has_head:
            x_glb,x_part = self.forward_multi_branch(x,label,cam_label,view_label)
        elif self.branch!='all':
            x_glb,x_part = self.forward_single_branch(x,self.branch,label,cam_label,view_label)
        else:
            x_glb,x_part = self.forward_multi_branch_no_head(x,label,cam_label,view_label)
        #x_glb:(b,d),x_part:[b,d,5]
        x_glb = self.bottleneck(x_glb)

        x_part = torch.stack([self.part_bns[i](x_part[...,i])for i in range(x_part.size(2))],dim=2)
        x_glb = F.normalize(x_glb, dim=1)
        x_part = F.normalize(x_part, dim=1)
        assert x_part.size(2) == self.num_parts, 'x_part size: {} != num_parts: {}'.format(
            x_part.size(2), self.num_parts)  # check part num
        return {'global': x_glb, 'part': x_part.permute(2, 0, 1)}

    def make_bnneck(self, dims, init_func):
        bn = nn.BatchNorm1d(dims)
        bn.bias.requires_grad_(False)  # disable bias update
        bn.apply(init_func)
        return bn


def lwtgpf(verbose=True,**kwargs):
    h,w = kwargs['img_size']
    model = LWTGPF(**kwargs)
    inputs = (torch.randn(1, 3, h, w),None,torch.as_tensor(1) if model.base.cam_num > 1 else None, None,)
    get_model_info(model,inputs) if verbose else None
    return model


if __name__ == '__main__':
#     # arch, img_size, sie_coef, camera_num, view_num, out_indices, drop_path_rate,
#     # drop_rate, attn_drop_rate, pretrain_path, num_parts, has_head, fpn_channels, head_channels, head_nums,
#     # granularities, branch, convert_type, swin_verbose,
#     # img_size = img_size, drop_rate = drop_rate, attn_drop_rate = attn_drop_rate, sie_coef = sie_coef,
#     # drop_path_rate = drop_path_rate, verbose = swin_verbose, pretrained_path = pretrain_path,
#     # convert_type = convert_type, out_indices = out_indices, camera = camera_num, view = view_nu
    model = lwtgpf(verbose=True,arch='lwtgpf',img_size=(416,512),drop_rate=0.,attn_drop_rate=0.,sie_coef=3.0,drop_path_rate=0.1,
                   camera_num =6,view_num=0,out_indices=(1,2,3),
                   pretrain_path='../../pretrain/luperson/swin/swin_base_teacher.pth',num_parts=5,has_head=True,
                   global_feature_type='mean',granularities=[2,3],
                   branch='all',convert_type='luperson',swin_verbose=True)



