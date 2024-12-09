# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from util.misc import NestedTensor
# from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 out_indices=[0, 1, 2, 3]
                 ):
        super().__init__()
        self.dims = dims

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.patch_embed = stem
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return tuple(outs)

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     return x


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        outs = self.forward_features(x)

        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

    def forward_raw(self, x):
        return self.forward_features(x)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def equals(str1, str2): return str1.casefold() == str2.casefold()
# Sampling methods
FIXED_P, VAR_P_BATCH, VAR_P_IMG, LEARNABLE_P_IMG, LEARNABLE_P_BATCH, CURRICULUM_P = 'fixed', 'variable_per_batch', 'variable_per_img', 'learnable_per_img', 'learnable_per_batch', 'curriculum_per_patch'
NONE, SINGLE_MODALITY, MOOD_MODALITY, SINGLE_OR_MOOD = None, 'single_modality', 'mood_modality', 'single_or_mood'

from models.dino.modality_regularizer import ModalityRegularizer
import random

def sample_n_patches_per_img(num_samples, batch_size, no_of_patches, device):
    ids_shuffle = torch.argsort(torch.rand(batch_size, no_of_patches, device=device), dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    picked_mod2 = torch.ones([batch_size, no_of_patches], device=device)
    picked_mod2[:, :num_samples] = 0
    picked_mod2 = torch.gather(picked_mod2, dim=1, index=ids_restore)

    return picked_mod2

class MixedModalityConvNeXt(ConvNeXt):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0,
                 layer_scale_init_value=0.000001, head_init_scale=1, out_indices=[0, 1, 2, 3],
                 sampling_method: str = VAR_P_BATCH, modality_regularization = SINGLE_MODALITY,
                 output_target_modality_map: bool = False, ir_modality_ratio: float = 0.5):
        super().__init__(in_chans, num_classes, depths, dims, drop_path_rate, layer_scale_init_value, head_init_scale, out_indices)
        self.sampling_method = sampling_method
        self.p = ir_modality_ratio
        self.regularize_modality = ModalityRegularizer(strong_modality_no=0, regularization_method=modality_regularization) # Either None, single_modality, mood_modality, single_or_mood
        self.output_target_modality_map = output_target_modality_map


    def forward_raw(self, x):
        x = self.forward_features(x)
        return x

    def batch_mix_modalities(self, mod1, mod2, p = 0.5):
        assert mod1.shape == mod2.shape, 'Assert Error: Both Modalities should have the same shape!'
        
        batch_size, patch_dim, patch_no_x, patch_no_y = mod1.shape
        if equals(self.sampling_method, LEARNABLE_P_BATCH):
            sigma = self.p_variance * F.tanh(torch.normal(mean=torch.zeros(1), std=torch.ones(1))).to(mod1.device)
            norm = torch.normal(mean=sigma * torch.ones(1, patch_no_x * patch_no_y, device=mod1.device), std=torch.ones(1, patch_no_x * patch_no_y, device=mod1.device))
            norm = (norm  + self.avg) / self.std
            mod1_picked = self.unit(norm)#.unsqueeze(-1)
            stats = {}
            stats['p']   = mod1_picked.sum() / mod1_picked.numel()
            stats['avg'] = self.avg.item()
            stats['std'] = self.std.item()
            stats['p_variance'] = self.p_variance.item()
            if self.log_stats:
                from datetime import datetime
                with open(self.stats_file_name, 'a+') as f:
                    f.write(f"Iteration no : {self.iteration_no} | p : {stats['p']} | avg : {stats['avg']} | std : {stats['std']} | p_variance : {stats['p_variance']} |\n")
                self.iteration_no += 1
            # A : Repeat dim=0 batch
            mod1_picked = mod1_picked.repeat(batch_size, 1)
            # B : Shuffle by dimension
            indices = torch.argsort(torch.rand(*mod1_picked.shape), dim=-1)
            mod1_picked = mod1_picked[torch.arange(mod1_picked.shape[0]).unsqueeze(-1), indices].unsqueeze(-1)
            mod2_picked = 1 - mod1_picked

            # Mixing
            mod1_c = mod1.reshape(batch_size, patch_dim, -1).transpose(-1, -2)
            mod2_c = mod2.reshape(batch_size, patch_dim, -1).transpose(-1, -2)
            x = mod1_c * mod1_picked + mod2_c * mod2_picked
            x = x.transpose(-1, -2).reshape(batch_size, patch_dim, patch_no_x, patch_no_y)
        else:
            with torch.no_grad():
                no_of_patches = patch_no_x * patch_no_y
                if p == 1:
                    num_samples = batch_size - 1
                elif p == 0:
                    num_samples = 1
                else:
                    num_samples = int(p * no_of_patches)
                mod2_map = sample_n_patches_per_img(num_samples, batch_size, no_of_patches, mod1.device)
                mod1_map = 1 - mod2_map
                mod1 = mod1.reshape(batch_size, patch_dim, -1).transpose(1,-1)
                mod2 = mod2.reshape(batch_size, patch_dim, -1).transpose(1,-1)
                x = mod1 * mod1_map.unsqueeze(-1) + mod2 * mod2_map.unsqueeze(-1)
                x = x.transpose(1,-1).reshape(batch_size, patch_dim, patch_no_x, patch_no_y)
                if self.output_target_modality_map:
                    x = (x, mod1_map.reshape(batch_size, patch_no_x, patch_no_y))

        return x

    def mix_modalities(self, mod1, mod2, p = 0.5):
        assert mod1.shape == mod2.shape, 'Assert Error: Both Modalities should have the same shape!'
        mod1, mod2 = self.regularize_modality(mod1, mod2)
        if equals(self.sampling_method, VAR_P_BATCH):
            p = random.uniform(0, 1)
        elif equals(self.sampling_method, CURRICULUM_P):
            curr_epoch = int(self.iter // self.iter_per_epoch)
            if curr_epoch > 8:
                p = random.uniform(0, 1)
                # p = 0.25
            else:
                p = (curr_epoch / 8) * 0.25
            self.iter += 1
        elif equals(self.sampling_method, FIXED_P):
            p = self.p
        x = self.batch_mix_modalities(mod1, mod2, p)
        
        return x

    def mixed_forward_features(self, tensor_list):
        outs = []
        for i in range(4):
            if i == 0:
                mod1, mod2 = tensor_list
                mod1, mod2 = mod1.tensors, mod2.tensors
                mod1, mod2 = self.downsample_layers[i](mod1), self.downsample_layers[i](mod2)
                
                if self.output_target_modality_map:    
                    x, modality_map = self.mix_modalities(mod1, mod2)
                else:
                    x = self.mix_modalities(mod1, mod2)
                x = self.stages[i](x)
                    
            else:
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return (tuple(outs), modality_map) if self.output_target_modality_map else tuple(outs)

    def forward(self, tensor_list: NestedTensor):
        if self.training:
            outs = self.mixed_forward_features(tensor_list)
            if self.output_target_modality_map:
                (outs, modality_map) = outs
            m = tensor_list[0].mask
        else:
            x = tensor_list.tensors
            outs = self.forward_features(x)
            m = tensor_list.mask

        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)
        
        if self.output_target_modality_map and self.training:
            outs_dict[2].modality_map = modality_map

        return outs_dict

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# @register_model
# def convnext_tiny(pretrained=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_tiny_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def convnext_small(pretrained=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_small_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def convnext_base(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def convnext_large(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_xlarge_22k'] if in_22k else model_urls['convnext_xlarge_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

def build_convnext(modelname, pretrained,backbone_dir=None, **kw):
    assert modelname in ['convnext_xlarge_22k', 'mixed_modality_convnext_xlarge_22k']

    mixed_modality_prefix = 'mixed_modality_'

    model_para_dict = {
        'convnext_xlarge_22k': dict(
            depths=[3, 3, 27, 3],
            dims=[256, 512, 1024, 2048],
        ),
        'mixed_modality_convnext_xlarge_22k': dict(
            depths=[3, 3, 27, 3],
            dims=[256, 512, 1024, 2048],
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    if mixed_modality_prefix not in modelname:
        del kw_cgf['sampling_method']
        del kw_cgf['modality_regularization']
        del kw_cgf['output_target_modality_map']
        del kw_cgf['ir_modality_ratio']
    model = MixedModalityConvNeXt(**kw_cgf) if mixed_modality_prefix in modelname else ConvNeXt(**kw_cgf)
    if pretrained:
        url = model_urls[modelname.replace(mixed_modality_prefix, '')]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, model_dir=backbone_dir, map_location="cpu", check_hash=True)
        _tmp_st_output = model.load_state_dict(checkpoint["model"], strict=False)
        print(str(_tmp_st_output))

    return model