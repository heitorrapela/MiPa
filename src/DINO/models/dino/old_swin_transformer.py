# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------
# modified from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import random
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from util.misc import NestedTensor
import math
from models.dino.diff_ge import DifferentiableStepUnit

swin_backbones_list = ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k',
                         'mixed_modality_swin_T_224_1k', 'mixed_modality_swin_B_224_22k', 'mixed_modality_swin_B_384_22k',
                         'mixed_modality_swin_L_224_22k', 'mixed_modality_swin_L_384_22k']

def sample_n_patches_per_img(num_samples, batch_size, no_of_patches, device):
    ids_shuffle = torch.argsort(torch.rand(batch_size, no_of_patches, device=device), dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    picked_mod2 = torch.ones([batch_size, no_of_patches], device=device)
    picked_mod2[:, :num_samples] = 0
    picked_mod2 = torch.gather(picked_mod2, dim=1, index=ids_restore)

    return picked_mod2

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        dilation (bool): if True, the output size if 16x downsample, ow 32x downsample.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 dilation=False,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation

        if use_checkpoint:
            print("use_checkpoint!!!!!!!!!!!!!!!!!!!!!!!!")

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            num_features[-1] = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=num_features[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=downsamplelist[i_layer],
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')


    def forward_raw(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)


            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # outs:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]
        return tuple(outs)


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors

        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # out:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]

        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

def equals(str1, str2): return str1.casefold() == str2.casefold()
# Sampling methods
FIXED_P, VAR_P_BATCH, VAR_P_IMG, LEARNABLE_P_IMG, LEARNABLE_P_BATCH, CURRICULUM_P = 'fixed', 'variable_per_batch', 'variable_per_img', 'learnable_per_img', 'learnable_per_batch', 'curriculum_per_patch'
NONE, SINGLE_MODALITY, MOOD_MODALITY, SINGLE_OR_MOOD = None, 'single_modality', 'mood_modality', 'single_or_mood'
# Modality regularization methods
class ModalityRegularizer(nn.Module):
    def __init__(self, strong_modality_no: int = 0, regularization_method: str = NONE, prob: int = 5):
        super().__init__()
        self.method = regularization_method if regularization_method is not None else ''
        self.strong_modality_no = strong_modality_no
        self.prob = prob
    
    def single_modality(self, mod1, mod2):
        mods = [mod1, mod2]
        mods[self.strong_modality_no] = torch.zeros(mod1.shape, device=mod1.device)

        return mods

    def mood_modality(self, mod1, mod2):
        mods = [mod1, mod2]
        mods[self.strong_modality_no] = (mod1 + mod2) / 2

        return mods

    def forward(self, mod1, mod2):
        with torch.no_grad():
            if random.randint(0, self.prob) == 0:
                if equals(self.method, SINGLE_MODALITY):
                    mod1, mod2 = self.single_modality(mod1, mod2)
                elif equals(self.method, MOOD_MODALITY):
                    mod1, mod2 = self.mood_modality(mod1, mod2)
                elif equals(self.method, SINGLE_OR_MOOD):
                    mod1, mod2 = self.mood_modality(mod1, mod2) if bool(random.getrandbits(1)) else self.single_modality(mod1, mod2)
            
            return mod1, mod2

class MixedModalitySwinTransformer(SwinTransformer):
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, dilation=False, use_checkpoint=False, sampling_method: str = VAR_P_BATCH, modality_regularization = SINGLE_MODALITY,
                 avg_init: float = -0.5, std_init: float = 1, p_variance_init: float = 0.25,  output_target_modality_map: bool = False, log_stats: bool = True, ir_modality_ratio: float = 0.5):
        super().__init__(pretrain_img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, out_indices, frozen_stages, dilation, use_checkpoint)
        self.sampling_method = sampling_method
        self.iter = 0
        self.iter_per_epoch = 2000
        self.p = ir_modality_ratio
        if self.sampling_method.lower() in LEARNABLE_P_IMG or self.sampling_method.lower() in LEARNABLE_P_BATCH:
            # patch_num = pretrain_img_size // patch_size ** 2
            self.avg = torch.nn.Parameter(avg_init * torch.ones(1), requires_grad = True)
            self.std = torch.nn.Parameter(std_init * torch.ones(1), requires_grad = True)
            self.p_variance = torch.nn.Parameter(p_variance_init * torch.ones(1), requires_grad = True)
            self.unit = DifferentiableStepUnit()
            self.log_stats = log_stats
            if log_stats:
                self.iteration_no = 0
                from datetime import datetime
                self.stats_file_name = f'Stats_Learnable_{datetime.now().strftime("%Y_%m_%d")}.log'
                with open(self.stats_file_name, 'a+') as f:
                    f.write("Stats for Learnable P for each iterations\n")
        self.regularize_modality = ModalityRegularizer(strong_modality_no=0, regularization_method=modality_regularization) # Either None, single_modality, mood_modality, single_or_mood
        self.output_target_modality_map = output_target_modality_map

    def batch_mix_modalities(self, mod1, mod2, p = 0.5):
        assert mod1.shape == mod2.shape, 'Assert Error: Both Modalities should have the same shape!'

        # Maybe use single modality
        mod1, mod2 = self.regularize_modality(mod1, mod2)
        
        batch_size, patch_dim, patch_no_x, patch_no_y = mod1.shape
        if self.sampling_method.lower() in LEARNABLE_P_BATCH:
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
                    x = (x, mod1_map)

        return x

    def img_mix_modalities(self, mod1, mod2):
        assert mod1.shape == mod2.shape, 'Assert Error: Both Modalities should have the same shape!'
        batch_size, patch_dim, patch_no_x, patch_no_y = mod1.shape
        
        if self.sampling_method.lower() in LEARNABLE_P_IMG:
            # Sampling
            norm = torch.normal(mean=torch.zeros(batch_size, patch_no_x * patch_no_y, device=mod1.device), std=torch.ones(batch_size, patch_no_x * patch_no_y, device=mod1.device))
            norm = (norm - self.avg) / self.std
            mi = norm[norm>0].min()
            mod1_picked = (torch.clamp(F.relu(norm), max=mi) / mi).unsqueeze(-1) 
            mod2_picked = 1 - mod1_picked
            # mod1_picked = norm.ge(0).unsqueeze(-1) # TODO ge() not diffentiable !!
            # mod2_picked = ~ mod1_picked

            # Maybe use single modality
            if self.modality_regularization:
                with torch.no_grad():
                    if self.modality_regularization.lower() in SINGLE_MODALITY:
                        sample = random.randint(0, 10)
                        if 3 == sample:
                            mod1 = torch.zeros(mod1.shape, device=mod1.device)
                        if 5 == sample:
                            mod2 = torch.zeros(mod2.shape, device=mod2.device)
                    elif self.modality_regularization.lower() in MOOD_MODALITY:
                        sample = random.randint(0, 10)
                        if 3 == sample:
                            mod1 = (mod1 + mod2)/2
                        if 5 == sample:
                            mod2 = (mod1 + mod2)/2
                    elif self.modality_regularization.lower() in SINGLE_OR_MOOD:
                        sample = random.randint(0, 20)
                        if 4 == sample:
                            mod1 = (mod1 + mod2)/2
                        if 9 == sample:
                            mod1 = torch.zeros(mod1.shape, device=mod1.device)
                        if 12 == sample:
                            mod2 = (mod1 + mod2)/2
                        if 16 == sample:
                            mod2 = torch.zeros(mod2.shape, device=mod2.device)

            # Mixing
            mod1_c = mod1.clone().reshape(batch_size, patch_dim, -1).transpose(-1, -2)
            mod2_c = mod2.clone().reshape(batch_size, patch_dim, -1).transpose(-1, -2)
            x = mod1_c * mod1_picked + mod2_c * mod2_picked
            x = x.transpose(-1, -2).reshape(batch_size, patch_dim, patch_no_x, patch_no_y)

        else :
            with torch.no_grad():
                # Sampling
                mod1_picked = torch.randn(batch_size, patch_no_x * patch_no_y, device=mod1.device).ge(0).unsqueeze(-1)
                mod2_picked = ~ mod1_picked

                # Mixing
                mod1_c = mod1.clone().reshape(batch_size, patch_dim, -1).transpose(-1, -2)
                mod2_c = mod2.clone().reshape(batch_size, patch_dim, -1).transpose(-1, -2)
                x = mod1_c * mod1_picked + mod2_c * mod2_picked
                x = x.transpose(-1, -2).reshape(batch_size, patch_dim, patch_no_x, patch_no_y)

        return x


    def mix_modalities(self, mod1, mod2, p = 0.5):
        assert mod1.shape == mod2.shape, 'Assert Error: Both Modalities should have the same shape!'
        if self.sampling_method.lower() in FIXED_P.lower() or 'batch' in self.sampling_method.lower():
            if self.sampling_method.lower() in VAR_P_BATCH.lower():
                p = random.uniform(0, 1)
            elif self.sampling_method.lower() in CURRICULUM_P.lower():
                curr_epoch = int(self.iter // self.iter_per_epoch)
                if curr_epoch > 8:
                    p = random.uniform(0, 1)
                    # p = 0.25
                else:
                    p = (curr_epoch / 8) * 0.25
                self.iter += 1
            # elif self.sampling_method.lower() in LEARNABLE_P_IMG.lower():
            elif self.sampling_method.casefold() in FIXED_P.lower():
                p = self.p
            x = self.batch_mix_modalities(mod1, mod2, p)
        else:
            x = self.img_mix_modalities(mod1, mod2)

        return x

    def transformer_forward(self, x, mask):
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
     
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

    def forward(self, tensor_list: NestedTensor):
        if self.training:
            mod1, mod2 = tensor_list
            mod1, mod2 = mod1.tensors, mod2.tensors
            mod1, mod2 = self.patch_embed(mod1), self.patch_embed(mod2)
            x = self.mix_modalities(mod1, mod2)
            if self.output_target_modality_map:
                (x, modality_map) = x
            mask = tensor_list[0].mask
        else:
            x = tensor_list.tensors
            x = self.patch_embed(x)
            mask = tensor_list.mask
        
        out = self.transformer_forward(x, mask)
        if self.output_target_modality_map and self.training:
            out[2].modality_map = modality_map

        return out
        
        
def build_swin_transformer(modelname, pretrain_img_size, **kw):
    assert modelname in swin_backbones_list

    model_para_dict = {
        'swin_T_224_1k': dict(
            embed_dim=96,
            depths=[ 2, 2, 6, 2 ],
            num_heads=[ 3, 6, 12, 24],
            window_size=7
        ),        
        'swin_B_224_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=7
        ),
        'swin_B_384_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=12
        ),
        'swin_L_224_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=7
        ),
        'swin_L_384_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=12
        ),
        'mixed_modality_swin_T_224_1k': dict(
            embed_dim=96,
            depths=[ 2, 2, 6, 2 ],
            num_heads=[ 3, 6, 12, 24],
            window_size=7
        ),        
        'mixed_modality_swin_B_224_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=7
        ),
        'mixed_modality_swin_B_384_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=12
        ),
        'mixed_modality_swin_L_224_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=7
        ),
        'mixed_modality_swin_L_384_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=12
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    if 'mixed_modality_' not in modelname:
        del kw_cgf['sampling_method']
        del kw_cgf['modality_regularization']
        del kw_cgf['output_target_modality_map']
        del kw_cgf['ir_modality_ratio']

    model = MixedModalitySwinTransformer(pretrain_img_size=pretrain_img_size, **kw_cgf) if 'mixed_modality_' in modelname else \
            SwinTransformer(pretrain_img_size=pretrain_img_size, **kw_cgf)
    return model

if __name__ == "__main__":
    model = build_swin_transformer('mixed_modality_swin_L_384_22k', 384, dilation=True)
    x = torch.rand(2, 3, 1024, 1024)
    y = model.forward_raw(x)
    x = torch.rand(2, 3, 384, 384)
    y = model.forward_raw(x)