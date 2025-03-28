# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from sinkhorn import SinkhornDistance
from sngp import BertLinear, spectral_norm
from uncertainty_evaluations import wasserstein_distance_matmul, kl_distance_matmul
from modeling_finetune_dist import DistVisionTransformer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m




class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, gumbel_softmax = False,
            sinkformer = False, max_iter=3, eps=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gumbel_softmax = gumbel_softmax
        self.sinkformer = sinkformer
        if sinkformer:
            self.sink = SinkhornDistance(eps=eps, max_iter=max_iter)


    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        if self.gumbel_softmax:
            attn = F.gumbel_softmax(attn,dim=-1)
        elif self.sinkformer:
            former_attn_shape = attn.shape
            attn = attn.view(-1, former_attn_shape[2], former_attn_shape[3])
            attn = self.sink(attn)[0]
            attn = attn * attn.shape[-1]
            attn = attn.view(former_attn_shape)
            attn = attn.half()
            # make it to half() for finetuning
            
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualStoSelfAttention(nn.Module):
    def __init__(self, h_size, n_heads, prob_attn, prob_h, tau1 =1, tau2 = 20, n_centroids = 2, rel_pos_bias = None):
        super(DualStoSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.h_size = h_size
        self.head_dim = self.h_size // self.n_heads

        self.query = nn.Linear(h_size, h_size)
        self.key = nn.Linear(h_size, h_size)
        self.value = nn.Linear(h_size, h_size)

        self.dropout_attn = nn.Dropout(p=prob_attn)
        self.dropout_h = nn.Dropout(p=prob_h)
        self.proj = nn.Linear(h_size, h_size)

        self.layer_norm = nn.LayerNorm(h_size)
        self.tau1 = h_size ** 0.5
        self.tau2 = h_size ** 0.5
        #self.tau1 = tau1
        #self.tau2 = tau2
        self.n_centroids = n_centroids

        self.centroid = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.empty(self.head_dim, self.n_centroids), a=-0.5, b=0.5),
            requires_grad=True)

    def forward(self, input, rel_pos_bias=None):
        qq = self.query(input)
        kk = self.key(input)
        vv = self.value(input)

        qq = qq.view(input.shape[0], -1, self.n_heads, self.head_dim)
        kk = kk.view(input.shape[0], -1, self.n_heads, self.head_dim)

        '''

        qq_ = torch.einsum("nshd,dc->nshc", [qq, self.q_centroid])
        q_prob = F.gumbel_softmax(qq_, tau=self.tau1, hard=False, dim=-1)
        #pdb.set_trace()
        sto_qq = torch.einsum("nshc,cd->nshd", [q_prob, self.q_centroid.T])
        sto_qq = sto_qq.view(input.shape[0],-1,self.n_heads,self.head_dim)
        '''

        kk_ = torch.einsum("nshd,dc->nshc", [kk, self.centroid])
        prob = F.gumbel_softmax(kk_, tau=self.tau1, hard=True, dim=-1)
        sto_kk = torch.einsum("nshc,cd->nshd", [prob, self.centroid.T])
        sto_kk = sto_kk.view(input.shape[0], -1, self.n_heads, self.head_dim)

        vv = vv.view(input.shape[0], -1, self.n_heads, self.head_dim)

        qq = qq.transpose(1, 2)
        sto_kk = sto_kk.transpose(1, 2)
        vv = vv.transpose(1, 2)

        interact = torch.matmul(qq, sto_kk.transpose(-1, -2))
        sto_attn_weights = F.gumbel_softmax(interact, tau=self.tau2, hard=True, dim=3)

        attn_weights = sto_attn_weights
        attn_weights = self.dropout_attn(attn_weights)

        output = torch.matmul(attn_weights, vv)
        output = output.transpose(1, 2)
        output = output.contiguous().view(input.shape[0], -1, self.h_size)

        output = self.dropout_h(self.proj(output))

        # could be why its not learning here
        #output = self.layer_norm(output + input)

        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, gumbel_softmax = False, sinkformer = False, h_sto_trans = False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if h_sto_trans:
            self.attn = DualStoSelfAttention( h_size = dim, n_heads = num_heads, prob_attn = attn_drop , prob_h = drop)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                gumbel_softmax= gumbel_softmax, sinkformer = sinkformer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            fc_feature = self.drop_path(self.mlp(self.norm2(x)))
            x = x + fc_feature
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            fc_feature = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x = x + fc_feature
        return x, fc_feature




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, linear_classifier=False, has_masking=False,
                 learn_layer_weights=False, layernorm_before_combine=False, gp_layer = False, het_layer = False,
                 sinkformer = False, gumbel_softmax = False, h_sto_trans = False, sngp = False ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if has_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                sinkformer = sinkformer, gumbel_softmax=gumbel_softmax, h_sto_trans = h_sto_trans )
            for i in range(depth)])
        self.use_mean_pooling = use_mean_pooling
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim, elementwise_affine=not linear_classifier) if use_mean_pooling else None
        if sngp:
            self.fc_norm = spectral_norm(BertLinear(i_dim = embed_dim, o_dim = embed_dim))
        self.het_layer = het_layer
        if gp_layer or sngp:
            self.head = SNGP(embed_dim, embed_dim, num_classes=num_classes)
        if het_layer:
            self.head = MCSoftmaxDenseFA(num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)


        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if has_masking:
            trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        self.learn_layer_weights = learn_layer_weights
        self.layernorm_before_combine = layernorm_before_combine
        if learn_layer_weights:
            self.layer_log_weights = nn.Parameter(torch.zeros(depth,))

        if not gp_layer and not het_layer:
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, bool_masked_pos=None):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        if bool_masked_pos is not None and self.training:
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.view(bool_masked_pos.size(0), -1, 1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        layer_xs = []
        for blk in self.blocks:
            x, _ = blk(x, rel_pos_bias=rel_pos_bias)  # B x T x C
            layer_xs.append(x)

        if self.learn_layer_weights:
            layer_xs = [
                layer_x.mean(1) if self.use_mean_pooling else layer_x[:, 0]
                for layer_x in layer_xs
            ]
            layer_xs = [
                F.layer_norm(layer_x.float(), layer_x.shape[-1:])
                if self.layernorm_before_combine else layer_x
                for layer_x in layer_xs
            ]
            weights = self.layer_log_weights.softmax(-1)
            return F.linear(torch.stack(layer_xs, -1), weights)
        else:
            x = self.norm(x)
            if self.fc_norm is not None:
                t = x[:, 1:, :]
                return self.fc_norm(t.mean(1))
            else:
                return x[:, 0]

    def forward(self, x, bool_masked_pos=None):
        x = self.forward_features(x, bool_masked_pos)
        x = x.float() if self.het_layer else x
        x = self.head(x)
        return x

class SNGP(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_inducing,
                 gp_kernel_scale=1.0,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 n_power_iterations=1,
                 spec_norm_bound=0.95,
                 scale_random_features=True,
                 normalize_input=True,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 epochs=40,
                 num_classes=3,
                 device='gpu'):
        super(SNGP, self).__init__()
        #note, I deleted the spectral norm part
        self.final_epochs = epochs - 1
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = hidden_size

        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_normalize_layer = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes, bias=False)
        # bert gp_output_bias_trainable is false
        #update, issues with num_classes so I changed num_classes , num_inducing....
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
        device = torch.device('cuda')
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        self._random_feature = RandomFeatureLinear(self.pooled_output_dim, num_inducing)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    '''def extract_bert_features(self, latent_feature):
        # https://github.com/google/uncertainty-baselines/blob/b3686f75a10b1990c09b8eb589657090b8837d2c/uncertainty_baselines/models/bert_sngp.py#L336
        # Extract BERT encoder output (i.e., the CLS token).
        first_token_tensors = latent_feature[:, 0, :]
        cls_output = self.last_pooled_layer(first_token_tensors)
        return cls_output'''

    def gp_layer(self, gp_inputs, update_cov=True):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias
        if update_cov:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, cls_output,return_gp_cov: bool = False,
                update_cov: bool = True):

        #latent_feature, _ = self.backbone(input_ids, token_type_ids, attention_mask)
        #cls_output = self.extract_bert_features(latent_feature)
        gp_feature, gp_output = self.gp_layer(cls_output, update_cov=update_cov)
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            return gp_output, gp_cov_matrix

        return gp_output


MIN_SCALE_MONTE_CARLO = 1e-3
class MCSoftmaxOutputLayerBase(nn.Module):
  """Base class for MC heteroscesastic output layers.
  Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton and
  Jesse Berent. Correlated Input-Dependent Label Noise in Large-Scale Image
  Classification. In Proc. of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR), 2021, pp. 1551-1560.
  https://arxiv.org/abs/2105.10305
  """

  def __init__(self,
               num_classes,
               logit_noise=torch.distributions.normal.Normal,
               temperature=1.0,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: float = None,
               temperature_upper_bound: float = None,
               name='MCSoftmaxOutputLayerBase'):
    """Creates an instance of MCSoftmaxOutputLayerBase.
    Args:
      num_classes: Integer. Number of classes for classification task.
      logit_noise: tfp.distributions instance. Must be a location-scale
        distribution. Valid values: tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance. If False the __call__ method will output None for the
        predictive_variance tensor.
      share_samples_across_batch: Boolean. If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference, setting = True may solve.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Useful when a single output Tensor is required e.g.
        tf.keras.Sequential models require a single output Tensor.
      eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.
    Returns:
      MCSoftmaxOutputLayerBase instance.
    Raises:
      ValueError if logit_noise not in tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
    """
    if logit_noise not in (torch.distributions.normal.Normal,
                           torch.distributions.gumbel.Gumbel):
      raise ValueError('logit_noise must be Normal, Logistic or Gumbel')

    super(MCSoftmaxOutputLayerBase, self).__init__()

    self._num_classes = num_classes
    self._logit_noise = logit_noise
    self._temperature = temperature
    self._train_mc_samples = train_mc_samples
    self._test_mc_samples = test_mc_samples
    self._compute_pred_variance = compute_pred_variance
    self._share_samples_across_batch = share_samples_across_batch
    self._logits_only = logits_only
    self._eps = eps
    self._return_unaveraged_logits = return_unaveraged_logits
    self._name = name
    self._tune_temperature = tune_temperature
    self._temperature_lower_bound = temperature_lower_bound
    self._temperature_upper_bound = temperature_upper_bound

    self._pre_sigmoid_temperature = None

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.
    Args:
      scale: Tensor of shape
        [batch_size, 1 if num_classes == 2 else num_classes].
        Scale parameters of the distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      Tensor. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = scale.shape[0]

    dist = self._logit_noise(
        loc=torch.zeros([num_noise_samples, self._num_classes], dtype=scale.dtype),
        scale=torch.ones([num_noise_samples, self._num_classes],
                      dtype=scale.dtype))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    return noise_samples.permute((1, 0, 2)) * torch.unsqueeze(scale, 1)

  def _get_temperature(self):
    if self._tune_temperature:
      return compute_temperature(
          self._pre_sigmoid_temperature,
          lower=self._temperature_lower_bound,
          upper=self._temperature_upper_bound)
    else:
      return self._temperature

  def _compute_mc_samples(self, locs, scale, num_samples, seed):
    """Utility function to compute Monte-Carlo samples (using softmax).
    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = torch.unsqueeze(locs, axis=1)
    noise_samples = self._compute_noise_samples(scale, num_samples, seed)
    latents = locs + noise_samples
    temperature = self._get_temperature()
    if self._num_classes == 2:
      return torch.sigmoid(latents / temperature)
    else:
      return torch.nn.functional.softmax(latents / temperature)

  def _compute_predictive_mean(self, locs, scale, total_mc_samples, seed):
    """Utility function to compute the estimated predictive distribution.
    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      total_mc_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      Tensor of shape [batch_size, 1 if num_classes == 2 else num_classes]
      - the mean of the MC samples and Tensor containing the unaveraged samples.
    """
    '''if self._compute_pred_variance and seed is None:
      seed = utils.gen_int_seed()'''

    samples = self._compute_mc_samples(locs, scale, total_mc_samples, seed)

    return torch.mean(samples, axis=1), samples

  def _compute_predictive_variance(self, mean, locs, scale, seed, num_samples):
    """Utility function to compute the per class predictive variance.
    Args:
      mean: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Estimated predictive
        distribution.
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      seed: Python integer for seeding the random number generator.
      num_samples: Integer. Number of Monte-Carlo samples to take.
    Returns:
      Tensor of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. Estimated predictive variance.
    """
    mean = torch.unsqueeze(mean, axis=1)

    mc_samples = self._compute_mc_samples(locs, scale, num_samples, seed)
    total_variance = torch.mean((mc_samples - mean)**2, axis=1)

    return total_variance

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".
    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".
    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return

  def call(self, inputs, training=True, seed=None):
    """Computes predictive and log predictive distribution.
    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).
    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.
    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). For multi-class
      classification i.e. num_classes > 2 logits = log_probs and logits can be
      used with the standard tf.nn.sparse_softmax_cross_entropy_with_logits loss
      function. For binary classification i.e. num_classes = 2, logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      tf.nn.sigmoid_cross_entropy_with_logits loss function.
    Raises:
      ValueError if seed is provided but model is running in graph mode.
    """
    # Seed shouldn't be provided in graph mode.


    return

  def get_config(self):
    config = {
        'num_classes': self._num_classes,
        'logit_noise': self._logit_noise,
        'temperature': self._temperature,
        'train_mc_samples': self._train_mc_samples,
        'test_mc_samples': self._test_mc_samples,
        'compute_pred_variance': self._compute_pred_variance,
        'share_samples_across_batch': self._share_samples_across_batch,
        'logits_only': self._logits_only,
        'tune_temperature': self._tune_temperature,
        'temperature_lower_bound': self._temperature_lower_bound,
        'temperature_upper_bound': self._temperature_upper_bound,
        'name': self._name,
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config




class MCSoftmaxDenseFA(MCSoftmaxOutputLayerBase):
  """Softmax and factor analysis approx to heteroscedastic predictions."""

  def __init__(self,
               num_classes,
               num_factors = 10,
               temperature=1.0,
               parameter_efficient=False,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=True,
               eps=1e-7,
               dtype=None,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: float = None,
               temperature_upper_bound: float = None,
               name='MCSoftmaxDenseFA'):
    """Creates an instance of MCSoftmaxDenseFA.
    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    y = softmax(u / temperature)
    ```
    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
    u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
    ```
    where V(x) is a matrix of dimension [num_classes, R=num_factors]
    and d(x) is a vector of dimension [num_classes, 1]
    num_factors << num_classes => approx to sampling ~ N(mu(x), sigma(x))
    This is a MC softmax heteroscedastic drop in replacement for a
    tf.keras.layers.Dense output layer. e.g. simply change:
    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```
    to
    ```python
    logits = MCSoftmaxDenseFA(...)(x)[0]
    ```
    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use in approximation to full
        rank covariance matrix.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If false then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance. If False the __call__ method will output None for the
        predictive_variance tensor.
      share_samples_across_batch: Boolean. If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference setting = True may solve.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Set True to serialize tf.keras.Sequential models.
      eps: Float. Clip probabilities into [eps, 1.0] before applying log.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.
    Returns:
      MCSoftmaxDenseFA instance.
    """
    # no need to model correlations between classes in binary case
    assert num_classes > 2
    assert num_factors <= num_classes

    super(MCSoftmaxDenseFA, self).__init__(
        num_classes, logit_noise=torch.distributions.normal.Normal,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only,
        eps=eps,
        return_unaveraged_logits=return_unaveraged_logits,
        tune_temperature=tune_temperature,
        temperature_lower_bound=temperature_lower_bound,
        temperature_upper_bound=temperature_upper_bound,
        name=name)

    self._num_factors = num_factors
    self._parameter_efficient = parameter_efficient
    self._device = torch.device('cuda')
    if parameter_efficient:
      self._scale_layer_homoscedastic = None
      self._scale_layer_heteroscedastic = None
    else:
      self._scale_layer = None
      '''self._scale_layer = tf.keras.layers.Dense(
          num_classes * num_factors, name=name + '_scale_layer', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)'''

    self._loc_layer = None
    self._diag_layer = None

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".
    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    self._loc_layer = torch.nn.Linear(inputs.shape[1], self._num_classes ).to(self._device)
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".
    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
    Returns:
      Tuple of tensors of shape ([batch_size, num_classes * num_factors],
      [batch_size, num_classes]).
    """
    self._diag_layer = torch.nn.Linear(inputs.shape[1], self._num_classes ).to(self._device)
    if self._parameter_efficient:
      return (inputs, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
    else:
      self._scale_layer = torch.nn.Linear(inputs.shape[1], self._num_classes*self._num_factors).to(self._device)
      return (self._scale_layer(inputs),
              self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples, seed):
    """Compute samples of the diagonal elements logit noise.
    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = diag_scale.shape[0]

    dist = torch.distributions.normal.Normal(
        loc=torch.zeros([num_noise_samples, self._num_classes],
                     dtype=diag_scale.dtype).to(self._device),
        scale=torch.ones([num_noise_samples, self._num_classes],
                      dtype=diag_scale.dtype).to(self._device))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    diag_noise_samples = dist.sample(num_samples).to(self._device)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    diag_noise_samples = diag_noise_samples.permute((1, 0, 2))

    return diag_noise_samples * torch.unsqueeze(diag_scale, dim = 1)

  def _compute_standard_normal_samples(self, factor_loadings, num_samples,
                                       seed):
    """Utility function to compute samples from a standard normal distribution.
    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = factor_loadings.shape[0]

    dist = torch.distributions.normal.Normal(
        loc=torch.zeros([num_noise_samples, self._num_factors],
                     dtype=factor_loadings.dtype).to(self._device),
        scale=torch.ones([num_noise_samples, self._num_factors],
                      dtype=factor_loadings.dtype).to(self._device))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    standard_normal_samples = dist.sample(num_samples).to(self._device)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    standard_normal_samples = standard_normal_samples.permute((1, 0, 2))

    if self._share_samples_across_batch:
      standard_normal_samples = torch.tile(standard_normal_samples,
                                        (factor_loadings.shape[0], 1, 1))

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.
    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.
    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    self._scale_layer_homoscedastic = torch.nn.Linear(standard_normal_samples.shape[1], self._num_classes).to(self._device)
    self._scale_layer_heteroscedastic = torch.nn.Linear(factor_loadings.shape[1], self._num_classes).to(self._device)

    if self._parameter_efficient:
      res = self._scale_layer_homoscedastic(standard_normal_samples)
      res *= torch.unsqueeze(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
    else:
      # reshape scale vector into factor loadings matrix
      factor_loadings = torch.reshape(factor_loadings,
                                   (-1, self._num_classes, self._num_factors))

      # transform standard normal into ~ full rank covariance Gaussian samples
      res = torch.einsum('ijk,iak->iaj', factor_loadings, standard_normal_samples)
    return res + diag_noise_samples

  def get_config(self):
    return

  def forward(self, inputs, training=True, seed=42):
      """Computes predictive and log predictive distribution.
      Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
      to compute predictive distribution. O(mc_samples * num_classes).
      Args:
        inputs: Tensor. The input to the heteroscedastic output layer.
        training: Boolean. Whether we are training or not.
        seed: Python integer for seeding the random number generator.
      Returns:
        Tensor logits if logits_only = True. Otherwise,
        tuple of (logits, log_probs, probs, predictive_variance). For multi-class
        classification i.e. num_classes > 2 logits = log_probs and logits can be
        used with the standard tf.nn.sparse_softmax_cross_entropy_with_logits loss
        function. For binary classification i.e. num_classes = 2, logits
        represents the argument to a sigmoid function that would yield probs
        (logits = inverse_sigmoid(probs)), so logits can be used with the
        tf.nn.sigmoid_cross_entropy_with_logits loss function.
      Raises:
        ValueError if seed is provided but model is running in graph mode.
      """
      # Seed shouldn't be provided in graph mode.


      locs = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none
      scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

      if training:
          total_mc_samples = self._train_mc_samples
      else:
          total_mc_samples = self._test_mc_samples

      total_mc_samples = [total_mc_samples]
      probs_mean, _ = self._compute_predictive_mean(
          locs, scale, total_mc_samples, seed)

      pred_variance = None
      if self._compute_pred_variance:
          pred_variance = self._compute_predictive_variance(
              probs_mean, locs, scale, seed, total_mc_samples)

      probs_mean = torch.clamp(probs_mean, self._eps, 1.0)
      log_probs = torch.log(probs_mean)

      if self._num_classes == 2:
          # inverse sigmoid
          probs_mean = torch.clamp(probs_mean, self._eps, 1.0 - self._eps)
          logits = log_probs - torch.log(1.0 - probs_mean)
      else:
          logits = log_probs

      if self._logits_only:
          return logits

      return logits, log_probs, probs_mean, pred_variance



@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("pretrained_cfg")
    _ = kwargs.pop("pretrained_cfg_overlay")
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def dist_beit_base_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("pretrained_cfg")
    _ = kwargs.pop("pretrained_cfg_overlay")
    model = DistVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    _ = kwargs.pop("pretrained_cfg")
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
