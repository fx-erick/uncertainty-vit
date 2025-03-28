import math
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from modeling_finetune_try import DropPath, Mlp, PatchEmbed, RelativePositionBias
from timm.models import create_model

from uncertainty_evaluations import wasserstein_distance_matmul, kl_distance_matmul

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, gumbel_softmax = False, sinkformer = False, h_sto_trans = False):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            gumbel_softmax=gumbel_softmax, sinkformer=sinkformer)

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

    def forward(self, x_mean, x_cov, rel_pos_bias=None):
        mean, cov = self.attn(self.norm1(x_mean), self.norm1(x_cov), rel_pos_bias=rel_pos_bias)
        if self.gamma_1 is None:
            x_mean = x_mean + self.drop_path(mean)
            fc_feature_mean = self.drop_path(self.mlp(self.norm2(x_mean)))

            x_cov = x_cov + self.drop_path(cov)
            fc_feature_cov = self.drop_cov(self.mlp(self.norm2(x_cov)))

        else:
            x_mean = x_mean + self.drop_path(self.gamma_1 * mean)
            fc_feature_mean = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_mean)))

            x_cov = x_cov + self.drop_path(self.gamma_1 * cov)
            fc_feature_cov = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cov)))

        x_mean = x_mean + fc_feature_mean
        x_cov = x_cov + fc_feature_cov
        return x_mean,x_cov

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, gumbel_softmax = False,
            sinkformer = False, max_iter=3, eps=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.all_head_dim = all_head_dim
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.cov_qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.cov_q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.cov_v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
            self.cov_q_bias = None
            self.cov_v_bias = None


        self.activation = nn.ELU()

        '''self.attn_dropout = nn.Dropout(attn_drop)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.cov_dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)'''

        self.distance_metric = 'wasserstein'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.cov_proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)



    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, cov_x, rel_pos_bias = None):
        B, N, C = x.shape
        qkv_bias = cov_qkv_bias = None

        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            cov_qkv_bias = torch.cat((self.cov_q_bias, torch.zeros_like(self.cov_v_bias, requires_grad=False), self.cov_v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #q = self.transpose_for_scores(q)
        #k = self.transpose_for_scores(k)
        #v = self.transpose_for_scores(v)

        cov_qkv = self.activation(F.linear(input=cov_x, weight=self.qkv.weight, bias=cov_qkv_bias) )+ 1
        cov_qkv = cov_qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        cov_q, cov_k, cov_v = cov_qkv[0], cov_qkv[1], cov_qkv[2]

        '''cov_q = self.transpose_for_scores(cov_q)
        cov_k = self.transpose_for_scores(cov_k)
        cov_v = self.transpose_for_scores(cov_v)'''

        q = q * self.scale
        #cov_q = cov_q * self.scale

        #if self.distance_metric == 'wasserstein':
        #    attention_scores = d2s_gaussiannormal(wasserstein_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #else:
        #    attention_scores = d2s_gaussiannormal(kl_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        if self.distance_metric == 'wasserstein':
            #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer), self.gamma)
            attn = -wasserstein_distance_matmul(q,cov_q, k, cov_k)
            #attn = attn / torch.max(torch.abs(attn))
            #attn = -torch.log(torch.sigmoid(attn + 1e-24))
            # switch between this for up ( trial 57) and down ( trial 56 )
            attn = torch.sigmoid(attn + 1e-24)

        else:
            attn = (q @ k.transpose(-2, -1))

        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        #mean_context_layer = torch.matmul(attn, v)
        #cov_context_layer = torch.matmul(attn ** 2, cov_v)
        mean_context_layer = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        cov_context_layer = (attn**2 @ cov_v).transpose(1, 2).reshape(B, N, -1)
        '''cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_dim,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)'''

        mean_hidden_states = self.proj(mean_context_layer)
        mean_hidden_states = self.proj_drop(mean_hidden_states)
        #mean_hidden_states = torch.sigmoid(mean_hidden_states)
        #mean_hidden_states = mean_hidden_states + input_mean_tensor

        cov_hidden_states = self.cov_proj(cov_context_layer)
        cov_hidden_states = self.proj_drop(cov_hidden_states)
        #cov_hidden_states = torch.sigmoid(cov_hidden_states)
        #cov_hidden_states = cov_hidden_states + input_cov_tensor

        return mean_hidden_states, cov_hidden_states

class DistVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, linear_classifier=False, has_masking=False,
                 learn_layer_weights=False, layernorm_before_combine=False, gp_layer=False, het_layer=False,
                 sinkformer=False, gumbel_softmax=False, h_sto_trans=False, sngp=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cov_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cov_cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))



        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cov_pos_drop = nn.Dropout(p=drop_rate)

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
                sinkformer=sinkformer, gumbel_softmax=gumbel_softmax, h_sto_trans=h_sto_trans)
            for i in range(depth)])
        self.use_mean_pooling = use_mean_pooling
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim, elementwise_affine=not linear_classifier) if use_mean_pooling else None

        self.head = nn.Linear(embed_dim, num_classes)
        self.cov_lm_head = nn.Identity()
        #self.cov_head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cov_cls_token, std=.02)


        self.apply(self._init_weights)
        self.fix_init_weight()

        self.learn_layer_weights = learn_layer_weights
        self.layernorm_before_combine = layernorm_before_combine
        if learn_layer_weights:
            self.layer_log_weights = nn.Parameter(torch.zeros(depth, ))



    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.attn.cov_proj.weight.data, layer_id + 1)
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

    '''def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()'''

    def forward_features(self, x, bool_masked_pos=None):
        mean_x = self.patch_embed(x)
        cov_x = self.cov_patch_embed(x)
        batch_size, seq_len, _ = mean_x.size()

        mean_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cov_cls_tokens = self.cov_cls_token.expand(batch_size,-1,-1)# stole cls_tokens impl from Phil Wang, thanks


        mean_x = torch.cat((mean_cls_tokens, mean_x), dim=1)
        cov_x = torch.cat((cov_cls_tokens, cov_x), dim=1)

        mean_x = self.pos_drop(mean_x)
        cov_x = self.cov_pos_drop(cov_x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None


        for blk in self.blocks:
            mean_x, cov_x = blk(mean_x, cov_x, rel_pos_bias=rel_pos_bias)  # B x T x C                layer_xs.append(x)


        mean_x = self.norm(mean_x)
        cov_x = self.norm(cov_x)
        if self.fc_norm is not None:
            mean_t = mean_x[:, 1:, :]
            cov_t = cov_x[:, 1:, :]
            return self.fc_norm(mean_t.mean(1)), self.fc_norm(cov_t.mean(1))
        else:
            return mean_x[:, 0], cov_x[:,0]

    def forward(self, x, bool_masked_pos=None):
        mean_x, cov_x = self.forward_features(x, bool_masked_pos)

        '''mean_x = mean_x / torch.max(torch.abs(mean_x))
        cov_x = cov_x / torch.max(torch.abs(cov_x))'''
        '''max = torch.max((torch.abs(mean_x)))
        mean = mean_x + torch.max(torch.abs(mean_x)) + 1e-5
        cov = cov_x + torch.max(torch.abs(cov_x)) + 1e-5

        std = torch.sqrt(cov)
        m = torch.distributions.normal.Normal(mean, std, validate_args=None)

        out = m.sample()
        out = out - max'''
        out = self.head(mean_x)
        return mean_x, cov_x, out


if __name__ == '__main__':
    model = DistVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values = 0.1, use_shared_rel_pos_bias = True)
    a = torch.rand(20,3,224,224)
    mean_x,cov_x, out = model(a)
    pass