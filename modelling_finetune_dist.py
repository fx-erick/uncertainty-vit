import math
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from uncertainty_evaluations import wasserstein_distance_matmul, kl_distance_matmul

class DistBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, gumbel_softmax = False, sinkformer = False):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = DistAttention(
            hidden_size=dim, num_attention_heads=num_heads,
            attention_probs_dropout_prob=attn_drop, hidden_dropout_prob=drop, rel_pos_bias = None)

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
        if self.gamma_1 is None:
            x_mean, x_cov = self.attn(self.norm1(x_mean), self.norm1(x_cov), rel_pos_bias=rel_pos_bias)

            x_mean = x_mean + self.drop_path(x_mean)
            fc_feature_mean = self.drop_path(self.mlp(self.norm2(x_mean)))
            x_mean = x_mean + fc_feature_mean

            x_cov = x_cov + self.drop_path(x_cov)
            fc_feature_cov = self.drop_cov(self.mlp(self.norm2(x_cov)))
            x_cov = x_cov + fc_feature_cov
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            fc_feature = self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x = x + fc_feature
        return x_mean,x_cov
        return x_mean,x_cov

class DistAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob,
                 rel_pos_bias = None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.mean_dense = nn.Linear(hidden_size, hidden_size)
        self.cov_dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.distance_metric = 'wasserstein'
        #self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)



    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, rel_pos_bias = None):
        mixed_mean_query_layer = self.mean_query(input_mean_tensor)
        mixed_mean_key_layer = self.mean_key(input_mean_tensor)
        mixed_mean_value_layer = self.mean_value(input_mean_tensor)

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor)) + 1
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor)) + 1
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor)) + 1

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        #if self.distance_metric == 'wasserstein':
        #    attention_scores = d2s_gaussiannormal(wasserstein_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #else:
        #    attention_scores = d2s_gaussiannormal(kl_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        if self.distance_metric == 'wasserstein':
            #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer), self.gamma)
            attention_scores = -wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)
        else:
            attention_scores = -kl_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        mean_context_layer = torch.matmul(attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(attention_probs ** 2, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        #mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        #cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

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

            self.mean_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.cov_patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

            self.mean_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cov_cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

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
                    sinkformer=sinkformer, gumbel_softmax=gumbel_softmax, h_sto_trans=h_sto_trans)
                for i in range(depth)])
            self.use_mean_pooling = use_mean_pooling
            self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
            self.fc_norm = norm_layer(embed_dim, elementwise_affine=not linear_classifier) if use_mean_pooling else None
            if sngp:
                self.fc_norm = spectral_norm(BertLinear(i_dim=embed_dim, o_dim=embed_dim))
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
                self.layer_log_weights = nn.Parameter(torch.zeros(depth, ))

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
            mean_x = self.mean_patch_embed(x)
            cov_x = self.cov_patch_embed(x)
            batch_size, seq_len, _ = mean_x.size()

            mean_cls_tokens = self.mean_cls_token.expand(batch_size, -1, -1)
            cov_cls_tokens = self.cov_cls_token.expand(batch_size,-1,-1)# stole cls_tokens impl from Phil Wang, thanks


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
            mean_x, cov_x = self.forward_features(x, bool_masked_pos)
            x = x.float() if self.het_layer else x
            x = self.head(x)
            return x
