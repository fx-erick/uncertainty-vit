import math
from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from modeling_finetune import PatchEmbed, RelativePositionBias
from modeling_finetune_dist import Block



class DistVisionTransformerForCyclicalTraining(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 init_scale=0.001, linear_classifier=False, has_masking=False,
                 init_std=0.02, gp_layer=False, het_layer=False,
                 sinkformer=False, gumbel_softmax=False, h_sto_trans=False, stosa=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cov_patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cov_cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cov_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
        self.norm = norm_layer(embed_dim)


        self.lm_head = nn.Linear(embed_dim, embed_dim)
        self.cov_lm_head = nn.Linear(embed_dim, embed_dim)

        #self.cov_head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cov_cls_token, std=.02)


        self.apply(self._init_weights)
        self.fix_init_weight()


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

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, bool_masked_pos, layer_results = None):
        mean_x = self.patch_embed(x)
        cov_x = self.cov_patch_embed(x)
        batch_size, seq_len, _ = mean_x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cov_cls_tokens = self.cov_cls_token.expand(batch_size,-1,-1)# stole cls_tokens impl from Phil Wang, thanks

        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        cov_mask_token = self.cov_mask_token.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            w = bool_masked_pos.view(bool_masked_pos.size(0), -1, 1).type_as(mask_token)
            mean_x = mean_x * (1 - w) + mask_token * w
            w = bool_masked_pos.view(bool_masked_pos.size(0), -1, 1).type_as(cov_mask_token)
            cov_x = cov_x * (1 - w) + cov_mask_token * w


        mean_x = torch.cat((cls_tokens, mean_x), dim=1)
        cov_x = torch.cat((cov_cls_tokens, cov_x), dim=1)

        mean_x = self.pos_drop(mean_x)
        cov_x = self.cov_pos_drop(cov_x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        mean_z = []
        cov_z = []
        for blk in self.blocks:
            mean_x, cov_x = blk(mean_x, cov_x, rel_pos_bias=rel_pos_bias)  # B x T x C                layer_xs.append(x)
            if layer_results == 'end':
                mean_z.append(mean_x)
                cov_z.append(cov_x)



        return (mean_z, cov_z) if layer_results else (self.norm(mean_x), self.norm(cov_x))

    def forward(self, x, bool_masked_pos, return_all_tokens = False, layer_results = None):
        mean_x, cov_x = self.forward_features(x, bool_masked_pos, layer_results =layer_results)



        if layer_results:
            return [z[:, 1:] for z in mean_x], [y[:, 1:] for y in cov_x]
        elif return_all_tokens:
            mean_x = mean_x[:, 1:]
            cov_x = cov_x[:, 1:]
            return self.lm_head(mean_x), self.cov_lm_head(cov_x)
        else:
            mean_x = mean_x[:, 1:]
            cov_x = cov_x[:, 1:]
            fsz = mean_x.size(-1)
            bool_masked_pos = bool_masked_pos.flatten().bool()

            mean_x = mean_x.reshape(-1, fsz)[bool_masked_pos]
            cov_x = cov_x.reshape(-1, fsz)[bool_masked_pos]
            return self.lm_head(mean_x), self.cov_lm_head(cov_x)



if __name__ == '__main__':
    model = DistVisionTransformerForCyclicalTraining(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values = 0.1, use_shared_rel_pos_bias = True)
    a = torch.rand(20,3,224,224)
    b = torch.FloatTensor(20,1,14,14).uniform_() > 0.8

    mean_x,cov_x = model(a, b, layer_results = "end")
    dumbi = mean_x[0].size(-1)
    pass