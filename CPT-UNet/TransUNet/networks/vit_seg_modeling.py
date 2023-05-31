# coding=utf-8

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import *

# import vit_seg_configs as configs
# from vit_seg_modeling_resnet_skip import *

import torch.nn.functional as F

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Transformer_layer(nn.Module):
    """
    transformer layer
    """

    def __init__(self, config, vis):
        super(Transformer_layer, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights


class Fusion_layer(nn.Module):
    def __init__(self, cin):
        super(Fusion_layer, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(cin, cin, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cin, 1),
        )

    def forward(self, x):
        return x + self.fusion(x)


class Fusion(nn.Module):
    """
    交叉融合模块

    """

    def __init__(self, cin, up_scales):
        super(Fusion, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(768, cin, 1),
                                *([nn.UpsamplingBilinear2d(scale_factor=2),
                                   Conv2dReLU(cin, cin, kernel_size=3, stride=1, padding=1)] * (
                                          up_scales // 2))
                                )
        self.down = nn.Sequential(
            *[Conv2dReLU(cin, cin, kernel_size=3, stride=2, padding=1) for _ in
              range(up_scales // 2)],
            nn.Conv2d(cin, 768, 1)
        )

        fusion_num = 4
        self.cwf = nn.Sequential(*[Fusion_layer(cin * 2) for _ in range(fusion_num)])

        self.trans_mlp = nn.Sequential(*([nn.LayerNorm(768),
                                          nn.Linear(768, 768 // 2),
                                          nn.GELU(),
                                          nn.Linear(768 // 2, 768)] * (fusion_num - 1)))

        self.cnn_mlp = nn.Sequential(*([nn.Conv2d(cin, cin, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(cin, cin, 3, 1, 1)] * (fusion_num - 1)))

    def forward(self, x, trans_x):
        B, n_patch, hidden = trans_x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        trans_x = trans_x.permute(0, 2, 1).view(B, hidden, h, w)

        # 上采样通道层
        trans_x = self.up(trans_x)
        x = torch.cat([x, trans_x], dim=1)

        # 融合trans和cnn
        x = self.cwf(x)
        x, trans_x = torch.chunk(x, chunks=2, dim=1)
        x = self.cnn_mlp(x)
        trans_x = self.down(trans_x)
        trans_x = self.trans_mlp(trans_x.view(B, hidden, -1).permute(0, 2, 1).contiguous())

        return x, trans_x

    
class Transformer(nn.Module):
    """关键模块"""

    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        # cnn
        width_factor = 1
        block_units = (3, 4, 9)
        width = int(64 * width_factor)
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i
                 in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1',
                  PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for
                 i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1',
                  PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))
                 for i in range(2, block_units[2] + 1)],
            ))),
        ]))

        # 对输入的图像进行embedding
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=3,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # transformer层
        self.vis = vis  # 可视化
        self.layer = nn.ModuleList([Transformer_layer(config, vis) for _ in
                      range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # 融合模块
        up_scales = [4, 2, 1]
        c_ins = [256, 512, 1024]
        self.fusion = nn.ModuleList([Fusion(cin, scale) for cin, scale in zip(c_ins, up_scales)])

    def forward(self, x):
        cnn_x = x
        b, c, in_size, _ = cnn_x.size()
        attn_weights = []
        atten_features = []
        cnn_features = []

        # cnn
        cnn_x = self.root(cnn_x)
        cnn_features.append(cnn_x)  # 1/2
        cnn_x = self.pool(cnn_x)
        # print('after root+pool:', cnn_x.shape, cnn_x.device, cnn_x.is_contiguous())
        
        # embedding
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2).contiguous()  # (B, n_patches, hidden)
        x = x + self.position_embeddings
        x = self.dropout(x)
        # print('after embedding:', x.shape, x.is_contiguous())
        
        
        # --------------
        # 第一层
        cnn_x = self.body[0](cnn_x)
        cnn_features.append(cnn_x)

        for idx, layer_block in enumerate(self.layer[:4], 1):
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)

        # print(f'first layer: cnn_x:{cnn_x.shape} x:{x.shape}')
        # 融合模块
        # cnn_x:torch.Size([2, 256, 56, 56])
        # x:torch.Size([2, 196, 768])
        x1, x2 = self.fusion[0](cnn_x, x)
        cnn_x =cnn_x + x1
        x = x + x2

        # 第二层
        cnn_x = self.body[1](cnn_x)
        cnn_features.append(cnn_x)

        for layer_block in self.layer[4:8]:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)

        # print(f'second layer: cnn_x:{cnn_x.shape} x:{x.shape}')
        # 融合模块
        # cnn_x:torch.Size([2, 512, 28, 28])
        # x:torch.Size([2, 196, 768])
        x1, x2 = self.fusion[1](cnn_x, x)
        cnn_x =cnn_x + x1
        x = x + x2
        
        # 第三层
        cnn_x = self.body[-1](cnn_x)

        for layer_block in self.layer[8:]:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)

        # print(f'third layer: cnn_x:{cnn_x.shape} x:{x.shape}')
        # 融合模块
        # cnn_x:torch.Size([2, 512, 14, 14])
        # x:torch.Size([2, 196, 768])
        _, encoded = self.fusion[2](cnn_x, x)  # encoded.shape= [2, 196, 768]

        return encoded, attn_weights, cnn_features[::-1]


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),  # !
        )
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])  # 512 256 128 64
        out_channels = decoder_channels  # (256, 128, 64, 16)
        skip_channels = [512, 256, 64, 0]  # [512, 256, 64, 16]
        self.blocks = nn.ModuleList([DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
                                     zip(in_channels, out_channels, skip_channels)])

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).view(B, hidden, h, w)
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            # 上采样+融合模块
            x = decoder_block(x, skip=features[i] if i < 3 else None)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, cnn_features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, cnn_features)
        logits = self.segmentation_head(x)
        return logits

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

if __name__ == '__main__':
    configs = CONFIGS['ViT-B_16']
    model = VisionTransformer(configs, img_size=224)
    x = torch.randn(2, 3, 224, 224)
    x = model(x)
    print(x.shape)
