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
import vit_seg_configs as configs
from vit_seg_modeling_resnet_skip import *

# import vit_seg_configs as configs
# from vit_seg_modeling_resnet_skip import *

import torch.nn.functional as F

logger = logging.getLogger(__name__)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )
def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis, num, add):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]  # 12
        self.attention_head_size = int(
            config.hidden_size[num + add] / self.num_attention_heads)  # 取整对应起来
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size[num], self.all_head_size)
        self.key = Linear(config.hidden_size[num], self.all_head_size)
        self.value = Linear(config.hidden_size[num], self.all_head_size)

        self.out = Linear(config.hidden_size[num + add], config.hidden_size[num + add])
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
    def __init__(self, config, num, add):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size[num + add], config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size[num + add])
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
    def __init__(self, config, vis, num):
        super(Transformer_layer, self).__init__()
        add = 1 if num % 4 == 0 else 0
        num = num // 4 + 1 * (num % 4 != 0)
        self.attention_norm = LayerNorm(config.hidden_size[num], eps=1e-6)
        self.attn = Attention(config, vis, num, add)
        self.ffn_norm = LayerNorm(config.hidden_size[num + add], eps=1e-6)
        self.ffn = Mlp(config, num, add)
        self.adjust = nn.Linear(config.hidden_size[num],
                                config.hidden_size[num + add]) if add == 1 else nn.Identity()

    def forward(self, x):
       
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + self.adjust(h)

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
    def __init__(self, cin, up_scales, c_nums):
        super(Fusion, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(c_nums, cin, 1),
                                *([nn.UpsamplingBilinear2d(scale_factor=2),
                                   Conv2dReLU(cin, cin, kernel_size=3, stride=1, padding=1)] * (
                                          up_scales // 2))
                                )
        self.down = nn.Sequential(
            *[Conv2dReLU(cin, cin, kernel_size=3, stride=2, padding=1) for _ in
              range(up_scales // 2)],
            nn.Conv2d(cin, c_nums, 1)
        )

        fusion_num = 4
        self.cwf = nn.Sequential(*[Fusion_layer(cin * 2) for _ in range(fusion_num)])

        self.trans_mlp = nn.Sequential(*([nn.LayerNorm(c_nums),
                                          nn.Linear(c_nums, c_nums // 2),
                                          nn.GELU(),
                                          nn.Linear(c_nums // 2, c_nums)] * (fusion_num - 1)))

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
        
class ResBlock(nn.Module):
    def __init__(self,
                 conv,
                 n_feats,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(True),
                 res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Transformer(nn.Module):
    """关键模块"""
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        # cnn
        width_factor = 1
        block_units = (3, 4, 9)
        conv = default_conv
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

        self.head = nn.Sequential(
            conv(3, 3, kernel_size=1),
            ResBlock(conv, 3, 5, act=nn.ReLU(True)),
            ResBlock(conv, 3, 5, act=nn.ReLU(True)),
        )
        self.tail = conv(3, 3, kernel_size=1)

        # 对输入的图像进行embedding
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=3,
                                       out_channels=config.hidden_size[0],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size[0]))
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # transformer层
        self.vis = vis  # 可视化
        self.layer = nn.ModuleList([Transformer_layer(config, vis, num) for num in
                                    range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size[0], eps=1e-6)

        # 融合模块
        up_scales = [4, 2, 1]
        c_ins = [256, 512, 1024]
        self.fusion = nn.ModuleList(
            [Fusion(cin, scale, config.hidden_size[num]) for cin, scale, num in
             zip(c_ins, up_scales, range(1, 4))])

    def forward(self, x):
        
        # x=self.head(x)
        cnn_x = x
        # identity=x
        batch_size=x.shape[0]

        b, c, in_size, _ = cnn_x.size()
        attn_weights = []
        atten_features = []
        cnn_features = []

        # cnn
        cnn_x = self.root(cnn_x)
        cnn_features.append(cnn_x)  # 1/2
        cnn_x = self.pool(cnn_x)

        # embedding
        
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
       # print(x.shape)
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        #print(x.shape)
        x = x + self.position_embeddings
        x = self.dropout(x)

        #print("x input tranform:",x.shape)
        # --------------
        # 第一层
        
        cnn_x = self.body[0](cnn_x)
        cnn_features.append(cnn_x)

        for idx, layer_block in enumerate(self.layer[:2], 1):
        #for idx, layer_block in enumerate(self.layer[:4], 1):
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)
       
        # 融合模块
        # cnn_x:torch.Size([2, 256, 56, 56])
        # x:torch.Size([2, 196, 384])
        x1, x2 = self.fusion[0](cnn_x, x)

        

        cnn_x = cnn_x + x1
        x = x + x2

        # 第二层
        
        cnn_x = self.body[1](cnn_x)
        cnn_features.append(cnn_x)

        for layer_block in self.layer[2:4]:
        #for layer_block in self.layer[4:8]:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)

        # 融合模块
        # cnn_x:torch.Size([2, 512, 28, 28])
        # x:torch.Size([2, 196, 192])
        x1, x2 = self.fusion[1](cnn_x, x)
        # print("fusion 2 cnnx:",x1.shape)
        cnn_x = cnn_x + x1
        x = x + x2
        # 第三层
        
        cnn_x = self.body[-1](cnn_x)

        for layer_block in self.layer[4:6]:
        #for layer_block in self.layer[8:]:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        atten_features.append(x)

        # 融合模块
        # cnn_x:torch.Size([2, 512, 14, 14])
        # x:torch.Size([2, 196, 96])
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

class PVT_MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(PVT_MDTA, self).__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        
        # (C, 4H, 4W) -> (4H, 4W, C)
        self.proj_q = nn.Conv2d(channels, channels, 1)
        self.proj_k = nn.Conv2d(channels, channels, 1)
        self.proj_v = nn.Conv2d(channels, channels, 1)

        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # (B, C, H, W) -> (B, 4H, 4W, C // 4H)
        b, c, h, w = x.shape
        q = self.proj_q(x).reshape(b, self.num_heads, self.head_channels, -1)
        k = self.proj_k(x).reshape(b, self.num_heads, self.head_channels, -1)
        v = self.proj_v(x).reshape(b, self.num_heads, self.head_channels, -1)

        # Transpose (B, 4H, 4W, C//4H) -> (4H, B, 4W, C//4H)
        q = q.transpose(1, 0, 2, 3)   
        k = k.transpose(1, 0, 2, 3)
        v = v.transpose(1, 0, 2, 3)

        # MDTA - Scale dot product attention
        scale = q.shape[-1]**-0.5 
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 0, 2, 3).reshape(b, c, h, w)
        
        # Add and norm 
        x = self.conv(x)
        x = self.norm(x)
        return x

class MDTA(nn.Module):
    '''***IMPORTANT*** - The channels must be zero when divided by num_heads'''
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # *3 for q, k, v & chunk(3, dim=1)
        # 1x1 Conv to aggregate pixel-wise cross-channel context
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # 3x3 DWConv to encode channel-wise spatial context
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        # 1x1 Point-wise Conv
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        '''(N, C, H, W) -> (N, C, H, W)
        Output of MDTA feature should be added to input feature x'''
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)  # (N, C, H, W)

        # divide the # of channels into heads & learn separate attention map
        q = q.reshape(b, self.num_heads, -1, h * w)  # (N, num_heads, C/num_heads, HW)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        # CxC Attention map instead of HWxHW (when num_heads=1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature,
                             dim=-1)  # (N, num_heads, C/num_heads, C/num_heads)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))  # attn*v: (N, num_heads, C/num_heads, HW)

        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)  # channel expansion
        # 1x1 conv to extend feature channel
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)

        # 3x3 DConv (groups=input_channels) -> each input channel is convolved with its own set of filters
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)

        # 1x1 conv to reduce channels back to original input dimension
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        '''HxWxC -> HxWxC
        Output of GDFN feature should be added to input feature x'''
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        # Gating: the element-wise product of 2 parallel paths of linear transformation layers
        x = self.project_out(F.gelu(x1) * x2)

        return x

class PVTBlock(nn.Module):
    '''***IMPORTANT*** - The channels must be zero when divided by num_heads'''
    def __init__(self, channels, num_heads, expansion_factor):
        super(PVTBlock, self).__init__()
        assert channels % num_heads == 0 
        self.norm1 = nn.LayerNorm(channels)
        self.attn = PVT_MDTA(channels, num_heads) # Use PVT Multi-Dims Transformer Attention
        self.norm2 = nn.LayerNorm(channels) 
        self.ffn = GDFN(channels, expansion_factor) 
    def forward(self, x):
        '''(N, C, H, W) -> (N, C, H, W)'''
        b, c, h, w = x.shape
        
        # PAD x to multiply of 4 in spatial dims - Important for PVT!
        pad_h = (4 - h%4)%4
        pad_w = (4 - w%4)%4
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Add PVT MDTA output
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                                       .transpose(-2, -1).contiguous().reshape(b, c, h+pad_h, w+pad_w)) 
        
        # Crop back to original shape
        x = x[:,:,:h,:w]
        
        # Add GDFN output 
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                                       .transpose(-2, -1).contiguous().reshape(b, c, h, w))
        return x
class TransformerBlock(nn.Module):
    '''***IMPORTANT*** - The channels must be zero when divided by num_heads'''
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        assert channels % num_heads == 0
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)
    def forward(self, x):
        '''(N, C, H, W) -> (N, C, H, W)'''
        b, c, h, w = x.shape
        # Add MDTA output feature
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        # ADD GDFN output feature
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        #print("in:",in_channels,"out",out_channels)
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
        self.tranformblock=PVTBlock(in_channels,4,10)
    def forward(self, x, skip=None):
        x=self.tranformblock(x)
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
            config.hidden_size[3],
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
        # x     cnn_future
        B, n_patch, hidden = hidden_states.size()  #reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            # 上采样+融合模块
            #print(i,"-----------x before decoder:",x.shape)
            x = decoder_block(x, skip=features[i] if i < 3 else None)
            #print("x after decoder:",x.shape)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, zero_head=False, vis=False,num_classes=4):
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
        """identity 是需要+的head"""
        # conv=Conv2d(identity.shape[1],x.shape[1],1)
        # x=x+conv(identity)

        logits = self.segmentation_head(x)
        return logits

# 802,816
# 200,704
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
   # print(x.shape)
    x = model(x)
   # print(x.shape)
