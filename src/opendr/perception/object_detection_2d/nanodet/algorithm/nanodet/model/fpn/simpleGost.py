# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import (
    Conv, DWConv, ConvQuant, DWConvQuant, MultiOutput, Concat, fuse_modules)

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v = new_v + divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        activation="ReLU",
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layers(activation)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class Sum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x[0]
        for inputs in x[1:]:
            y = y + inputs
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, r=2, dw_k=5, s=1, act=True, quant=False):  # ch_in, ch_out, kernel, ratio, dw_kernel, stride
        super().__init__()
        self.c2 = c2
        c_ = math.ceil(c2 / r)
        nc = c_ * (r - 1)  # new channels in cheap operation

        conv = ConvQuant if quant else Conv
        # c_ = (c2+1.9) // 2  # hidden channels
        # c_ = c2
        self.primary_conv = conv(c1, c_, k, s, k // 2, act=act)
        self.cheap_operation = conv(c_, nc, dw_k, 1, dw_k // 2, c_, act=act)

    def forward(self, x):
        x = self.primary_conv(x)
        return torch.cat((x, self.cheap_operation(x)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c_, c2, k=3, s=1, se_ratio=0.0, act=True, quant=False):  # ch_in, ch_out
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        conv = ConvQuant if quant else Conv
        self.gb = nn.Sequential(
            GhostConv(c1, c_, 1, 2, act=act, quant=quant),  # pw
            conv(c_, c_, k, s=s, p=(k - 1) // 2, g=c_, act=None) if s > 1 else nn.Identity(), # Depth-wise convolution
            SqueezeExcite(c_, se_ratio=se_ratio) if has_se else nn.Identity(),
            GhostConv(c_, c2, 1, 2, act=False, quant=quant),  # pw-linear
        )
        if c1 == c2 and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv(c1, c1, k=k, s=s, p=(k - 1) // 2, g=c1, act=None),
                conv(c1, c2, k=1, s=1, p=0, act=None),
            )

    def forward(self, x):
        residual = x
        x = self.gb(x)
        return x + self.shortcut(residual)


class SimpleGB(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand=1,
        kernel_size=5,
        num_blocks=1,
        use_res=False,
        activation="LeakyReLU",
        quant=False,
    ):
        super(SimpleGB, self).__init__()
        conv = ConvQuant if quant else Conv
        self.use_res = use_res
        if use_res:
            self.reduce_conv = conv(in_channels, out_channels, k=1, s=1, p=0, act=act_layers(activation))

        blocks = []
        for idx in range(num_blocks):
            in_ch = in_channels if idx == 0 else out_channels
            blocks.append(
                GhostBottleneck(
                    in_ch,
                    int(in_ch * expand),
                    out_channels,
                    k=kernel_size,
                    act=act_layers(activation),
                    quant=quant
                )
            )
        self.blocks = nn.Sequential(*blocks)

    @torch.jit.unused
    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class SimpleGPAN(nn.Module):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        activation (str): Activation layer name.
            Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=False,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2, mode="nearest"), #"bilinear"),
        activation="LeakyReLU",
        quant=False,
    ):
        super(SimpleGPAN, self).__init__()
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.needed = []
        self.out_stages = []
        modes = ["linear", "bilinear", "bicubic", "trilinear"]
        si = len(self.in_channels)  # starting idx

        if use_depthwise:
            conv = DWConvQuant if quant else DWConv
        else:
            conv = ConvQuant if quant else Conv

        # build top-down blocks
        reduce_layer = conv(in_channels[si-1], out_channels, 1, act=act_layers(activation))
        reduce_layer.i = si
        reduce_layer.f = 2

        top_down_blocks = nn.Sequential()
        for idx in range(len(in_channels) -1, 0, -1):
            name_idx = len(in_channels)-1-idx

            upsample = nn.Upsample(**upsample_cfg, align_corners=False if upsample_cfg.mode in modes else None)
            upsample.i = si + 1
            upsample.f = -1
            top_down_blocks.add_module(f"Upsample_{name_idx}", upsample)

            conv_temp = conv(in_channels[idx - 1], out_channels, 1, act=act_layers(activation))
            conv_temp.i = si + 2  # Conv
            conv_temp.f = idx - 1
            top_down_blocks.add_module(f"Conv_{name_idx}", conv_temp)

            concat = Concat(1)
            concat.i = si + 3
            concat.f = [si + 1, -1]
            top_down_blocks.add_module(f"Concat_{name_idx}", concat)

            gb = SimpleGB(out_channels * 2, out_channels, expand=expand, kernel_size=kernel_size, num_blocks=num_blocks,
                          use_res=use_res, activation=activation, quant=quant)
            gb.i = si + 4  # SimpleGB
            gb.f = -1
            top_down_blocks.add_module(f"GB_{name_idx}", gb)

            self.needed.append(idx-1)
            self.needed.append(si + 1)
            si += 4
        ei_top_down = si  # starting idx
        self.needed.append(si)
        self.out_stages.append(si)

        bottom_up_blocks = nn.Sequential()
        for idx in range(len(in_channels) - 1):
            conv_temp = conv(out_channels, out_channels, k=kernel_size, s=2,  p=kernel_size // 2, act=act_layers(activation))
            conv_temp.i = si + 1
            conv_temp.f = -1
            bottom_up_blocks.add_module(f"Downsample_{idx}", conv_temp)

            concat = Concat(1)
            concat.i = si + 2
            concat.f = [-1, ei_top_down-(4*(1+idx))]
            bottom_up_blocks.add_module(f"Concat_{idx}", concat)

            gb = SimpleGB(out_channels * 2, out_channels, kernel_size=kernel_size, expand=expand, num_blocks=num_blocks,
                          use_res=use_res, activation=activation, quant=quant)
            gb.i = si + 3
            gb.f = -1
            bottom_up_blocks.add_module(f"GB_{idx}", gb)
            self.needed.append(ei_top_down-(4*(1+idx)))
            si += 3
            self.needed.append(si)
            self.out_stages.append(si)

        # extra layers
        extra_lvl = nn.Sequential()
        extra_lvl_out_conv_from = self.needed[-1]
        for idx in range(num_extra_level):
            conv_temp = conv(out_channels, out_channels, k=kernel_size, s=2, p=kernel_size // 2, act=act_layers(activation))
            conv_temp.i = si + 1
            conv_temp.f = len(self.in_channels)
            extra_lvl.add_module(f"CI_{idx}", conv_temp)

            conv_temp = conv(out_channels, out_channels, k=kernel_size, s=2, p=kernel_size // 2,
                             act=act_layers(activation))
            conv_temp.i = si + 2
            conv_temp.f = extra_lvl_out_conv_from
            extra_lvl.add_module(f"CO_{idx}", conv_temp)

            conv_temp = conv(out_channels, out_channels, k=kernel_size, s=2, p=kernel_size // 2,
                             act=act_layers(activation))
            conv_temp.i = si + 3
            conv_temp.f = [-1, si + 1]
            extra_lvl.add_module(f"Sum_{idx}", conv_temp)
            extra_lvl.append(
                nn.Sequential(
                    conv(out_channels, out_channels, k=kernel_size, s=2, p=kernel_size // 2, act=act_layers(activation)),
                    conv(out_channels, out_channels, k=kernel_size, s=2, p=kernel_size // 2, act=act_layers(activation)),
                    Sum()
                )

            )
            self.needed.append(len(self.in_channels))
            self.needed.append(si+1)
            si += 3
            self.needed.append(si)
            self.out_stages.append(si)

        self.fpn = nn.Sequential(
            reduce_layer,
            *top_down_blocks,
            *bottom_up_blocks,
            *extra_lvl,
            MultiOutput(),
        )
        self.fpn[-1].i = si+1
        self.fpn[-1].f = self.out_stages

    def fuse(self):
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        y = []
        for input in x:
            y.append(input)
        for layer in self.fpn:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x if layer.i in self.needed else None)
        return x


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    model = SimpleGPAN(
        in_channels=[128, 256, 512],
        out_channels=256,
        use_depthwise=False,
        kernel_size=5,
        num_blocks=1,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=False
    ).eval()
    imgszs = [(1, 128, 136, 240), (1, 256, 68, 120), (1, 512, 34, 60)]
    __dumy_input = [torch.empty(*imgsz, dtype=torch.float, device="cpu") for imgsz in imgszs]
    writer = SummaryWriter(f'./models/gostPan')
    _ = model(__dumy_input)
    writer.add_graph(model.eval(), [__dumy_input], use_strict_trace=False)
    writer.close()
    print(model)
    print("=========================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("=========================================")
    model.fuse()
    print(model)
    print("=========================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("=========================================")
