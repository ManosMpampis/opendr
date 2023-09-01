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
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import Conv, DWConv, ConvQuant, DWConvQuant
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, quant=False):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        conv = ConvQuant if quant else Conv
        c_ = c2 // 2  # hidden channels
        # c_ = c2
        self.cv1 = conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = conv(c_, c_, 3, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, act=True, quant=False):  # ch_in, ch_out
        super().__init__()
        # c_ = c2 // 2
        c_ = c2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1, act=act, quant=quant),  # pw
            GhostConv(c_ , c2, 1, 1, act=False, quant=quant))  # pw-linear

    def forward(self, x):
        return self.conv(x)


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
        num_blocks=1,
        activation="LeakyReLU",
        quant=False,
    ):
        super(SimpleGB, self).__init__()

        blocks = []
        for idx in range(num_blocks):
            in_ch = in_channels if idx == 0 else in_channels // 2
            blocks.append(
                GhostBottleneck(
                    in_ch,
                    out_channels,
                    act=act_layers(activation),
                    quant=quant
                )
            )
        self.blocks = nn.Sequential(*blocks)

    @torch.jit.unused
    def forward(self, x):
        out = self.blocks(x)
        return out


class SimpleGPAN_2(nn.Module):
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
        num_blocks=1,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=False,
    ):
        super(SimpleGPAN_2, self).__init__()
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_depthwise:
            conv = DWConvQuant if quant else DWConv
        else:
            conv = ConvQuant if quant else Conv

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg, align_corners=False)

        self.reduce_layer0 = conv(in_channels[0], out_channels, 1, act=act_layers(activation))
        self.reduce_layer1 = conv(in_channels[1], out_channels, 1, act=act_layers(activation))
        self.reduce_layer2 = conv(in_channels[2], out_channels, 1, act=act_layers(activation))

        self.top_down_blocks0 = SimpleGB(
                    out_channels * 2,
                    out_channels,
                    num_blocks,
                    activation=activation,
                    quant=quant,
                )

        self.top_down_blocks1 = SimpleGB(
            out_channels * 2,
            out_channels,
            num_blocks,
            activation=activation,
            quant=quant,
        )

        self.downsamples0 = conv(
                    out_channels,
                    out_channels,
                    k=kernel_size,
                    s=2,
                    p=kernel_size // 2,
                    act=act_layers(activation),
                )

        self.downsamples1 = conv(
            out_channels,
            out_channels,
            k=kernel_size,
            s=2,
            p=kernel_size // 2,
            act=act_layers(activation),
        )

        self.bottom_up_blocks0 = SimpleGB(out_channels * 2, out_channels, num_blocks, activation=activation, quant=quant)
        self.bottom_up_blocks1 = SimpleGB(out_channels * 2, out_channels, num_blocks, activation=activation, quant=quant)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, inputs: List[Tensor]):
        """
        Args:
            inputs (List[Tensor]): input features.
        Returns:
            List[Tensor]: multi level features.
        """
        input0 = self.reduce_layer0(inputs[0])
        input1 = self.reduce_layer1(inputs[1])
        input2 = self.reduce_layer2(inputs[2])

        # top-down path
        # first path
        upsample_feat = self.upsample(input2)
        inner_out0 = self.top_down_blocks1(
            torch.cat([upsample_feat, input1], 1)
        )

        # inner_outs[inner_out, inpu2]
        # second path

        upsample_feat = self.upsample(inner_out0)
        inner_out1 = self.top_down_blocks0(
            torch.cat([upsample_feat, input0], 1)
        )
        # inner_outs[inner_out1, inner_out0, input2]

        # outs = [inner_out1]
        # bottom-up path
        # first path
        downsample_feat = self.downsamples0(inner_out1)
        out0 = self.bottom_up_blocks0(
                torch.cat([downsample_feat, inner_out0], 1)
            )

        # outs = [inner_out1, out0]
        downsample_feat = self.downsamples1(out0)
        out1 = self.bottom_up_blocks1(
            torch.cat([downsample_feat, input2], 1)
        )
        outs = [inner_out1, out0, out1]

        return outs


if __name__ == '__main__':
    model = SimpleGPAN_2(
        in_channels=[8, 8],
        out_channels=8,
        use_depthwise=False,
        kernel_size=3,
        num_blocks=2,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=True
    ).eval()

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
    model = SimpleGPAN_2(
        in_channels=[8, 8],
        out_channels=8,
        use_depthwise=False,
        kernel_size=3,
        num_blocks=2,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=False).eval()
    model = model.fuse()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
