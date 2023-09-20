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
        self.primary_conv = conv(c1, c_, k, s, None, g, act=act)
        self.cheap_operation = conv(c_, c_, 3, 1, None, c_, act=act)

    def forward(self, x):
        x = self.primary_conv(x)
        return torch.cat((x, self.cheap_operation(x)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c_, c2, act=True, quant=False):  # ch_in, ch_out
        super().__init__()
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1, act=act, quant=quant),  # pw
            GhostConv(c_ , c2, 1, 1, act=False, quant=quant))  # pw-linear
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                c1,
                c1,
                3,
                stride=1,
                padding=(3 - 1) // 2,
                groups=c1,
                bias=False  # True, #False,,
            ),
            nn.BatchNorm2d(c1),  #
            nn.Conv2d(c1, c2, 1, stride=1, padding=0, bias=False),  # True,False),
            nn.BatchNorm2d(c2),  #
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
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


class SimpleGPAN_1(nn.Module):
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
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=False,
    ):
        super(SimpleGPAN_1, self).__init__()
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        if use_depthwise:
            conv = DWConvQuant if quant else DWConv
        else:
            conv = ConvQuant if quant else Conv

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg, align_corners=False)
        # self.reduce_layers = nn.ModuleList()
        # for idx in range(len(in_channels)):
        #     self.reduce_layers.append(
        #         conv(
        #             in_channels[idx],
        #             out_channels,
        #             1,
        #             act=act_layers(activation),
        #         )
        #     )

        self.reduce_layer0 = conv(in_channels[0], out_channels, 1, act=act_layers(activation))
        self.reduce_layer1 = conv(in_channels[1], out_channels, 1, act=act_layers(activation))

        # self.top_down_blocks = nn.ModuleList()
        # for idx in range(len(in_channels) - 1, 0, -1):
        #     self.top_down_blocks.append(
        #         SimpleGB(
        #             out_channels * 2,
        #             out_channels,
        #             expand,
        #             activation=activation,
        #             quant=quant,
        #         )
        #     )

        self.top_down_blocks = SimpleGB(
                    out_channels * 2,
                    out_channels,
                    expand,
                    activation=activation,
                    quant=quant,
                )

        # build bottom-up blocks
        # self.downsamples = nn.ModuleList()
        # self.bottom_up_blocks = nn.ModuleList()
        # for idx in range(len(in_channels) - 1):
        #     self.downsamples.append(
        #         conv(
        #             out_channels,
        #             out_channels,
        #             k=kernel_size,
        #             s=2,
        #             p=kernel_size // 2,
        #             act=act_layers(activation),
        #         )
        #     )
        #     self.bottom_up_blocks.append(
        #         SimpleGB(
        #             out_channels * 2,
        #             out_channels,
        #             expand,
        #             activation=activation,
        #             quant=quant,
        #         )
        #     )

        self.downsamples = conv(
                    out_channels,
                    out_channels,
                    k=kernel_size,
                    s=2,
                    p=kernel_size // 2,
                    act=act_layers(activation),
                )
        self.bottom_up_blocks = SimpleGB(out_channels * 2, out_channels, expand, activation=activation, quant=quant)

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
        # inputs = [
        #     reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
        # ]

        input0 = self.reduce_layer0(inputs[0])
        input1 = self.reduce_layer1(inputs[1])

        # top-down path

        upsample_feat = self.upsample(input1)
        inner_out = self.top_down_blocks(
            torch.cat([upsample_feat, input0], 1)
        )

        # inner_outs = [inputs[-1]]
        # for idx in range(len(self.in_channels) - 1, 0, -1):
        #     feat_heigh = inner_outs[0]
        #     feat_low = inputs[idx - 1]
        #
        #     # inner_outs[0] = feat_heigh
        #
        #     upsample_feat = self.upsample(feat_heigh)
        #
        #     inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
        #         torch.cat([upsample_feat, feat_low], 1)
        #     )
        #     inner_outs.insert(0, inner_out)

        # bottom-up path
        downsample_feat = self.downsamples(inner_out)
        out = self.bottom_up_blocks(
                torch.cat([downsample_feat, input1], 1)
            )
        outs = [inner_out, out]

        # outs = [inner_outs[0]]
        # for idx in range(len(self.in_channels) - 1):
        #     feat_low = outs[-1]
        #     feat_height = inner_outs[idx + 1]
        #     downsample_feat = self.downsamples[idx](feat_low)
        #     out = self.bottom_up_blocks[idx](
        #         torch.cat([downsample_feat, feat_height], 1)
        #     )
        #     outs.append(out)

        return outs


if __name__ == '__main__':
    model = SimpleGPAN_1(
        in_channels=[8, 8],
        out_channels=8,
        use_depthwise=False,
        kernel_size=3,
        expand=1,
        num_blocks=1,
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
    model = SimpleGPAN_1(
        in_channels=[8, 8],
        out_channels=8,
        use_depthwise=False,
        kernel_size=3,
        expand=1,
        num_blocks=1,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="LeakyReLU",
        quant=False).eval()
    model = model.fuse()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
