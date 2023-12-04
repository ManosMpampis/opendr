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

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import Conv, C3, Concat
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers


class Wm:
    def __init__(self, width_multiplier=0.5):
        self.width_multiplier = width_multiplier

    def __call__(self, width):
        return int(width * self.width_multiplier)


class Dm:
    def __init__(self, depth_multiplier=0.33):
        self.depth_multiplier = depth_multiplier

    def __call__(self, depth):
        return max(round(depth * self.depth_multiplier), 1) if depth > 1 else depth


class YoloPan(nn.Module):
    YoloArch = dict(
        x=dict(width_multiple=1.25, depth_multiple=1.33),
        l=dict(width_multiple=1.0,  depth_multiple=1.0),
        m=dict(width_multiple=0.75, depth_multiple=0.67),
        s=dict(width_multiple=0.5,  depth_multiple=0.33),
        n=dict(width_multiple=0.25, depth_multiple=0.33),
    )
    def __init__(
        self,
        arch="s",
        kernel_size=3,
        upsample_cfg=None,
        activation="SiLU",
    ):
        super(YoloPan, self).__init__()

        if upsample_cfg is None:
            upsample_cfg = dict(scale_factor=2, mode="nearest")
        width_multiple = self.YoloArch[arch]["width_multiple"]
        depth_multiple = self.YoloArch[arch]["depth_multiple"]
        wm = Wm(width_multiple)
        dm = Dm(depth_multiple)

        # build top-down blocks
        self.layer_10 = Conv(wm(1024), wm(512), 1, 1, act=act_layers(activation))
        self.layer_11 = nn.Upsample(**upsample_cfg, align_corners=False if upsample_cfg["mode"] != "nearest" else None)
        self.layer_12 = Concat(1)
        self.layer_13 = C3(wm(1024), wm(512), dm(3), False)

        self.layer_14 = Conv(wm(512), wm(256), 1, 1, act=act_layers(activation))
        self.layer_15 = nn.Upsample(**upsample_cfg, align_corners=False if upsample_cfg["mode"] != "nearest" else None)
        self.layer_16 = Concat(1)
        self.layer_17 = C3(wm(512), wm(256), dm(3), False)

        self.layer_18 = Conv(wm(256), wm(256), kernel_size, 2, act=act_layers(activation))
        self.layer_19 = Concat(1)
        self.layer_20 = C3(wm(512), wm(512), dm(3), False)

        self.layer_21 = Conv(wm(512), wm(512), kernel_size, 2, act=act_layers(activation))
        self.layer_22 = Concat(1)
        self.layer_23 = C3(wm(1024), wm(1024), dm(3), False)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, inputs):
        x_10 = self.layer_10(inputs[-1])
        x = self.layer_11(x_10)
        x = self.layer_12([x, inputs[-2]])
        x = self.layer_13(x)

        x_14 = self.layer_14(x)
        x = self.layer_15(x_14)
        x = self.layer_16([x, inputs[-3]])
        x_17 = self.layer_17(x)

        x = self.layer_18(x_17)
        x = self.layer_19([x, x_14])
        x_20 = self.layer_20(x)

        x = self.layer_21(x_20)
        x = self.layer_22([x, x_10])
        out = [x_17, x_20, self.layer_23(x)]
        return out


if __name__ == '__main__':
    model = YoloPan(
        arch="s",
        kernel_size=3,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        activation="SiLU",
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
