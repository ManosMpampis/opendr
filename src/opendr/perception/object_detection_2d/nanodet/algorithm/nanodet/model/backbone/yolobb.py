from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import Conv, SPPF, C3
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers

# [from, number, module, args]
# [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
#  [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#  [-1, 3, C3, [128]],
#  [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#  [-1, 6, C3, [256]],
#  [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#  [-1, 9, C3, [512]],
#  [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#  [-1, 3, C3, [1024]],
#  [-1, 1, SPPF, [1024, 5]],  # 9
#  ]


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


class Yolo(nn.Module):
    YoloArch = dict(
        x=dict(width_multiple=1.25, depth_multiple=1.33),
        l=dict(width_multiple=1.0, depth_multiple=1.0),
        m=dict(width_multiple=0.75, depth_multiple=0.67),
        s=dict(width_multiple=0.5, depth_multiple=0.33),
        n=dict(width_multiple=0.25, depth_multiple=0.33),
    )
    def __init__(
        self,
        arch="s",
        activation="SiLU",
    ):
        super(Yolo, self).__init__()
        self.activation = activation
        width_multiple = self.YoloArch[arch]["width_multiple"]
        depth_multiple = self.YoloArch[arch]["depth_multiple"]
        wm = Wm(width_multiple)
        dm = Dm(depth_multiple)

        self.layer_0 = Conv( 3, wm(64), k=6, s=2, p=2, act=act_layers(self.activation))
        self.layer_1 = Conv(wm(64), wm(128), k=3, s=2, p=None, act=act_layers(self.activation))
        self.layer_2 = C3(wm(128), wm(128), dm(3))
        self.layer_3 = Conv(wm(128), wm(256), k=3, s=2, p=None, act=act_layers(self.activation))
        self.layer_4 = C3(wm(256), wm(256), dm(6))
        self.layer_5 = Conv(wm(256), wm(512), k=3, s=2, p=None, act=act_layers(self.activation))
        self.layer_6 = C3(wm(512), wm(512), dm(6))
        self.layer_7 = Conv(wm(512), wm(1024), k=3, s=2, p=None, act=act_layers(self.activation))
        self.layer_8 = C3(wm(1024), wm(1024), dm(6))
        self.layer_9 = SPPF(wm(1024), wm(1024), 5)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        k = self.layer_4(x)
        x = self.layer_5(k)
        y = self.layer_6(x)
        x = self.layer_7(y)
        x = self.layer_8(x)
        return k, y, self.conv_9(x)


if __name__ == '__main__':
    model = Yolo().eval()
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
