from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import Conv, DWConv
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers


class VggBackbone(nn.Module):

    def __init__(
        self,
        activation="ReLU",
        use_depthwise=False,
    ):
        super(VggBackbone, self).__init__()
        self.activation = activation

        conv = DWConv if use_depthwise else Conv

        self.conv_0 = conv(3, 8, k=3, s=2, p=1, act=act_layers(self.activation))
        self.conv_1 = conv(8, 8, k=3, s=1, p=1, act=act_layers(self.activation))
        self.conv_2 = conv(8, 16, k=3, s=2, p=1, act=act_layers(self.activation))
        self.conv_3 = conv(16, 16, k=3, s=1, p=1, act=act_layers(self.activation))
        self.conv_4 = conv(16, 32, k=3, s=2, p=1, act=act_layers(self.activation))
        self.conv_5 = conv(32, 32, k=3, s=1, p=1, act=act_layers(self.activation))
        self.conv_6 = conv(32, 64, k=3, s=2, p=1, act=act_layers(self.activation))

        # self.init_weights(pretrain=pretrain)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x, self.conv_6(x)



if __name__ == '__main__':
    model = VggBackbone(quant=True).eval()
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
