from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import Conv, DWConv, ConvQuant, DWConvQuant
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers

# [
#     [3, 8, 3, 2, 1, conv],
#     [8, 8, 3, 2, 1, conv],
#     [8, 8, 3, 1, 1, conv],
#     [8, 8, 3, 2, 1, conv],
#     [[-1, -2], concat]
#     []
# ]
class SimpleCnn(nn.Module):

    def __init__(
        self,
        activation="ReLU",
        use_depthwise=False,
        quant=False,
        pretrain=False
    ):
        super(SimpleCnn, self).__init__()
        self.activation = activation
        if use_depthwise:
            conv = DWConvQuant if quant else DWConv
        else:
            conv = ConvQuant if quant else Conv

        self.conv_0 = conv(3, 8, k=3, s=2, p=1, act=act_layers(self.activation))
        self.conv_1 = conv(8, 8, k=3, s=2, p=1, act=act_layers(self.activation))
        self.conv_2 = conv(8, 8, k=3, s=1, p=1, act=act_layers(self.activation))
        self.conv_3 = conv(8, 8, k=3, s=2, p=1, act=act_layers(self.activation))

        self.init_weights(pretrain=pretrain)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x, self.conv_3(x)

    def init_weights(self, pretrain=True):
        if pretrain:
            url = "https://download.pytorch.org/models/vgg16-397923af.pth"
            pretrained_state_dict = model_zoo.load_url(url)
            print("=> loading pretrained model {}".format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            for m in self.modules():
                if self.activation == "LeakyReLU":
                    nonlinearity = "leaky_relu"
                    a = self.act.negative_slope
                else:
                    nonlinearity = "relu"
                    a = 0
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity, a=0
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


if __name__ == '__main__':
    model = SimpleCnn(quant=True).eval()
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
