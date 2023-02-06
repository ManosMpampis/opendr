from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="ReLU"):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act = act_layers(activation)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="ReLU"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = act_layers(activation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ResNetSmall(nn.Module):
    resnet_spec = {
        4: (BasicBlock, [1, 1]),
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
    }

    def __init__(
        self,
        depth,
        first_conv_features=2,
        out_stages=(1, 2, 3, 4),
        stages_features=(2, 2, 3, 3),
        stages_strides=(2, 2, 3, 3),
        activation="ReLU",
        pretrain=True
    ):
        super(ResNetSmall, self).__init__()
        if depth not in self.resnet_spec:
            raise KeyError("invalid resnet depth {}".format(depth))
        assert set(out_stages).issubset((1, 2, 3, 4))
        self.activation = activation
        block, layers = self.resnet_spec[depth]
        if len(stages_features) != len(layers):
            raise KeyError(f"Not enough features for block it should be {len(layers)} and are {len(stages_features)}")
        self.depth = depth
        self.inplanes = first_conv_features
        self.out_stages = out_stages

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(2)
        self.act = act_layers(self.activation)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers_list = nn.ModuleList()
        for blocks, block_feature, block_strides in zip(layers, stages_features, stages_strides):
            self.layers_list.append(self._make_layer(block, block_feature, blocks, stride=block_strides))
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.init_weights(pretrain=pretrain)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, activation=self.activation)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation))

        return nn.Sequential(*layers)

    @torch.jit.unused
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        output = []
        counter = 0
        for res_layer in self.layers_list:
            x = res_layer(x)
            counter += 1
            if counter in self.out_stages:
                output.append(x)
        # for i in range(1, 5):
        #     res_layer = getattr(self, "layer{}".format(i))
        #     x = res_layer(x)
        #     if i in self.out_stages:
        #         output.append(x)

        return output

    def init_weights(self, pretrain=True):
        if pretrain:
            url = model_urls["resnet{}".format(self.depth)]
            pretrained_state_dict = model_zoo.load_url(url)
            print("=> loading pretrained model {}".format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            for m in self.modules():
                if self.activation == "LeakyReLU":
                    nonlinearity = "leaky_relu"
                else:
                    nonlinearity = "relu"
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
