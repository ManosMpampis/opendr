from __future__ import absolute_import, division, print_function

import torch.jit
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers


class VggSmall(nn.Module):

    def __init__(
        self,
        out_stages=(1, 2, 3, 4),
        stages_inplanes=(3, 8, 8, 6),
        stages_outplanes=(8, 8, 6, 6),
        stages_strides=(2, 1, 1, 1),
        maxpool_after=(0, 1, 0, 0),
        maxpool_stride=1,
        activation="ReLU",
        pretrain=True
    ):
        super(VggSmall, self).__init__()
        assert set(out_stages).issubset((1, 2, 3, 4, 5))
        self.activation = activation
        self.names = [f"{i}" for i in range(len(stages_inplanes))]
        if len(stages_outplanes) != len(stages_inplanes):
            raise KeyError(f"Not enough features for block it should be {len(stages_inplanes)} and are {len(stages_outplanes)}")

        self.out_stages = out_stages
        self.act = act_layers(self.activation)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=maxpool_stride, padding=1)
        # self.maxpool_stages = maxpool_after
        # self.conv1_1 = nn.Conv2d(3, stages_outplanes[0], kernel_size=3, stride=stages_strides[0], padding=1, bias=False)
        self.layers_list = nn.ModuleList()
        for block in range(len(stages_inplanes)):
            self.layers_list.append(self._make_layer(self.names[block], stages_inplanes[block], stages_outplanes[block], stages_strides[block], maxpool_after[block]))

        # self.conv1_2 = nn.Conv2d(stages_outplanes[0], stages_outplanes[1], kernel_size=3, stride=stages_strides[1], padding=1, bias=False)
        #
        # self.conv2_1 = nn.Conv2d(stages_outplanes[1], stages_outplanes[2], kernel_size=3, stride=stages_strides[2], padding=1, bias=False)
        # self.conv2_2 = nn.Conv2d(stages_outplanes[2], stages_outplanes[3], kernel_size=3, stride=stages_strides[3], padding=1, bias=False)
        self.init_weights(pretrain=pretrain)

    def _make_layer(self, name, inplanes, outplanes, stride, maxpool):
        layer = nn.Sequential()
        layer.add_module(f"conv{name}", nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, stride=stride, bias=False))
        layer.add_module(f"batchNorm{name}", nn.BatchNorm2d(outplanes))
        if maxpool == 1:
            layer.add_module(f"maxpool{name}", self.maxpool)

        return layer

    @torch.jit.unused
    def forward(self, x):
        output = []
        counter = 0
        for layer in self.layers_list:
            x = layer(x)
            x = self.act(x)
            counter += 1
            if counter in self.out_stages:
                output.append(x)

        return output

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
                else:
                    nonlinearity = "relu"
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
