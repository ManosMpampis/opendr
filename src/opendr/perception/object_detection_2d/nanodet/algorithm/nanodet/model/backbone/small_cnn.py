from __future__ import absolute_import, division, print_function

import torch
import torch.jit
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import (Conv, ConvPool,
                                                                                               ConvQuant, ConvPoolQuant)
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class MultiOutput(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self):
        super(MultiOutput, self).__init__()

    def forward(self, x):
        outs = [out for out in x]
        return outs


class VggSmall(nn.Module):

    def __init__(
        self,
        out_stages=(0, 1, 2, 3),
        stages_inplanes=(3, 8, 8, 6),
        stages_outplanes=(8, 8, 6, 6),
        stages_strides=(2, 1, 1, 1),
        stages_kernels=(3, 3, 3, 3),
        stages_padding=(1, 1, 1, 1),
        maxpool_after=(0, 1, 0, 0),
        maxpool_stride=1,
        activation="ReLU",
        quant=False,
        pretrain=True
    ):
        super(VggSmall, self).__init__()
        self.num_layers = len(stages_inplanes)
        for layers_args in [stages_outplanes, stages_kernels, stages_strides, stages_padding, maxpool_after]:
            if len(layers_args) != self.num_layers :
                raise KeyError(
                    f"Not all convolution args have the same length")
        assert set(out_stages).issubset(range(len(stages_inplanes)))

        act = act_layers(activation)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=maxpool_stride, padding=1)
        Convs = (ConvQuant, ConvPoolQuant) if quant else (Conv, ConvPool)
        self.out_stages = out_stages

        self.backbone = nn.ModuleList()
        for idx, (inch, ouch, k, s, p, mp) in enumerate(zip(stages_inplanes, stages_outplanes, stages_kernels, stages_strides, stages_padding, maxpool_after)):
            conv = Convs[1] if mp != 0 else Convs[0]
            self.backbone.append(conv(inch, ouch, k=k, s=s, p=p, act=act, pool=maxpool))
            self.backbone[-1].i = idx
            self.backbone[-1].f = -1

        self.backbone.append(MultiOutput())
        self.backbone[-1].i = -1
        self.backbone[-1].f = self.out_stages

        self.backbone = nn.Sequential(*self.backbone)
        self.init_weights(pretrain=pretrain)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    @torch.jit.unused
    def forward(self, x):
        y = []
        for layer in self.backbone:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x if layer.i in self.out_stages else None)
        return x

    def init_weights(self, pretrain=None):
        if pretrain:
            import re
            url = model_urls[pretrain]  # "https://download.pytorch.org/models/vgg16-397923af.pth"
            pretrained_state_dict = model_zoo.load_url(url)
            print("=> loading pretrained model {}".format(url))

            pattern = r'\.(.*?)\.'
            pre_idxs = []
            for key, value in pretrained_state_dict.items():
                if key.startswith("features"):
                    pre_idxs.append(int(re.findall(pattern, key)[0]))
            pre_idxs = set(pre_idxs)
            new_idxs = []
            for i, pre_idx in enumerate(pre_idxs):
                if i % 2:
                    continue
                new_idxs.append(pre_idx)
            # pre_idxs = [re.findall(pattern, number[0]) for number in pretrained_state_dict.items()]
            custom_mapping = {}
            for idx, pre_idx in enumerate(new_idxs):
                if idx in range(self.num_layers):
                    custom_mapping[f'backbone.{idx}.conv.weight'] = f'features.{pre_idx}.weight'
                    custom_mapping[f'backbone.{idx}.bn.weight'] = f'features.{pre_idx+1}.weight'
                    custom_mapping[f'backbone.{idx}.bn.bias'] = f'features.{pre_idx+1}.bias'
                    custom_mapping[f'backbone.{idx}.bn.running_mean'] = f'features.{pre_idx+1}.running_mean'
                    custom_mapping[f'backbone.{idx}.bn.running_var'] = f'features.{pre_idx+1}.running_var'

            updated_state_dict = {}
            for key, value in pretrained_state_dict.items():
                for custom_key, pretrained_key in custom_mapping.items():
                    if key.startswith(pretrained_key):
                        # key.endswith()
                        updated_key = key.replace(pretrained_key, custom_key)
                        ws = self.state_dict()[updated_key].size()
                        print(ws[0])
                        if len(ws) == 4:
                            updated_state_dict[updated_key] = value[:ws[0], :ws[1], :ws[2], :ws[3]]
                        else:
                            updated_state_dict[updated_key] = value[:ws[0]]

            temp = self.load_state_dict(updated_state_dict, strict=False)
            print(temp)
            print("")


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    model = VggSmall(
        out_stages=(2, 3),
        stages_inplanes=(3, 8, 8, 8),
        stages_outplanes=(8, 8, 8, 8),
        stages_strides=(2, 2, 1, 2),
        maxpool_after=(0, 0, 1, 0),
        maxpool_stride=1,
        activation="ReLU",
        pretrain=True
    )
    for m in model.modules():
        print(m)
    print("=========================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model = model.to("cpu")
    imgsz = (1, 3, 1080, 1920)
    hf = False
    __dumy_input = torch.empty(*imgsz, dtype=torch.half if hf else torch.float, device="cpu")
    writer = SummaryWriter(f'./models/vggSmall')
    writer.add_graph(model.eval(), __dumy_input)
    writer.close()