"""
ConvModule refers from MMDetection
RepVGGConvModule refers from RepVGG: Making VGG-style ConvNets Great Again
"""
import warnings
from typing import List
import numpy as np
import torch
import torch.nn as nn

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.init_weights\
    import constant_init, kaiming_init
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.norm import build_norm_layer


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_modules(m):
    if isinstance(m, Conv) and hasattr(m, 'bn'):
        m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
        delattr(m, 'bn')  # remove batchnorm
        m.forward = m.forward_fuse  # update forward
    elif isinstance(m, ConvQuant) and (hasattr(m, "bn") and hasattr(m, "act")):
        if isinstance(m.bn, nn.BatchNorm2d) and isinstance(m.act, nn.ReLU):
            torch.quantization.fuse_modules(m, [["conv", "bn", "act"]], inplace=True)
        elif isinstance(m.act, nn.ReLU) and not isinstance(m.bn, nn.BatchNorm2d):
            torch.quantization.fuse_modules(m, [["conv", "act"]], inplace=True)
        elif isinstance(m.bn, nn.BatchNorm2d) and not isinstance(m.act, nn.ReLU):
            torch.quantization.fuse_modules(m, [["conv", "bn"]], inplace=True)
        delattr(m, 'bn')
        delattr(m, 'act')
        m.forward = m.forward_fuse
    elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
        for m_in_list in m:
            fuse_modules(m_in_list)


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.init_weights()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv) or isinstance(m, ConvQuant):
                nonlinearity = "leaky_relu" if isinstance(self.act, nn.LeakyReLU) else "relu"
                a = m.act.negative_slope if isinstance(self.act, nn.LeakyReLU) else 0
                nn.init.kaiming_normal_(
                    m.conv.weight, mode="fan_out", nonlinearity=nonlinearity, a=a
                )
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()


class ConvQuant(Conv):
    # Standard convolution include Quantization with args(ch_in, ch_out, kernel, stride, padding, groups, dilation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, pool=True):
        super(ConvQuant, self).__init__(c1=c1, c2=c2, k=k, s=s, p=p, g=g, d=d, act=act)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x

    def forward_fuse(self, x):
        x = self.quant(x)
        x = super().forward_fuse(x)
        x = self.dequant(x)
        return x

    def temp(self):
        model = ConvQuant()
        print(model)

        def prepare_save(model, fused):
            from torch.utils.mobile_optimizer import optimize_for_mobile
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)
            torchscript_model = torch.jit.script(model)
            torchscript_model_optimized = optimize_for_mobile(torchscript_model)
            torch.jit.save(torchscript_model_optimized, "model.pt" if not fused else "model_fused.pt")

        prepare_save(model, False)

        model = ConvQuant()
        model_fused = torch.quantization.fuse_modules(model, [["conv", "bn", "relu"]], inplace=False)
        print(model_fused)

        prepare_save(model_fused, True)


class ConvPool(Conv):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, pool=True):
        super(ConvPool, self).__init__(c1=c1, c2=c2, k=k, s=s, p=p, g=g, d=d, act=act, pool=pool)
        self.pool = self.default_pool if pool is True else pool if isinstance(pool, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.pool(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.act(self.pool(self.conv(x)))


class ConvPoolQuant(ConvPool):
    # Standard convolution include Quantization with args(ch_in, ch_out, kernel, stride, padding, groups, dilation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, pool=True):
        super(ConvPoolQuant, self).__init__(c1=c1, c2=c2, k=k, s=s, p=p, g=g, d=d, act=act)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x

    def forward_fuse(self, x):
        x = self.quant(x)
        x = super().forward_fuse(x)
        x = self.dequant(x)
        return x


class DWConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, act=True, g=None, pool=True):
        super().__init__()
        self.depthwise = Conv(c1, c1, k, s=s, p=p, d=d, g=c1, act=act)
        self.pointwise = Conv(c1, c2, k=1, s=1, p=0, act=act)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DWConvQuant(DWConv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, act=True, g=None, pool=True):
        super(DWConvQuant, self).__init__(c1=c1, c2=c2, k=k, s=s, p=p, g=g, d=d, act=act)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x

    def forward_fuse(self, x):
        x = self.quant(x)
        x = super().forward_fuse(x)
        x = self.dequant(x)
        return x


class DWConvPool(nn.Module):
    # Standard convolution include Quantization with args(ch_in, ch_out, kernel, stride, padding, groups, dilation)
    default_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, pool=True):
        super(DWConvPool, self).__init__()
        self.pool = self.default_pool if pool is True else pool if isinstance(pool, nn.Module) else nn.Identity()
        self.depthwise = ConvPool(c1, c1, k, s=s, p=p, d=d, g=c1, act=act, pool=pool)
        self.pointwise = ConvPool(c1, c2, k=1, s=1, p=0, act=act, pool=pool)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DWConvPoolQuant(DWConvPool):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, d=1, act=True, g=None, pool=True):
        super(DWConvPoolQuant, self).__init__(c1=c1, c2=c2, k=k, s=s, p=p, g=g, d=d, act=act, pool=pool)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)
        return x

    def forward_fuse(self, x):
        x = self.quant(x)
        x = super().forward_fuse(x)
        x = self.dequant(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



class MultiOutput(nn.Module):
    # Output a list of tensors
    def __init__(self):
        super(MultiOutput, self).__init__()

    def forward(self, x):
        outs = [out for out in x]
        return outs


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.s = start_dim
        self.e = end_dim

    def forward(self, x):
        return torch.flatten(x, start_dim=self.s, end_dim=self.e)


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        activation="ReLU",
        inplace=True,
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv2d(  #
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @torch.jit.unused
    @property
    def norm(self):
        if self.norm_name is not None:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
            a = self.act.negative_slope
        else:
            nonlinearity = "relu"
            a = 0
        kaiming_init(self.conv, nonlinearity=nonlinearity, a=a)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    @torch.jit.unused
    def forward(self, x, norm: bool = True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and (norm is not None) and (self.with_norm is not None) and (self.norm is not None):
                x = self.norm(x)
            elif layer == "act" and (self.activation is not None):
                x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        norm_cfg=dict(type="BN"),
        activation="ReLU",
        inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),  # ("depthwise", "act", "pointwise", "act"), #("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 6
        assert set(order) == {
            "depthwise",
            "dwnorm",
            "act",
            "pointwise",
            "pwnorm",
            "act",
        }

        self.with_norm = norm_cfg['type'] != "None"
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.dwnorm = nn.Identity()
            self.pwnorm = nn.Identity()
        # build activation layer
        self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
            a = self.act.negative_slope
        else:
            nonlinearity = "relu"
            a = 0
        kaiming_init(self.depthwise, nonlinearity=nonlinearity, a=a)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity, a=a)
        if self.with_norm:
            constant_init(self.dwnorm, 1, bias=0)
            constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x):
        for layer_name in self.order:
            if layer_name == "depthwise":
                x = self.depthwise(x)
            elif layer_name == "pointwise":
                x = self.pointwise(x)
            elif layer_name == "dwnorm" and (self.dwnorm is not None):
                x = self.dwnorm(x)
            elif layer_name == "pwnorm" and (self.pwnorm is not None):
                x = self.pwnorm(x)
            elif layer_name == "act" and (self.activation is not None):
                x = self.act(x)
        return x


class RepVGGConvModule(nn.Module):
    """
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
    https://github.com/DingXiaoH/RepVGG
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="ReLU",
        padding_mode="zeros",
        deploy=False,
        **kwargs
    ):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=padding_11,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you
    #   do to the other models.  May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )
