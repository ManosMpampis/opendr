# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yoloHead.py --cfg yolov5s.yaml
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Dict
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss import YoloLoss


class Wm:
    def __init__(self, width_multiplier=0.5):
        self.width_multiplier = width_multiplier

    def __call__(self, width):
        return int(width * self.width_multiplier)


class YoloHead(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    YoloArch = dict(
        x=dict(width_multiple=1.25, depth_multiple=1.33),
        l=dict(width_multiple=1.0, depth_multiple=1.0),
        m=dict(width_multiple=0.75, depth_multiple=0.67),
        s=dict(width_multiple=0.5, depth_multiple=0.33),
        n=dict(width_multiple=0.25, depth_multiple=0.33),
    )

    def __init__(self, num_classes, loss, arch, anchors, inplace=True, imgsz=640):  # detection layer
        super().__init__()
        width_multiple = self.YoloArch[arch]["width_multiple"]
        depth_multiple = self.YoloArch[arch]["depth_multiple"]
        wm = Wm(width_multiple)
        input_channels = [wm(256), wm(512), wm(1024)]
        if isinstance(anchors, int):
            anchors = [list(range(anchors * 2))] * len(input_channels)

        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in input_channels)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

        self.loss_cfg = loss

        hyperparameters = {
            "cls_pw": self.loss_cfg.cls_pw,
            "obj_pw": self.loss_cfg.obj_pw,
            "label_smoothing": self.loss_cfg.get('label_smoothing', 0.0),
            "fl_gamma": self.loss_cfg.fl_gamma,
            "box": self.loss_cfg.box * (3/self.nl),
            "obj": self.loss_cfg.obj * ((imgsz / 640) ** 2 * 3 / self.nl),
            "cls": self.loss_cfg.cls * ((self.nc / 80) * (3 / self.nl)),
        }
        autobalance = ...
        self.loss_fn = YoloLoss("cuda", hyperparameters, self, autobalance)
        self.input_smple = ...
        self.build_strides_anchors()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    def graph_forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def build_strides_anchors(self):
        s = 256  # 2x min stride
        self.stride = torch.tensor([s / x.shape[-2] for x in self(torch.zeros(self.input_smple))])  # forward
        self.check_anchor_order()
        self.anchors /= self.stride.view(-1, 1, 1)
        self.stride = self.stride
        self._initialize_biases()  # only run once

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + self.nc] += math.log(0.6 / (self.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def check_anchor_order(self):
        # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
        a = self.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
        da = a[-1] - a[0]  # delta a
        ds = self.stride[-1] - self.stride[0]  # delta s
        if da and (da.sign() != ds.sign()):  # same order
            print(f'AutoAnchor: Reversing anchor order')
            self.anchors[:] = self.anchors.flip(0)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if torch.jit.is_scripting() or not torch.__version__[:4] == "1.13":
            yv, xv = torch.meshgrid(y, x)
        else:
            yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    def loss(self, preds, gt_meta):
        #TODO: FIND LOSS AND ASSIGNER

        # loss, loss_items = compute_loss(pred, targets.to(device))

        loss, loss_items = self.loss_fn(preds, )

        loss = ...
        loss_states = ...
        return loss, loss_states