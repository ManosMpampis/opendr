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

from typing import Dict
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone import build_backbone
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn import build_fpn
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head import build_head


def _load_hparam(model: str):
    from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (load_config, cfg)
    import os
    from pathlib import Path

    """ Load hyperparameters for nanodet models and training configuration

    :parameter model: The name of the model of which we want to load the config file
    :type model: str
    :return: config with hyperparameters
    :rtype: dict
    """
    # assert (
    #         model in _MODEL_NAMES
    # ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
    full_path = list()
    path = Path(__file__).parent.parent.parent.parent.parent / "algorithm" / "config"
    wanted_file = "nanodet_{}.yml".format(model)
    for root, dir, files in os.walk(path):
        if wanted_file in files:
            full_path.append(os.path.join(root, wanted_file))
    assert (len(full_path) == 1), f"You must have only one nanodet_{model}.yaml file in your config folder"
    load_config(cfg, full_path[0])
    return cfg


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        self.epoch = 0

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head(x)
        return x

    def inference(self, img):#meta: Dict[str, torch.Tensor]):
        with torch.no_grad():
            preds = self(img)
        return preds

    def forward_train(self, gt_meta):
        preds = self(gt_meta["img"])
        loss, loss_states = self.head.loss(preds, gt_meta)

        return preds, loss, loss_states

    def fuse(self):
        if hasattr(self.backbone, "fuse"):
            self.backbone.fuse()
        else:
            print(f"Backbone {self.backbone} does not have fuse function, run regular instead")
        if hasattr(self, "fpn"):
            if hasattr(self.fpn, "fuse"):
                self.fpn.fuse()
            else:
                print(f"FPN {self.fpn} does not have fuse function, run regular instead")
        if hasattr(self, "head"):
            if hasattr(self.head, "fuse"):
                self.backbone.fuse()
            else:
                print(f"Head {self.head} does not have fuse function, run regular instead")

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == '__main__':
    import copy
    cfg = _load_hparam("test")
    model_cfg = copy.deepcopy(cfg.model)
    name = model_cfg.arch.pop("name")
    assert name == "OneStageDetector"


    model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
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