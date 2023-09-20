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

import copy

import torch

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head import build_head
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch.one_stage_detector import OneStageDetector


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


class NanoDetPlus(OneStageDetector):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
        **kwargs
    ):
        super(NanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = [
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = [
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]
        head_out = self.head(fpn_feat)
        aux_head_out = self.aux_head(dual_fpn_feat)
        loss, loss_states = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        return head_out, loss, loss_states


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    cfg = _load_hparam("yoloLike")
    model_cfg = copy.deepcopy(cfg.model)
    name = model_cfg.arch.pop("name")
    assert name == "NanoDetPlus"

    model = NanoDetPlus(
        **model_cfg.arch
    ).eval().to("cpu")

    imgsz = (1, 3, 1088, 1920)
    __dumy_input = torch.empty(*imgsz, dtype=torch.float, device="cpu")

    writer = SummaryWriter(f'./models/full_model')
    writer.add_graph(model.eval(), __dumy_input)
    writer.close()

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
