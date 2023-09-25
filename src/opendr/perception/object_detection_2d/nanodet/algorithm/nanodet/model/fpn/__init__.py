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

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.fpn import FPN
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.ghost_pan import GhostPAN
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.pan import PAN
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.tan import TAN
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.simpleGost_1 import SimpleGPAN_1
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.simpleGost_2 import SimpleGPAN_2
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.simpleGost import SimpleGPAN
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.fpn.yoloPan import YoloPan


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop("name")
    if name == "FPN":
        return FPN(**fpn_cfg)
    elif name == "PAN":
        return PAN(**fpn_cfg)
    elif name == "TAN":
        return TAN(**fpn_cfg)
    elif name == "GhostPAN":
        return GhostPAN(**fpn_cfg)
    elif name == "SimpleGPAN":
        return SimpleGPAN(**fpn_cfg)
    elif name == "SimpleGPAN_1":
        return SimpleGPAN_1(**fpn_cfg)
    elif name == "SimpleGPAN_2":
        return SimpleGPAN_2(**fpn_cfg)
    elif name == "Yolo":
        return YoloPan(**fpn_cfg)
    else:
        raise NotImplementedError
