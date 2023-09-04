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

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.custom_csp import CustomCspNet
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.efficientnet_lite import EfficientNetLite
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.ghostnet import GhostNet
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.mobilenetv2 import MobileNetV2
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.repvgg import RepVGG
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.resnet import ResNet
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.shufflenetv2 import ShuffleNetV2

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.simpleConv import SimpleCnn
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.simpleConv_2 import SimpleCnn_2
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.simpleConv_3 import VggBackbone

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_ghostnet import GhostNetSmall
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_mobilenetv2 import MobileNetV2Small
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_repvgg import RepVGGSmall
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_resnet import ResNetSmall
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_shufflenetv2 import ShuffleNetV2Small
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.small_cnn import VggSmall


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ResNet":
        return ResNet(**backbone_cfg)
    elif name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    elif name == "MobileNetV2":
        return MobileNetV2(**backbone_cfg)
    elif name == "EfficientNetLite":
        return EfficientNetLite(**backbone_cfg)
    elif name == "CustomCspNet":
        return CustomCspNet(**backbone_cfg)
    elif name == "RepVGG":
        return RepVGG(**backbone_cfg)
    elif name == "GhostNetSmall":
        return GhostNetSmall(**backbone_cfg)
    elif name == "MobileNetV2Small":
        return MobileNetV2Small(**backbone_cfg)
    elif name == "RepVGGSmall":
        return RepVGGSmall(**backbone_cfg)
    elif name == "ResNetSmall":
        return ResNetSmall(**backbone_cfg)
    elif name == "ShuffleNetV2Small":
        return ShuffleNetV2Small(**backbone_cfg)
    elif name == "SmallVgg":
        return VggSmall(**backbone_cfg)
    elif name == "SimpleCnn":
        return SimpleCnn(**backbone_cfg)
    elif name == "SimpleCnn_2":
        return SimpleCnn_2(**backbone_cfg)
    elif name == "SimpleCnn_3":
        return VggBackbone(**backbone_cfg)
    else:
        raise NotImplementedError
