import copy

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.gfl_head import GFLHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_head import NanoDetHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.simple_conv_head import SimpleConvHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_simple_plus_head_1 import SimplifierNanoDetPlusHead_1
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_simple_plus_head_2 import SimplifierNanoDetPlusHead_2
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.simple_nanodet_plus import SimplifierNanoDetPlusHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    elif name == "SimplifierNanoDetPlusHead":
        return SimplifierNanoDetPlusHead(**head_cfg)
    elif name == "SimplifierNanoDetPlusHead_1":
        return SimplifierNanoDetPlusHead_1(**head_cfg)
    elif name == "SimplifierNanoDetPlusHead_2":
        return SimplifierNanoDetPlusHead_2(**head_cfg)
    else:
        raise NotImplementedError
