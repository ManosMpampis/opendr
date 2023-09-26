# Modifications Copyright 2021 - present, OpenDR European Project
#
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

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.batch_process import divisible_padding
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.transform import Pipeline
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model


class Predictor(nn.Module):
    def __init__(self, cfg, model, device="cuda", conf_thresh=0.35, iou_thresh=0.6, nms_max_num=100, hf=False, dynamic=False):
        super(Predictor, self).__init__()
        self.cfg = cfg
        self.device = device
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.nms_max_num = nms_max_num
        self.hf = hf
        self.fuse = self.cfg.model.arch.fuse
        self.ch_l = self.cfg.model.arch.ch_l
        self.dynamic = dynamic
        self.traced_model = None
        if self.cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.backbone.repvgg\
                import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)

        for para in model.parameters():
            para.requires_grad = False

        if self.fuse:
            model.fuse()
        if self.ch_l:
            model = model.to(memory_format=torch.channels_last)
        if self.hf:
            model = model.half()
        model.set_dynamic(self.dynamic)

        self.model = model.to(device).eval()

        self.pipeline = Pipeline(self.cfg.data.val.pipeline, self.cfg.data.val.keep_ratio)

    def trace_model(self, dummy_input):
        self.traced_model = torch.jit.trace(self, dummy_input)
        return True

    def script_model(self, img, height, width, warp_matrix):
        preds = self.traced_model(img, height, width, warp_matrix)
        scripted_model = self.postprocessing(preds, img, height, width, warp_matrix)
        return scripted_model

    def forward(self, img, height=torch.tensor(0), width=torch.tensor(0), warp_matrix=torch.tensor(0)):
        if torch.jit.is_scripting():
            return self.script_model(img, height, width, warp_matrix)
        # In tracing (Jit and Onnx optimizations) we must first run the pipeline before the graf,
        # cv2 is needed, and it is installed with abi cxx11 but torch is in cxx<11
        return self.model.inference(img)

    def preprocessing(self, img, bench=False):
        if bench:
            try:
                input_size = self.cfg.data.bench_test.input_size
            except AttributeError as e:
                print(f"{e}, val input will be used")
                input_size = self.cfg.data.val.input_size
        else:
            input_size = self.cfg.data.val.input_size
        height, width = img.shape[:2]
        meta = dict(id=0, height=height, width=width, raw_img=img, img=img)
        meta = self.pipeline(None, meta, input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)

        meta["img"] = meta["img"].half() if self.hf else meta["img"]
        meta["img"] = divisible_padding(
            meta["img"],
            divisible=torch.tensor(32, device=self.device, dtype=torch.half if self.hf else torch.float)
        )

        # meta["img"] = meta["img"].to(torch.uint8)
        _input = meta["img"]
        _input = _input.to(memory_format=torch.channels_last) if self.ch_l else _input
        _height = torch.as_tensor(height, device=self.device)
        _width = torch.as_tensor(width, device=self.device)
        _warp_matrix = torch.from_numpy(meta["warp_matrix"]).to(self.device)

        return _input, _height, _width, _warp_matrix

    def postprocessing(self, preds, input, height, width, warp_matrix):
        meta = dict(height=height.unsqueeze(0), width=width.unsqueeze(0), id=torch.zeros(1, 1), warp_matrix=warp_matrix.unsqueeze(0), img=input.unsqueeze(0))
        res = self.model.head.post_process(preds, meta, conf_thresh=self.conf_threshold, iou_thresh=self.iou_threshold,
                                           nms_max_num=self.nms_max_num)
        return res


class Postprocessor(nn.Module):
    def __init__(self, cfg, model, device="cuda", conf_thresh=0.35, iou_thresh=0.6, nms_max_num=100, hf=True):
        super(Postprocessor, self).__init__()
        self.cfg = cfg
        self.device = device
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.nms_max_num = nms_max_num
        self.hf = hf

        self.model = model.to(device).eval()

    def forward(self, preds, input, height, width, warp_matrix):
        meta = dict(height=height.unsqueeze(0), width=width.unsqueeze(0), id=torch.zeros(1, 1), warp_matrix=warp_matrix.unsqueeze(0), img=input.unsqueeze(0))
        if self.hf:
            meta["img"] = meta["img"].half()
            preds = preds.half()
        res = self.model.head.post_process(preds, meta, conf_thresh=self.conf_threshold, iou_thresh=self.iou_threshold,
                                           nms_max_num=self.nms_max_num)
        return res
