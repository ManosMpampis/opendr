# Copyright 2020-2023 OpenDR European Project
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

import argparse
from opendr.perception.object_detection_2d import NanodetLearner
from opendr.engine.data import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="simple_big_ch")
    parser.add_argument("--optimize-jit", help="", action="store_false")
    parser.add_argument("--optimize-trt", help="", action="store_false")
    parser.add_argument("--download", help="", action="store_true")
    parser.add_argument("--hf", help="", action="store_false")
    parser.add_argument("--fuse", help="", action="store_false")
    parser.add_argument("--ch_l", help="", action="store_true")
    parser.add_argument("--dynamic", help="", action="store_true")
    parser.add_argument("--repetitions", help="Determines the amount of repetitions to run", type=int, default=100)
    parser.add_argument("--warmup", help="Determines the amount of warmup runs", type=int, default=10)
    parser.add_argument("--conf-threshold", help="Determines the confident threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", help="Determines the iou threshold", type=float, default=0.6)
    parser.add_argument("--nms", help="Determines the max amount of bboxes the nms will output", type=int, default=30)
    args = parser.parse_args()

    dataset_metadata = {
        "data_root": "/media/manos/hdd/allea_datasets/weedDataset",
        "classes": ["poaceae", "brassicaceae"],
        "dataset_type": "weed",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]
    from opendr.perception.object_detection_2d.datasets import XMLBasedDataset
    dataset = XMLBasedDataset(root=f'{data_root}/test', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)



    nanodet = NanodetLearner(model_to_use=args.model, device=args.device, model_log_name=f"{args.model}")

    if args.download:
        nanodet.download("./predefined_examples", mode="pretrained", verbose=False)
        nanodet.load("./predefined_examples/nanodet_{}".format(args.model), verbose=False)
    nanodet.download("./predefined_examples", mode="images", verbose=False)

    if args.optimize_jit:
        nanodet.optimize(f"./jit/nanodet_{nanodet.cfg.check_point_name}", optimization="jit", hf=args.hf, verbose=False, new_load=False)
    if args.optimize_trt:
        nanodet.optimize(f"./trt/nanodet_{nanodet.cfg.check_point_name}", optimization="trt", hf=args.hf, verbose=False, new_load=False, dynamic=args.dynamic, calib_dataset=dataset)

    nanodet.benchmark(repetitions=args.repetitions, warmup=args.warmup, conf_threshold=args.conf_threshold,
                      iou_threshold=args.iou_threshold, nms_max_num=args.nms, hf=args.hf, fuse=args.fuse, ch_l=args.ch_l)
