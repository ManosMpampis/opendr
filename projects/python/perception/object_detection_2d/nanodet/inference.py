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
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.perception.object_detection_2d.datasets import XMLBasedDataset
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="simple_big_ch")
    parser.add_argument("--optimize", help="If specified will determine the optimization to be used (onnx, jit)",
                        type=str, default="", choices=["", "onnx", "jit", "trt"])
    parser.add_argument("--conf-threshold", help="Determines the confident threshold", type=float, default=0.2)
    parser.add_argument("--iou-threshold", help="Determines the iou threshold", type=float, default=0.3)
    parser.add_argument("--nms", help="Determines the max amount of bboxes the nms will output", type=int, default=500)
    parser.add_argument("--show", help="do not show image", action="store_false")
    args = parser.parse_args()

    nanodet = NanodetLearner(model_to_use=args.model, device="cuda", model_log_name=args.model)
    save_path = f"./temp/{nanodet.cfg.check_point_name}/model_best/"
    nanodet.cfg.defrost()
    nanodet.cfg.check_point_name = "model_state_best"
    nanodet.cfg.freeze()
    nanodet.load(save_path, verbose=True)

    dataset_metadata = {
        "data_root": "/media/manos/hdd/allea_datasets/weedDataset",
        "classes": ["poaceae", "brassicaceae"],
        "dataset_type": "weed",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]
    save_images = ["91d0255701a5851e8b32.jpg", "f7b31632cfc4c1f481d5.jpg"]
    dataset = XMLBasedDataset(root=f'{data_root}/test', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)
    if args.optimize != "":
        nanodet.optimize(f"./{args.optimize}/nanodet_{args.model}", optimization=args.optimize, hf=False, new_load=False)

    printed_classes = nanodet.classes + [f"GR: {cls}" for cls in nanodet.classes]
    for idx, (img, annotation) in enumerate(dataset):
        boxes = nanodet.infer(input=img, conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold,
                              nms_max_num=args.nms, hf=False)
        # for box in annotation.boxes:
        #     box.name = box.name + len(printed_classes)//2
        #     boxes.add_box(box)
        image_to_print = draw_bounding_boxes(img.opencv(), boxes, class_names=printed_classes, show=False)
        if args.show:
            image_to_print = cv2.resize(image_to_print, dsize=None, fx=0.8, fy=0.8)
            cv2.imshow('detections', image_to_print)
            cv2.waitKey(0)
            if dataset.image_paths[idx] in save_images:
                cv2.imwrite(f"/home/manos/Desktop/del_images/Nanodet/{dataset.image_paths[idx]}", image_to_print)
            cv2.destroyAllWindows()
