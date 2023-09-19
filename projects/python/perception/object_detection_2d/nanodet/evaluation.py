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
from opendr.perception.object_detection_2d.datasets import XMLBasedDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="simple_big_ch")
    args = parser.parse_args()

    nanodet = NanodetLearner(model_to_use=args.model, device="cuda")

    save_path = f"./temp/{nanodet.cfg.check_point_name}/model_best/"
    nanodet.cfg.defrost()
    nanodet.cfg.check_point_name = "model_state_best"
    # nanodet.cfg.data.val.input_size = [2456, 2054]
    nanodet.cfg.device.workers_per_gpu = 1
    nanodet.cfg.device.batchsize_per_gpu = 1
    nanodet.cfg.freeze()
    nanodet.load(save_path, verbose=True)
    dataset_metadata = {
        # "data_root": "/media/manos/hdd/Binary_Datasets/Football/1920x1088_22pos_2040neg/bigres",
        "data_root": "/media/manos/hdd/allea_datasets/weedDataset/cropped_images",
        "classes": ["poaceae", "brassicaceae"],
        "dataset_type": "weed",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]

    dataset = XMLBasedDataset(root=f'{data_root}/val', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)
    nanodet.eval(dataset, verbose=True, logging=True, mode="test")
