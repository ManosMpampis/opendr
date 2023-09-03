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

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="test_big")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", help="iterations of warmup", type=int, default=3)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving and evaluations",
                        type=int, default=1)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=500)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from",
                        type=int, default=0)
    parser.add_argument("--dataset_path", help="Path to dataset", type=str,
                        default="/media/manos/hdd/allea_datasets/weedDataset/cropped_images")
    args = parser.parse_args()

    dataset_metadata = {
        "data_root": args.dataset_path,
        "classes": ["poaceae", "brassicaceae"],
        "dataset_type": "WEED",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]

    dataset = XMLBasedDataset(root=f'{data_root}/train', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)

    # val_dataset = XMLBasedDataset(root=f'{data_root}/val', dataset_type=dataset_type, images_dir='images',
    #                               annotations_dir='annotations', classes=classes)
    val_dataset = None

    nanodet = NanodetLearner(model_to_use=args.model, iters=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                             checkpoint_after_iter=args.checkpoint_freq, checkpoint_load_iter=args.resume_from,
                             device="cuda", model_log_name=f"{args.model}", warmup_steps=args.warmup_steps)

    nanodet.fit(dataset, val_dataset, logging=True, profile=False, verbose=False)
    nanodet.save(f"./saved/nanodet_{args.model}")
