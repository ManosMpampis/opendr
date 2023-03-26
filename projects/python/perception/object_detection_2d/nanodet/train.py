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
    parser.add_argument("--model", help="Model for which a config file will be used", type=str, default="vgg_64_very_small") # "temp" "m" "plus_m_320"
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=0.01)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving and evaluations",
                        type=int, default=1)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=1000)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from",
                        type=int, default=0)

    args = parser.parse_args()

    dataset_metadata = {
        "data_root": "/media/manos/hdd/Binary_Datasets/Football/96x96_1pos_9neg",
        "classes": ["player"],
        "dataset_type": "BINARY_FOOTBALL",
    }
    data_root = dataset_metadata["data_root"]
    classes = dataset_metadata["classes"]
    dataset_type = dataset_metadata["dataset_type"]

    dataset = XMLBasedDataset(root=f'{data_root}/1image', dataset_type=dataset_type, images_dir='images',
                              annotations_dir='annotations', classes=classes)

    val_dataset = XMLBasedDataset(root=f'{data_root}/1image', dataset_type=dataset_type, images_dir='images',
                                  annotations_dir='annotations', classes=classes)

    # nanodet = NanodetLearner(model_to_use=args.model, iters=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
    #                          checkpoint_after_iter=args.checkpoint_freq, checkpoint_load_iter=args.resume_from,
    #                          device=args.device)

    nanodet = NanodetLearner(model_to_use=args.model, iters=None, lr=None, batch_size=1,
                             checkpoint_after_iter=1, checkpoint_load_iter=0,
                             device="cuda", model_log_name=f"{args.model}")

    # nanodet.download("./predefined_examples", mode="pretrained")
    # nanodet.load(f"./predefined_examples/nanodet_{args.model}", verbose=True)
    nanodet.fit(dataset, val_dataset, logging=True)
    nanodet.save(f"./saved/nanodet_{args.model}")
