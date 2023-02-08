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

import os
import datetime
import json
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ProgressBar
import torch_tensorrt as trt

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.check_point import save_model_state
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.collate import naive_collate
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset import build_dataset
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.trainer.task import TrainingTask
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.evaluator import build_evaluator
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.inferencer.utilities import Predictor
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    load_config,
    load_model_weight,
    mkdir,
)

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.engine.learners import Learner
from urllib.request import urlretrieve

import onnxruntime as ort

_MODEL_NAMES = {"EfficientNet_Lite0_320", "EfficientNet_Lite1_416", "EfficientNet_Lite2_512",
                "RepVGG_A0_416", "t", "g", "m", "m_416", "m_0.5x", "m_1.5x", "m_1.5x_416",
                "plus_m_320", "plus_m_1.5x_320", "plus_m_416", "plus_m_1.5x_416", "m_32", "custom"}


class NanodetLearner(Learner):
    def __init__(self, model_to_use="m", iters=None, lr=None, batch_size=None, checkpoint_after_iter=None,
                 checkpoint_load_iter=None, temp_path='', device='cuda', weight_decay=None, warmup_steps=None,
                 warmup_ratio=None, lr_schedule_T_max=None, lr_schedule_eta_min=None, grad_clip=None, model_log_name=None):

        """Initialise the Nanodet Learner"""

        self.cfg = self._load_hparam(model_to_use)
        self.lr_schedule_T_max = lr_schedule_T_max
        self.lr_schedule_eta_min = lr_schedule_eta_min
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip

        self.overwrite_config(lr=lr, weight_decay=weight_decay, iters=iters, batch_size=batch_size,
                              checkpoint_after_iter=checkpoint_after_iter, checkpoint_load_iter=checkpoint_load_iter,
                              temp_path=temp_path)

        self.lr = float(self.cfg.schedule.optimizer.lr)
        self.weight_decay = float(self.cfg.schedule.optimizer.weight_decay)
        self.iters = int(self.cfg.schedule.total_epochs)
        self.batch_size = int(self.cfg.device.batchsize_per_gpu)
        self.temp_path = self.cfg.save_dir
        self.checkpoint_after_iter = int(self.cfg.schedule.val_intervals)
        self.checkpoint_load_iter = int(self.cfg.schedule.resume)
        self.device = device
        self.classes = self.cfg.class_names

        super(NanodetLearner, self).__init__(lr=self.lr, iters=self.iters, batch_size=self.batch_size,
                                             checkpoint_after_iter=self.checkpoint_after_iter,
                                             checkpoint_load_iter=self.checkpoint_load_iter,
                                             temp_path=self.temp_path, device=self.device)

        self.ort_session = None
        self.jit_model = None
        self.trt_model = None
        self.predictor = None

        self.pipeline = None
        self.model = build_model(self.cfg.model)
        self.logger = None
        self.task = None
        if model_log_name is not None:
            writer = SummaryWriter(f'models/{model_log_name}')
            writer.add_graph(self.model.eval(), self.__dummy_input()[0].to("cpu").unsqueeze(0))
            writer.close()

    def _load_hparam(self, model: str):
        """ Load hyperparameters for nanodet models and training configuration

        :parameter model: The name of the model of which we want to load the config file
        :type model: str
        :return: config with hyperparameters
        :rtype: dict
        """
        assert (
                model in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        full_path = list()
        path = Path(__file__).parent / "algorithm" / "config"
        wanted_file = "nanodet_{}.yml".format(model)
        for root, dir, files in os.walk(path):
            if wanted_file in files:
                full_path.append(os.path.join(root, wanted_file))
        assert (len(full_path) == 1), f"You must have only one nanodet_{model}.yaml file in your config folder"
        load_config(cfg, full_path[0])
        return cfg

    def overwrite_config(self, lr=0.001, weight_decay=0.05, iters=10, batch_size=64, checkpoint_after_iter=0,
                         checkpoint_load_iter=0, temp_path=''):
        """
        Helping method for config file update to overwrite the cfg with arguments of OpenDR.
        :param lr: learning rate used in training
        :type lr: float, optional
        :param weight_decay: weight_decay used in training
        :type weight_decay: float, optional
        :param iters: max epoches that the training will be run
        :type iters: int, optional
        :param batch_size: batch size of each gpu in use, if device is cpu batch size
         will be used one single time for training
        :type batch_size: int, optional
        :param checkpoint_after_iter: after that number of epoches, evaluation will be
         performed and one checkpoint will be saved
        :type checkpoint_after_iter: int, optional
        :param checkpoint_load_iter: the epoch in which checkpoint we want to load
        :type checkpoint_load_iter: int, optional
        :param temp_path: path to a temporal dictionary for saving models, logs and tensorboard graphs.
         If temp_path='' the `cfg.save_dir` will be used instead.
        :type temp_path: str, optional
        """
        self.cfg.defrost()

        # Nanodet specific parameters
        if self.cfg.model.arch.head.num_classes != len(self.cfg.class_names):
            raise ValueError(
                "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
                "but got {} and {}".format(
                    self.cfg.model.arch.head.num_classes, len(self.cfg.class_names)
                )
            )
        if self.warmup_steps is not None:
            self.cfg.schedule.warmup.warmup_steps = self.warmup_steps
        if self.warmup_ratio is not None:
            self.cfg.schedule.warmup.warmup_ratio = self.warmup_ratio
        if self.lr_schedule_T_max is not None:
            self.cfg.schedule.lr_schedule.T_max = self.lr_schedule_T_max
        if self.lr_schedule_eta_min is not None:
            self.cfg.schedule.lr_schedule.eta_min = self.lr_schedule_eta_min
        if self.grad_clip is not None:
            self.cfg.grad_clip = self.grad_clip

        # OpenDR
        if lr is not None:
            self.cfg.schedule.optimizer.lr = lr
        if weight_decay is not None:
            self.cfg.schedule.optimizer.weight_decay = weight_decay
        if iters is not None:
            self.cfg.schedule.total_epochs = iters
        if batch_size is not None:
            self.cfg.device.batchsize_per_gpu = batch_size
        if checkpoint_after_iter is not None:
            self.cfg.schedule.val_intervals = checkpoint_after_iter
        if checkpoint_load_iter is not None:
            self.cfg.schedule.resume = checkpoint_load_iter
        if temp_path != '':
            self.cfg.save_dir = temp_path

        self.cfg.freeze()

    def save(self, path=None, verbose=True):
        """
        Method for saving the current model and metadata in the path provided.
        :param path: path to folder where model will be saved
        :type path: str, optional
        :param verbose: whether to print a success message or not
        :type verbose: bool, optional
        """

        path = path if path is not None else self.cfg.save_dir
        model = self.cfg.check_point_name

        os.makedirs(path, exist_ok=True)

        if self.ort_session:
            self._save_onnx(path, verbose=verbose)
            return
        if self.jit_model:
            self._save_jit(path, verbose=verbose)
            return

        metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                    "optimized": False, "optimizer_info": {}}

        metadata["model_paths"].append("nanodet_{}.pth".format(model))

        if self.task is None:
            print("You haven't called a task yet, only the state of the loaded or initialized model will be saved.")
            save_model_state(os.path.join(path, metadata["model_paths"][0]), self.model, None, verbose)
        else:
            self.task.save_current_model(os.path.join(path, metadata["model_paths"][0]), verbose)

        with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if verbose:
            print("Model metadata saved.")
        return

    def load(self, path=None, verbose=True):
        """
        Loads the model from the path provided.
        :param path: path of the directory where the model was saved
        :type path: str, optional
        :param verbose: whether to print a success message or not, defaults to True
        :type verbose: bool, optional
        """

        path = path if path is not None else self.cfg.save_dir

        model = self.cfg.check_point_name
        if verbose:
            print("Model name:", model, "-->", os.path.join(path, "nanodet_" + model + ".json"))
        with open(os.path.join(path, "nanodet_{}.json".format(model))) as f:
            metadata = json.load(f)

        if metadata['optimized']:
            if metadata['format'] == "onnx":
                self._load_onnx(os.path.join(path, metadata["model_paths"][0]), verbose=verbose)
                print("Loaded ONNX model.")
            else:
                self._load_jit(os.path.join(path, metadata["model_paths"][0]), verbose=verbose)
                print("Loaded JIT model.")
        else:
            ckpt = torch.load(os.path.join(path, metadata["model_paths"][0]), map_location=torch.device(self.device))
            self.model = load_model_weight(self.model, ckpt, verbose)
        if verbose:
            print("Loaded model weights from {}".format(path))
        pass

    def download(self, path=None, mode="pretrained", verbose=True,
                 url=OPENDR_SERVER_URL + "/perception/object_detection_2d/nanodet/"):

        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options are: ["pretrained",
        "images", "test_data"].
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str
        :param mode: one of: ["pretrained", "images", "test_data"], where "pretrained" downloads a pretrained
        network depending on the network chosen in the config file, "images" downloads example inference data,
        and "test_data" downloads additional images and corresponding annotations files
        :type mode: str
        :param verbose: if True, additional information is printed on STDOUT
        :type verbose: bool
        :param url: URL to file location on FTP server
        :type url: str
        """

        valid_modes = ["pretrained", "images", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path
        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":

            model = self.cfg.check_point_name

            path = os.path.join(path, "nanodet_{}".format(model))
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained checkpoint...")

            file_url = os.path.join(url, "pretrained",
                                    "nanodet_{}".format(model),
                                    "nanodet_{}.ckpt".format(model))

            urlretrieve(file_url, os.path.join(path, "nanodet_{}.ckpt".format(model)))

            if verbose:
                print("Downloading pretrain weights if provided...")

            file_url = os.path.join(url, "pretrained", "nanodet_{}".format(model),
                                    "nanodet_{}.pth".format(model))
            try:
                urlretrieve(file_url, os.path.join(path, "nanodet_{}.pth".format(model)))

                if verbose:
                    print("Making metadata...")
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                            "optimized": False, "optimizer_info": {}}

                param_filepath = "nanodet_{}.pth".format(model)
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

            except:
                print("Pretrain weights for this model are not provided!!! \n"
                      "Only the hole checkpoint will be download")

                if verbose:
                    print("Making metadata...")
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                            "optimized": False, "optimizer_info": {}}

                param_filepath = "nanodet_{}.ckpt".format(model)
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, "nanodet_{}.json".format(model)), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

        elif mode == "images":
            file_url = os.path.join(url, "images", "000000000036.jpg")
            if verbose:
                print("Downloading example image...")
            urlretrieve(file_url, os.path.join(path, "000000000036.jpg"))

        elif mode == "test_data":
            os.makedirs(os.path.join(path, "test_data"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train", "JPEGImages"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "train", "Annotations"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val", "JPEGImages"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "val", "Annotations"), exist_ok=True)
            # download image
            file_url = os.path.join(url, "images", "000000000036.jpg")
            if verbose:
                print("Downloading image...")
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "JPEGImages", "000000000036.jpg"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "JPEGImages", "000000000036.jpg"))
            # download annotations
            file_url = os.path.join(url, "annotations", "000000000036.xml")
            if verbose:
                print("Downloading annotations...")
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "Annotations", "000000000036.xml"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "Annotations", "000000000036.xml"))

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def __dummy_input(self):
        width, height = self.cfg.data.val.input_size
        dummy_input = (
            torch.randn((3, width, height), device=self.device, dtype=torch.float32),
            torch.tensor(width, device="cpu", dtype=torch.int64),
            torch.tensor(height, device="cpu", dtype=torch.int64),
            torch.eye(3, device="cpu", dtype=torch.float32),
        )
        return dummy_input

    def _save_onnx(self, onnx_path, do_constant_folding=False, verbose=True, nms_max_num=100):
        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, nms_max_num=nms_max_num)

        os.makedirs(onnx_path, exist_ok=True)
        export_path = os.path.join(onnx_path, "nanodet_{}.onnx".format(self.cfg.check_point_name))

        dummy_input = self.__dummy_input()

        torch.onnx.export(
            self.predictor,
            dummy_input[0],
            export_path,
            verbose=verbose,
            keep_initializers_as_inputs=True,
            do_constant_folding=do_constant_folding,
            opset_version=11,
            input_names=['data'],
            output_names=['output'],
            dynamic_axes={'data': {1: 'width',
                                   2: 'height'}}
        )

        metadata = {"model_paths": ["nanodet_{}.onnx".format(self.cfg.check_point_name)], "framework": "pytorch",
                    "format": "onnx", "has_data": False, "optimized": True, "optimizer_info": {},
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes}}

        with open(os.path.join(onnx_path, "nanodet_{}.json".format(self.cfg.check_point_name)),
                  'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        if verbose:
            print("Finished exporting ONNX model.")

        try:
            import onnxsim
        except:
            print("For compression in optimized models, install onnxsim and rerun optimize.")
            return

        import onnx
        if verbose:
            print("Simplifying ONNX model...")
        input_data = {"data": dummy_input[0].detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(export_path, input_data=input_data)
        if flag:
            onnx.save(model_sim, export_path)
            if verbose:
                print("ONNX simplified successfully.")
        else:
            if verbose:
                print("ONNX simplified failed.")

    def _load_onnx(self, onnx_path, verbose=True):
        if verbose:
            print("Loading ONNX runtime inference session from {}".format(onnx_path))

        self.ort_session = ort.InferenceSession(onnx_path)

    def _save_trt(self, trt_path, verbose=True, nms_max_num=100):
        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, nms_max_num=nms_max_num)

        # jit_path = trt_path
        # jit_model = torch.jit.load("jit/nanodet_m/nanodet_m.pth", map_location=self.device)

        os.makedirs(trt_path, exist_ok=True)
        export_path = os.path.join(trt_path, "nanodet_{}.ts".format(self.cfg.check_point_name))

        dummy_input = self.__dummy_input()#[0].unsqueeze(0)
        temp_inputs = [
            dummy_input[0],
            dummy_input[1].to(torch.int32),
            dummy_input[2].to(torch.int32),
            dummy_input[3]
        ]
        # inputs = [trt.Input(
        #     min_shape=(),
        #     opt_shape=(),
        #     max_shape=(),
        # )]

        self.predictor.trace_model(self.__dummy_input()[0].unsqueeze(0))
        # traced_model = self.predictor.traced_model
        jit_model = torch.jit.script(self.predictor.model)
        # print(jit_model.graph)
        # enabled_precisions = {torch.float, torch.half, torch.int8}
        enabled_precisions = {torch.float}

        # trt.logging.Level(4)

        with trt.logging.debug():
            trt_ts_module = trt.compile(
                jit_model,
                inputs=self.__dummy_input()[0].unsqueeze(0),
                enabled_precisions=enabled_precisions,
                ir="default",
                truncate_long_and_double=True
            )

        print("done")
        # torch.onnx.export(
        #     self.predictor,
        #     dummy_input[0],
        #     export_path,
        #     verbose=verbose,
        #     keep_initializers_as_inputs=True,
        #     do_constant_folding=do_constant_folding,
        #     opset_version=11,
        #     input_names=['data'],
        #     output_names=['output'],
        #     dynamic_axes={'data': {1: 'width',
        #                            2: 'height'}}
        # )

        metadata = {"model_paths": ["nanodet_{}.ts".format(self.cfg.check_point_name)], "framework": "pytorch",
                    "format": "tensorRT", "has_data": False, "optimized": True, "optimizer_info": {},
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes}}

        with open(os.path.join(trt_path, "nanodet_{}.json".format(self.cfg.check_point_name)),
                  'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        if verbose:
            print("Finished exporting tensorRT model.")

    def _load_trt(self, trt_path, verbose=True):
        if verbose:
            print("Loading TensorRT runtime inference session from {}".format(trt_path))

        self.jit_model = torch.jit.load(trt_path, map_location=self.device)

    def _save_jit(self, jit_path, verbose=True, nms_max_num=100):
        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, nms_max_num=nms_max_num)

        os.makedirs(jit_path, exist_ok=True)

        dummy_input = self.__dummy_input()

        with torch.no_grad():
            export_path = os.path.join(jit_path, "nanodet_{}.pth".format(self.cfg.check_point_name))
            self.predictor.trace_model(dummy_input)
            model_traced = torch.jit.script(self.predictor)

            metadata = {"model_paths": ["nanodet_{}.pth".format(self.cfg.check_point_name)], "framework": "pytorch",
                        "format": "pth", "has_data": False, "optimized": True, "optimizer_info": {},
                        "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes}}
            model_traced.save(export_path)

            with open(os.path.join(jit_path, "nanodet_{}.json".format(self.cfg.check_point_name)),
                      'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            if verbose:
                print("Finished export to TorchScript.")

    def _load_jit(self, jit_path, verbose=True):
        if verbose:
            print("Loading JIT model from {}.".format(jit_path))

        self.jit_model = torch.jit.load(jit_path, map_location=self.device)

    def optimize(self, export_path, verbose=True, optimization="jit", nms_max_num=100):
        """
        Method for optimizing the model with ONNX or JIT.
        :param export_path: The file path to the folder where the optimized model will be saved. If a model already
        exists at this path, it will be overwritten.
        :type export_path: str
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool, optional
        :param optimization: the kind of optimization you want to perform [jit, onnx]
        :type optimization: str
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int
        """

        optimization = optimization.lower()
        # if not os.path.exists(export_path):
        if optimization == "jit":
            self._save_jit(export_path, verbose=verbose, nms_max_num=nms_max_num)
        elif optimization == "onnx":
            self._save_onnx(export_path, verbose=verbose, nms_max_num=nms_max_num)
        elif optimization == "trt":
            self._save_trt(export_path, verbose=verbose, nms_max_num=nms_max_num)
        else:
            assert NotImplementedError
        with open(os.path.join(export_path, "nanodet_{}.json".format(self.cfg.check_point_name))) as f:
            metadata = json.load(f)
        if optimization == "jit":
            self._load_jit(os.path.join(export_path, metadata["model_paths"][0]), verbose)
        elif optimization == "onnx":
            self._load_onnx(os.path.join(export_path, metadata["model_paths"][0]), verbose)
        elif optimization == "trt":
            self._load_trt(os.path.join(export_path, metadata["model_paths"][0]), verbose)
        else:
            assert NotImplementedError

    def fit(self, dataset, val_dataset=None, logging_path='', verbose=True, logging=False, seed=123, local_rank=1):
        """
        This method is used to train the detector on the COCO dataset. Validation is performed in a val_dataset if
        provided, else validation is performed in training dataset.
        :param dataset: training dataset; COCO and Pascal VOC are supported as ExternalDataset types,
        with 'coco' or 'voc' dataset_type attributes. custom DetectionDataset types are not supported at the moment.
        Any xml type dataset can be use if voc is used in datatype.
        :type dataset: ExternalDataset, DetectionDataset not implemented yet
        :param val_dataset: validation dataset object
        :type val_dataset: ExternalDataset, DetectionDataset not implemented yet
        :param logging_path: subdirectory in temp_path to save logger outputs
        :type logging_path: str
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool
        :param logging: if set to True, text and STDOUT logging will be used
        :type logging: bool
        :param seed: seed for reproducibility
        :type seed: int
        :param local_rank: for distribution learning
        :type local_rank: int
        """

        mkdir(local_rank, self.cfg.save_dir)

        if logging:
            self.logger = NanoDetLightningLogger(self.temp_path + "/" + logging_path)
            self.logger.dump_cfg(self.cfg)

        if seed != '' or seed is not None:
            if logging:
                self.logger.info("Set random seed to {}".format(seed))
            pl.seed_everything(seed)

        if logging:
            self.logger.info("Setting up data...")
        elif verbose:
            print("Setting up data...")

        train_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "train")
        val_dataset = train_dataset if val_dataset is None else \
            build_dataset(self.cfg.data.val, val_dataset, self.cfg.class_names, "val")

        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=False,
        )

        # Load state dictionary
        model_resume_path = (
            os.path.join(self.temp_path, "checkpoints", "model_iter_{}.ckpt".format(self.checkpoint_load_iter))
            if self.checkpoint_load_iter > 0 else None
        )

        if logging:
            self.logger.info("Creating task...")
        elif verbose:
            print("Creating task...")
        self.task = TrainingTask(self.cfg, self.model, evaluator)

        gpu_ids = None
        accelerator = None
        if self.device == "cuda":
            gpu_ids = self.cfg.device.gpu_ids
            accelerator = None if len(gpu_ids) <= 1 else "ddp"

        trainer = pl.Trainer(
            default_root_dir=self.temp_path,
            max_epochs=self.iters,
            gpus=gpu_ids,
            check_val_every_n_epoch=self.checkpoint_after_iter,
            accelerator=accelerator,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            resume_from_checkpoint=model_resume_path,
            callbacks=[ProgressBar(refresh_rate=0)],
            logger=self.logger,
            benchmark=True,
            gradient_clip_val=self.cfg.get("grad_clip", 0.0),
        )

        trainer.fit(self.task, train_dataloader, val_dataloader)

    def eval(self, dataset, verbose=True, logging=False, local_rank=1):
        """
        This method performs evaluation on a given dataset and returns a dictionary with the evaluation results.
        :param dataset: dataset object, to perform evaluation on
        :type dataset: ExternalDataset, XMLBasedDataset
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool
        :param logging: if set to True, text and STDOUT logging will be used
        :type logging: bool
        :param local_rank: for distribution learning
        :type local_rank: int
        """

        timestr = datetime.datetime.now().__format__("%Y_%m_%d_%H:%M:%S")
        save_dir = os.path.join(self.cfg.save_dir, timestr)
        mkdir(local_rank, save_dir)

        if logging:
            self.logger = NanoDetLightningLogger(save_dir)

        self.cfg.update({"test_mode": "val"})

        if logging:
            self.logger.info("Setting up data...")
        elif verbose:
            print("Setting up data...")

        val_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "val")

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=naive_collate,
            drop_last=False,
        )
        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        if logging:
            self.logger.info("Creating task...")
        elif verbose:
            print("Creating task...")

        self.task = TrainingTask(self.cfg, self.model, evaluator)

        gpu_ids = None
        accelerator = None
        if self.device == "cuda":
            gpu_ids = self.cfg.device.gpu_ids
            accelerator = None if len(gpu_ids) <= 1 else "ddp"

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            gpus=gpu_ids,
            accelerator=accelerator,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            logger=self.logger,
        )
        if self.logger:
            self.logger.info("Starting testing...")
        elif verbose:
            print("Starting testing...")

        test_results = (verbose or logging)
        return trainer.test(self.task, val_dataloader, verbose=test_results)

    def infer(self, input, conf_threshold=0.35, iou_threshold=0.6, nms_max_num=100):
        """
        Performs inference
        :param input: input image to perform inference on
        :type input: opendr.data.Image
        :param conf_threshold: confidence threshold
        :type conf_threshold: float, optional
        :param iou_threshold: iou threshold
        :type iou_threshold: float, optional
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int
        :return: list of bounding boxes of last image of input or last frame of the video
        :rtype: opendr.engine.target.BoundingBoxList
        """
        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                                       iou_thresh=iou_threshold, nms_max_num=nms_max_num)

        if not isinstance(input, Image):
            input = Image(input)
        _input = input.opencv()

        _input, *metadata = self.predictor.preprocessing(_input)

        if self.ort_session:
            if self.jit_model or self.trt_model:
                warnings.warn(
                    "Warning: More than one optimizations are initialized, inference will run in ONNX mode by default.\n"
                    "To run in a specific optimization please delete the self.ort_session, self.jit_model or self.trt_model like: detector.ort_session = None.")
            preds = self.ort_session.run(['output'], {'data': _input.cpu().detach().numpy()})
            res = self.predictor.postprocessing(torch.from_numpy(preds[0]), _input, *metadata)
        elif self.jit_model:
            if self.trt_model:
                warnings.warn(
                    "Warning: Both JIT and TensorRT models are initialized, inference will run in JIT mode by default.\n"
                    "To run in TensorRT please delete the self.jit_model like: detector.jit_model = None.")
            res = self.jit_model(_input, *metadata)
        elif self.trt_model:
            preds = self.trt_model(_input)
            res = self.predictor.postprocessing(preds, _input, *metadata)
        else:
            preds = self.predictor(_input, *metadata)
            res = self.predictor.postprocessing(preds, _input, *metadata)

        bounding_boxes = []
        for label in res:
            for box in label:
                box = box.to("cpu")
                bbox = BoundingBox(left=box[0], top=box[1],
                                   width=box[2] - box[0],
                                   height=box[3] - box[1],
                                   name=box[5],
                                   score=box[4])
                bounding_boxes.append(bbox)
        bounding_boxes = BoundingBoxList(bounding_boxes)
        bounding_boxes.data.sort(key=lambda v: v.confidence)

        return bounding_boxes

    def benchmark(self, input, repetitions=1000, warmup=100, nms_max_num=100):
        """
        Performs inference
        :param repetitions: input image to perform inference on
        :type repetitions: opendr.data.Image
        :param warmup: confidence threshold
        :type warmup: float, optional
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int
        """

        import numpy as np

        if not isinstance(input, Image):
            input = Image(input)
        preprocess_input = input.opencv()

        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, nms_max_num=nms_max_num)

        # Preprocess measurement
        preprocess_starter, preprocess_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True)
        preprocess_timings = np.zeros((repetitions, 1))
        for run in range(warmup):
            _ = self.predictor.preprocessing(preprocess_input)
        for run in range(repetitions):
            preprocess_starter.record()
            _input, *metadata = self.predictor.preprocessing(preprocess_input)
            preprocess_ender.record()
            torch.cuda.synchronize()
            preprocess_timings[run] = preprocess_starter.elapsed_time(preprocess_ender)

        # Onnx measurements
        if self.ort_session:
            # Inference
            onnx_infer_starter, onnx_infer_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True)
            onnx_infer_timings = np.zeros((repetitions, 1))
            for run in range(warmup):
                _ = self.ort_session.run(['output'], {'data': _input.cpu().detach().numpy()})
            for run in range(repetitions):
                onnx_infer_starter.record()
                preds = self.ort_session.run(['output'], {'data': _input.cpu().detach().numpy()})
                onnx_infer_ender.record()
                torch.cuda.synchronize()
                onnx_infer_timings[run] = onnx_infer_starter.elapsed_time(onnx_infer_ender)
            # Do not measure postprocessing because we will measure it in the actual run
            res = self.predictor.postprocessing(torch.from_numpy(preds[0]), _input, *metadata)

        # Jit measurements
        if self.jit_model:
            jit_infer_starter, jit_infer_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True)
            jit_infer_timings = np.zeros((repetitions, 1))

            for run in range(warmup):
                _ = self.jit_model(_input, *metadata)
            for run in range(repetitions):
                jit_infer_starter.record()
                res = self.jit_model(_input, *metadata)
                jit_infer_ender.record()
                torch.cuda.synchronize()
                jit_infer_timings[run] = jit_infer_starter.elapsed_time(jit_infer_ender)
            res = res.cpu()

        # Original Python measurements
        infer_starter, infer_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        infer_timings = np.zeros((repetitions, 1))
        for run in range(warmup):
            _ = self.predictor(_input, *metadata)
        for run in range(repetitions):
            infer_starter.record()
            preds = self.predictor(_input, *metadata)
            infer_ender.record()
            torch.cuda.synchronize()
            infer_timings[run] = infer_starter.elapsed_time(infer_ender)

        # Post-processing measurements
        # post_starter, post_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # post_timings = np.zeros((repetitions, 1))
        # for run in range(warmup):
        #     _ = self.predictor.postprocessing(preds, _input, *metadata)
        # for run in range(repetitions):
        #     post_starter.record()
        #     res = self.predictor.postprocessing(preds, _input, *metadata)
        #     post_ender.record()
        #     torch.cuda.synchronize()
        #     post_timings[run] = post_starter.elapsed_time(post_ender)

        # Measure std and mean of times
        std_preprocess_timings = np.std(preprocess_timings)
        std_infer_timings = np.std(infer_timings)
        # std_post_timings = np.std(post_timings)

        if self.jit_model:
            std_jit_infer_timings = np.std(jit_infer_timings)

        if self.ort_session:
            std_onnx_infer_timings = np.std(onnx_infer_timings)

        mean_preprocess_timings = np.mean(preprocess_timings)
        mean_infer_timings = np.mean(infer_timings)
        # mean_post_timings = np.mean(post_timings)

        if self.jit_model:
            mean_jit_infer_timings = np.mean(jit_infer_timings)

        if self.ort_session:
            mean_onnx_infer_timings = np.mean(onnx_infer_timings)

        # mean times to fps, torch measures in milliseconds
        fps_preprocess_timings = 1000/mean_preprocess_timings
        fps_infer_timings = 1000/mean_infer_timings
        # fps_ifer_post_timings = 1000/(mean_infer_timings + mean_post_timings)
        # fps_post_timings = 1000/mean_post_timings
        if self.jit_model:
            fps_jit_infer_timings = 1000/mean_jit_infer_timings

        if self.ort_session:
            fps_onnx_infer_timings = 1000/mean_onnx_infer_timings
            # fps_onnx_infer_post_timings = 1000/(mean_onnx_infer_timings + mean_post_timings)

        # Print measurements
        print(f"\n\nMeasure of model: {self.cfg.check_point_name}")
        print(f"\n=== Python measurements === \n"
              f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
              f"infer          fps = {fps_infer_timings} evn/s\n")
              # f"postprocessing fps = {fps_post_timings} evn/s\n"
              # f"infer + postpr fps = {fps_ifer_post_timings} evn/s")
        if self.jit_model:
            print(f"\n\n=== JIT measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer + postpr fps = {fps_jit_infer_timings} evn/s")
        if self.ort_session:
            print(f"\n\n=== ONNX measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer          fps = {fps_onnx_infer_timings} evn/s\n")
                  # f"postprocessing fps = {fps_post_timings} evn/s\n"
                  # f"infer + postpr fps = {fps_onnx_infer_post_timings} evn/s\n\n\n")

        return