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
import math
import os
import datetime
import json
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
try:
    from pytorch_lightning.callbacks import TQDMProgressBar
except ImportError:
    from pytorch_lightning.callbacks import ProgressBar as TQDMProgressBar

try:
    try:
        import pycuda.autoprimaryctx as pycuda_autinit  # noqa
    except ModuleNotFoundError:
        import pycuda.autoinit as pycuda_autinit  # noqa
    var = pycuda_autinit

    import tensorrt as trt
    from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.inferencer import trt_dep
    TENSORRT_WARNING = None
except ImportError:
    TENSORRT_WARNING = ("TensorRT can be implemented only in GPU installation of OpenDR toolkit, please install"
                        "the toolkit with GPU capabilities first or install pycuda and TensorRT.")

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.check_point import save_model_state
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.arch import build_model
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.collate import naive_collate
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.dataset import build_dataset
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.trainer.task import TrainingTask
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.evaluator import build_evaluator
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.inferencer.utilities import Predictor
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (
    NanoDetLightningLogger,
    NanoDetLightningTensorboardLogger,
    cfg,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
    autobatch,
)

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.engine.learners import Learner
from urllib.request import urlretrieve

import time
import onnxruntime as ort

# _MODEL_NAMES = {"EfficientNet_Lite0_320", "EfficientNet_Lite1_416", "EfficientNet_Lite2_512",
#                 "RepVGG_A0_416", "t", "g", "m", "m_416", "m_0.5x", "m_1.5x", "m_1.5x_416",
#                 "plus_m_320", "plus_m_1.5x_320", "plus_m_416", "plus_m_1.5x_416", "m_32", "vgg_64", "custom", "temp"}


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("medium")
except:
    print("no matmul32 capabilities in torch version 1.9")


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
        self.model = self.model.to(device)

        self.logger = None
        self.task = None

        #warmup run
        _ = self.model(self.__dummy_input()[0])

        if model_log_name is not None:
            # if os.path.exists(f'./models/{model_log_name}'):
            #     import shutil
            #     shutil.rmtree(f'./models/{model_log_name}')
            writer = SummaryWriter(f'./models/{model_log_name}')
            writer.add_graph(self.model.eval().to("cpu"), self.__dummy_input()[0].to("cpu"))
            writer.close()
            self.model = self.model.to(device)

    def _load_hparam(self, model: str):
        """ Load hyperparameters for nanodet models and training configuration

        :parameter model: The name of the model of which we want to load the config file
        :type model: str
        :return: config with hyperparameters
        :rtype: dict
        """
        # assert (
        #         model in _MODEL_NAMES
        # ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        full_path = list()
        path = Path(__file__).parent / "algorithm" / "config"
        wanted_file = f"nanodet_{model}.yml"
        for root, dir, files in os.walk(path):
            if wanted_file in files:
                full_path.append(os.path.join(root, wanted_file))
        assert (len(full_path) == 1), f"You must have only one nanodet_{model}.yaml file in your config folder but found {len(full_path)}"
        load_config(cfg, full_path[0])
        return cfg

    def overwrite_config(self, lr=0.001, weight_decay=0.05, iters=10, batch_size=-1, checkpoint_after_iter=0,
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
                f"cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
                f"but got {self.cfg.model.arch.head.num_classes} and {len(self.cfg.class_names)}"
            )
        if self.warmup_steps is not None:
            self.cfg.schedule.warmup.steps = self.warmup_steps
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

        metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                    "optimized": False, "optimizer_info": {}}

        metadata["model_paths"].append(f"nanodet_{model}.pth")

        if self.task is None:
            self._info("You haven't called a task yet,"
                       " only the state of the loaded or initialized model will be saved.", True)
            save_model_state(os.path.join(path, metadata["model_paths"][0]), self.model, None, verbose)
        else:
            self.task.save_current_model(os.path.join(path, metadata["model_paths"][0]), verbose)

        with open(os.path.join(path, f"nanodet_{model}.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        self._info("Model metadata saved.", verbose)
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
        self._info(f"Model name: {model} --> {os.path.join(path, 'nanodet_' + model + '.json')}", verbose)
        with open(os.path.join(path, f"nanodet_{model}.json")) as f:
            metadata = json.load(f)

        if metadata['optimized']:
            if metadata['format'] == "onnx":
                self._load_onnx(os.path.join(path, metadata["model_paths"][0]), verbose=verbose)
                self._info("Loaded ONNX model.", True)
            elif metadata['format'] == "TensorRT":
                self._load_trt(os.path.join(path, metadata["model_paths"][0]), verbose=verbose)
                self._info("Loaded TensorRT model.", True)
            else:
                self._load_jit(os.path.join(path, metadata["model_paths"][0]), verbose=verbose)
                self._info("Loaded JIT model.", True)
        else:
            ckpt = torch.load(os.path.join(path, metadata["model_paths"][0]), map_location=torch.device(self.device))
            self.model = load_model_weight(self.model, ckpt, verbose)
        self._info(f"Loaded model weights from {path}", verbose)
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

            path = os.path.join(path, f"nanodet_{model}")
            if not os.path.exists(path):
                os.makedirs(path)

            checkpoint_file = os.path.join(path, f"nanodet_{model}.ckpt")
            if os.path.isfile(checkpoint_file):
                return

            self._info("Downloading pretrained checkpoint...", verbose)
            file_url = os.path.join(url, "pretrained",
                                    f"nanodet_{model}",
                                    f"nanodet_{model}.ckpt")

            urlretrieve(file_url, checkpoint_file)

            self._info("Downloading pretrain weights if provided...", verbose)
            file_url = os.path.join(url, "pretrained", f"nanodet_{model}",
                                    f"nanodet_{model}.pth")
            try:
                pytorch_save_file = os.path.join(path, f"nanodet_{model}.pth")
                if os.path.isfile(pytorch_save_file):
                    return

                urlretrieve(file_url, pytorch_save_file)

                self._info("Making metadata...", verbose)
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                            "optimized": False, "optimizer_info": {}}

                param_filepath = f"nanodet_{model}.pth"
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, f"nanodet_{model}.json"), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

            except:
                self._info("Pretrained weights for this model are not provided. \n"
                           "Only the whole checkpoint will be downloaded", True)

                self._info("Making metadata...", verbose)
                metadata = {"model_paths": [], "framework": "pytorch", "format": "pth", "has_data": False,
                            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                            "optimized": False, "optimizer_info": {}}

                param_filepath = f"nanodet_{model}.ckpt"
                metadata["model_paths"].append(param_filepath)
                with open(os.path.join(path, f"nanodet_{model}.json"), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

        elif mode == "images":
            file_url = os.path.join(url, "images", "000000000036.jpg")
            image_file = os.path.join(path, "000000000036.jpg")
            if os.path.isfile(image_file):
                return

            self._info("Downloading example image...", verbose)
            urlretrieve(file_url, image_file)

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

            self._info("Downloading image...", verbose)
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "JPEGImages", "000000000036.jpg"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "JPEGImages", "000000000036.jpg"))
            # download annotations
            file_url = os.path.join(url, "annotations", "000000000036.xml")

            self._info("Downloading annotations...", verbose)
            urlretrieve(file_url, os.path.join(path, "test_data", "train", "Annotations", "000000000036.xml"))
            urlretrieve(file_url, os.path.join(path, "test_data", "val", "Annotations", "000000000036.xml"))

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def __dummy_input(self, hf=False, ch_l=False):
        from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.batch_process import divisible_padding

        width, height = self.cfg.data.val.input_size
        dummy_img = divisible_padding(
            torch.zeros((3, height, width), device=self.device, dtype=torch.half if hf else torch.float32),
            divisible=torch.tensor(32, device=self.device, dtype=torch.half if hf else torch.float32)
        )
        dummy_img = dummy_img.contiguous(memory_format=torch.channels_last) if ch_l else dummy_img
        dummy_input = (
            dummy_img,
            torch.tensor(width, device=self.device, dtype=torch.int64),
            torch.tensor(height, device=self.device, dtype=torch.int64),
            torch.eye(3, device=self.device, dtype=torch.half if hf else torch.float32),
        )
        return dummy_input

    def __cv_dumy_input(self, hf=False, zeros=True):
        try:
            width, height = self.cfg.data.bench_test.input_size
        except AttributeError as e:
            self.info(f"{e}, not bench_est.input_size int yaml file, val will be used instead")
            width, height = self.cfg.data.val.input_size
        if zeros:
            output = torch.zeros((width, height, 3), device="cpu", dtype=torch.half if hf else torch.float32).numpy()
        else:
            output = torch.rand((width, height, 3), device="cpu", dtype=torch.half if hf else torch.float32).numpy()
        return output

    def _save_onnx(self, onnx_path, predictor, do_constant_folding=False, verbose=True):

        os.makedirs(onnx_path, exist_ok=True)
        export_path = os.path.join(onnx_path, f"nanodet_{self.cfg.check_point_name}.onnx")

        dummy_input = self.__dummy_input(hf=predictor.hf)
        dynamic = None
        if predictor.dynamic:
            assert not predictor.hf, '--hf not compatible with --dynamic, i.e. use either --hf or --dynamic but not both'
            dynamic = {"data": {2: 'width', 3: 'height'}, "output": {1: "feature_points"}}

        if verbose is False:
            ort.set_default_logger_severity(3)
        torch.onnx.export(
            predictor,
            dummy_input[0],
            export_path,
            verbose=verbose,
            keep_initializers_as_inputs=True,
            do_constant_folding=do_constant_folding,
            opset_version=11,
            input_names=['data'],
            output_names=['output'],
            dynamic_axes=dynamic or None
        )

        metadata = {"model_paths": [f"nanodet_{self.cfg.check_point_name}.onnx"], "framework": "pytorch",
                    "format": "onnx", "has_data": False, "optimized": True, "optimizer_info": {},
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes,
                                         "conf_threshold": predictor.conf_thresh,
                                         "iou_threshold": predictor.iou_thresh}}

        with open(os.path.join(onnx_path, f"nanodet_{self.cfg.check_point_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        self._info("Finished exporting ONNX model.", verbose)
        try:
            import onnxsim
        except:
            self._info("For compression in optimized models, install onnxsim and rerun optimize.", True)
            return

        import onnx
        self._info("Simplifying ONNX model...", verbose)
        input_data = {"data": dummy_input[0].detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(export_path, input_data=input_data)
        if flag:
            onnx.save(model_sim, export_path)
            self._info("ONNX simplified successfully.", verbose)
        else:
            self._info("ONNX simplified failed.", verbose)

    def _load_onnx(self, onnx_path, verbose=True):
        onnx_path = onnx_path[0]
        self._info(f"Loading ONNX runtime inference session from {onnx_path}", verbose)
        self.ort_session = ort.InferenceSession(onnx_path)
        return

    def _save_trt(self, trt_path, predictor, verbose=True):
        assert TENSORRT_WARNING is None, TENSORRT_WARNING
        os.makedirs(trt_path, exist_ok=True)

        export_path_onnx = os.path.join(trt_path, f"nanodet_{self.cfg.check_point_name}.onnx")
        export_path_trt = os.path.join(trt_path, f"nanodet_{self.cfg.check_point_name}.trt")
        export_path_json = os.path.join(trt_path, f"nanodet_{self.cfg.check_point_name}.json")

        if not os.path.exists(export_path_onnx):
            assert torch.__version__[2:4] == "13", \
                f"tensorRT onnx parser is not compatible with resize implementations of pytorch before version 1.13.0." \
                f" Please update your pytorch and try again, or provide a valid onnx file into {export_path_onnx}"
            self._save_onnx(trt_path, predictor, verbose=verbose)

        trt_logger_level = trt.Logger.INFO if verbose else trt.Logger.ERROR
        TRT_LOGGER = trt.Logger(trt_logger_level)

        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        config.max_workspace_size = trt_dep.GiB(4)

        network = builder.create_network(trt_dep.EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if not parser.parse_from_file(export_path_onnx):
            for error in range(parser.num_errors):
                self._info(parser.get_error(error), True)
            raise RuntimeError(f'Failed to parse the ONNX file: {export_path_onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            self._info(f'TensorRT: input "{inp.name}" with shape {inp.shape} {inp.dtype}', verbose)
        for out in outputs:
            self._info(f'TensorRT: output "{out.name}" with shape {out.shape} {out.dtype}', verbose)

        im = self.__dummy_input(hf=predictor.hf)[0]
        if predictor.dynamic:
            assert not predictor.hf, '--hf not compatible with --dynamic, i.e. use either --hf or --dynamic but not both'
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, im.shape[1], 320, 320), im.shape, im.shape)
            config.add_optimization_profile(profile)
        if predictor.hf:
            if not builder.platform_has_fast_fp16:
                self._info("Platform do not support fast fp16", True)
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        with open(export_path_trt, 'wb') as f:
            f.write(engine.serialize())

        metadata = {
            "model_paths": [f"nanodet_{self.cfg.check_point_name}.trt"],
            "framework": "pytorch", "format": "TensorRT", "has_data": False, "optimized": True, "optimizer_info": {},
            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes,
                                 "num_classes": len(self.classes)}, "hf": predictor.hf}

        with open(export_path_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return

    def _load_trt(self, trt_paths, verbose=True):
        self._info(f"Loading TensorRT runtime inference session from {trt_paths[0]}", verbose)
        trt_logger_level = trt.Logger.WARNING if verbose else trt.Logger.ERROR
        TRT_LOGGER = trt.Logger(trt_logger_level)
        with open(f'{trt_paths[0]}', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.trt_model = trt_dep.trt_model(engine, self.device)
        return

    def optimize_c_model(self, export_path, conf_threshold, iou_threshold, nms_max_num, hf=False, dynamic=False, verbose=True):
        os.makedirs(export_path, exist_ok=True)
        jit_path = os.path.join(export_path, "nanodet_{}.pth".format(self.cfg.check_point_name))

        predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                              iou_thresh=iou_threshold, nms_max_num=nms_max_num, hf=hf, dynamic=dynamic)

        model_jit_forward = predictor.c_script(self.__dummy_input(hf=predictor.hf))

        metadata = {"model_paths": ["nanodet_{}.pth".format(self.cfg.check_point_name)], "framework": "pytorch",
                    "format": "pth", "has_data": False, "optimized": True, "optimizer_info": {},
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes,
                                         "conf_threshold": predictor.conf_thresh,
                                         "iou_threshold": predictor.iou_thresh}}
        model_jit_forward.save(jit_path)

        with open(os.path.join(export_path, "nanodet_{}.json".format(self.cfg.check_point_name)),
                  'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        self._info("Finished export to TorchScript.", verbose)

    def _save_jit(self, jit_path, predictor, verbose=True):
        os.makedirs(jit_path, exist_ok=True)
        export_path = os.path.join(jit_path, "nanodet_{}.pth".format(self.cfg.check_point_name))

        model_traced = predictor.script_model() if predictor.dynamic else \
            predictor.trace_model(self.__dummy_input(hf=predictor.hf))

        metadata = {"model_paths": ["nanodet_{}.pth".format(self.cfg.check_point_name)], "framework": "pytorch",
                    "format": "pth", "has_data": False, "optimized": True, "optimizer_info": {},
                    "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes,
                                         "conf_threshold": predictor.conf_thresh,
                                         "iou_threshold": predictor.iou_thresh}}
        model_traced.save(export_path)

        with open(os.path.join(jit_path, "nanodet_{}.json".format(self.cfg.check_point_name)),
                  'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        self._info("Finished export to TorchScript.", verbose)

    def _load_jit(self, jit_path, verbose=True):
        jit_path = jit_path[0]
        self._info(f"Loading JIT model from {jit_path}.", verbose)
        self.jit_model = torch.jit.load(jit_path, map_location=self.device)

    def optimize(self, export_path, verbose=True, optimization="jit", conf_threshold=0.35, iou_threshold=0.6,
                 nms_max_num=100, hf=False, dynamic=False, ch_l=False, lazy_load=True):
        """
        Method for optimizing the model with ONNX, JIT or TensorRT.
        :param export_path: The file path to the folder where the optimized model will be saved. If a model already
        exists at this path, it will be overwritten.
        :type export_path: str
        :param verbose: if set to True, additional information is printed to STDOUT
        :type verbose: bool, optional
        :param optimization: the kind of optimization you want to perform [jit, onnx, trt]
        :type optimization: str
        :param conf_threshold: confidence threshold
        :type conf_threshold: float, optional
        :param iou_threshold: iou threshold
        :type iou_threshold: float, optional
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int, optional
        :param hf: determines if half precision is used.
        :type hf: bool, optional
        :param dynamic: determines if the model runs with dynamic input. Dynamic input leads to slower inference times.
        :type dynamic: bool, optional
        :param ch_l: determines if inference will run in channel-last format.
        :type ch_l: bool, optional
        :param lazy_load: enables loading optimized model from predetermined path without exporting it each time.
        :type lazy_load: bool, optional
        """

        optimization = optimization.lower()
        ch_l = ch_l and (optimization == "jit")
        if not os.path.exists(export_path) or not lazy_load:
            predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                                  iou_thresh=iou_threshold, nms_max_num=nms_max_num, hf=hf, dynamic=dynamic, ch_l=ch_l)
            # Initialization run for legacy_post_process = False
            _ = predictor(self.__dummy_input(hf=hf, ch_l=ch_l)[0])
            if optimization == "trt":
                self._save_trt(export_path, verbose=verbose, predictor=predictor)
            elif optimization == "jit":
                self._save_jit(export_path, verbose=verbose, predictor=predictor)
            elif optimization == "onnx":
                self._save_onnx(export_path, verbose=verbose, predictor=predictor)
            else:
                assert NotImplementedError
        with open(os.path.join(export_path, f"nanodet_{self.cfg.check_point_name}.json")) as f:
            metadata = json.load(f)
        if optimization == "trt":
            self._load_trt([os.path.join(export_path, path) for path in metadata["model_paths"]], verbose)
        elif optimization == "jit":
            self._load_jit([os.path.join(export_path, path) for path in metadata["model_paths"]], verbose)
        elif optimization == "onnx":
            self._load_onnx([os.path.join(export_path, path) for path in metadata["model_paths"]], verbose)
        else:
            assert NotImplementedError
        return

    def fit(self, dataset, val_dataset=None, logging_path='', verbose=True, logging=False, profile=False, seed=123, local_rank=1):
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
        :param profile: if set to True, tensorboard profiler will be saved to logging_path
        :type profile: bool
        :param seed: seed for reproducibility
        :type seed: int
        :param local_rank: for distribution learning
        :type local_rank: int
        """

        mkdir(local_rank, self.cfg.save_dir)

        if logging or verbose:
            logger_fn = NanoDetLightningTensorboardLogger if (logging and profile) else NanoDetLightningLogger
            save_dir = f"{self.temp_path}/{logging_path}" if logging else ""
            self.logger = logger_fn(
                save_dir=save_dir,
                verbose_only=False if logging else True
            )
            self.logger.dump_cfg(self.cfg)

        if seed != '' or seed is not None:
            self._info(f"Set random seed to {seed}", verbose)
            pl.seed_everything(seed)

        self._info("Setting up data...", verbose)

        train_dataset = build_dataset(self.cfg.data.train, dataset, self.cfg.class_names, "train")
        val_dataset = train_dataset if val_dataset is None else \
            build_dataset(self.cfg.data.val, val_dataset, self.cfg.class_names, "val")

        evaluator = build_evaluator(self.cfg.evaluator, val_dataset)

        if self.batch_size == -1:  # autobatch
            torch.backends.cudnn.benchmark = False
            batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
            self.batch_size = autobatch(model=self.model, imgsz=self.cfg.data.train.input_size, batch_size=32,
                                        divisible=32, batch_sizes=batch_sizes)

            self.batch_size = ((self.batch_size + 32 - 1) // 32) * 32

        nbs = self.cfg.device.effective_batchsize  # nominal batch size
        accumulate = 1
        if nbs > 1:
            accumulate = max(math.ceil(nbs / self.batch_size), 1)
            self.batch_size = round(nbs / accumulate)
            self._info(f"After calculate accumulation\n"
                       f"Batch size will be: {self.batch_size}\n"
                       f"With accumulation: {accumulate}.", verbose)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=naive_collate,
            drop_last=False,
        )

        # Load state dictionary
        model_resume_path = (
            os.path.join(self.temp_path, "checkpoints", f"model_iter_{self.checkpoint_load_iter}.ckpt")
            if self.checkpoint_load_iter > 0 else None
        )

        self._info("Creating task...", verbose)

        self.task = TrainingTask(self.cfg, self.model, evaluator, accumulate=accumulate)
        # self.task = TrainingTask(self.cfg, self.model, evaluator)

        if self.cfg.device.gpu_ids == -1 or self.device == "cpu":
            accelerator, devices, strategy, precision = ("cpu", None, None, cfg.device.precision)
        else:
            accelerator, devices, strategy, precision = ("gpu", cfg.device.gpu_ids, None, cfg.device.precision)
            assert len(devices) == 1, ("Distributed learning is not implemented, please use only"
                                       " one gpu device.")

        # if self.cfg.device.gpu_ids == -1 or self.device == "cpu":
        #     gpu_ids, precision = (None, self.cfg.device.precision)
        # else:
        #     gpu_ids, precision = (self.cfg.device.gpu_ids, self.cfg.device.precision)
        #     assert len(gpu_ids) == 1, ("Distributed learning is not implemented, please use only"
        #                                " one gpu device.")
        # trainer = pl.Trainer(
        #     default_root_dir=self.temp_path,
        #     max_epochs=self.iters,
        #     gpus=gpu_ids,
        #     check_val_every_n_epoch=self.checkpoint_after_iter,
        #     accelerator=None,
        #     accumulate_grad_batches=accumulate,
        #     log_every_n_steps=self.cfg.log.interval,
        #     num_sanity_val_steps=0,
        #     resume_from_checkpoint=model_resume_path,
        #     callbacks=[ProgressBar(refresh_rate=0)],
        #     logger=self.logger,
        #     profiler="pytorch" if profile else None,
        #     benchmark=cfg.get("cudnn_benchmark", True),
        #     precision=precision,
        #     gradient_clip_val=self.cfg.get("grad_clip", 0.0),
        # )

        trainer = pl.Trainer(
            default_root_dir=self.temp_path,
            max_epochs=self.iters,
            check_val_every_n_epoch=self.checkpoint_after_iter,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=0)],
            logger=self.logger,
            profiler="pytorch" if profile else None,
            benchmark=cfg.get("cudnn_benchmark", True),
            precision=precision,
            gradient_clip_val=self.cfg.get("grad_clip", 0.0),
        )

        # trainer.fit(self.task, train_dataloader, val_dataloader)
        trainer.fit(self.task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
        return

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

        if logging or verbose:
            self.logger = NanoDetLightningLogger(
                save_dir=save_dir if logging else "",
                verbose_only=False if logging else True
            )

        self.cfg.update({"test_mode": "val"})
        self._info("Setting up data...", verbose)

        val_dataset = build_dataset(self.cfg.data.val, dataset, self.cfg.class_names, "val")

        if self.batch_size == -1:  # autobatch
            torch.backends.cudnn.benchmark = False
            batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
            self.batch_size = autobatch(model=self.model, imgsz=self.cfg.data.val.input_size, batch_size=32,
                                        divisible=32, batch_sizes=batch_sizes)

            self.batch_size = ((self.batch_size + 32 - 1) // 32) * 32

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=False,
        )
        evaluator = build_evaluator(self.cfg.evaluator, val_dataset, logger=self.logger)

        self._info("Creating task...", verbose)

        self.task = TrainingTask(self.cfg, self.model, evaluator)

        if self.cfg.device.gpu_ids == -1 or self.device == "cpu":
            accelerator, devices, precision = ("cpu", None, cfg.device.precision)
        else:
            accelerator, devices, precision = ("gpu", cfg.device.gpu_ids, cfg.device.precision)
            assert len(devices) == 1, ("Distributed learning is not implemented, please use only"
                                       " one gpu device.")

        # if self.cfg.device.gpu_ids == -1 or self.device == "cpu":
        #     gpu_ids, precision = (None, self.cfg.device.precision)
        # else:
        #     gpu_ids, precision = (self.cfg.device.gpu_ids, self.cfg.device.precision)
        #     assert len(gpu_ids) == 1, ("Distributed learning is not implemented, please use only"
        #                                " one gpu device.")

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=self.cfg.log.interval,
            num_sanity_val_steps=0,
            logger=self.logger,
            precision=precision,
        )
        # trainer = pl.Trainer(
        #     default_root_dir=save_dir,
        #     gpus=gpu_ids,
        #     accelerator=None,
        #     log_every_n_steps=self.cfg.log.interval,
        #     num_sanity_val_steps=0,
        #     logger=self.logger,
        #     precision=precision,
        # )
        self._info("Starting testing...", verbose)

        test_results = (verbose or logging)
        return trainer.test(self.task, val_dataloader, verbose=test_results)

    def infer(self, input, conf_threshold=0.35, iou_threshold=0.6, nms_max_num=100, hf=False, dynamic=True, ch_l=False):
        """
        Performs inference
        :param input: input image to perform inference on
        :type input: opendr.data.Image
        :param conf_threshold: confidence threshold
        :type conf_threshold: float, optional
        :param iou_threshold: iou threshold
        :type iou_threshold: float, optional
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int, optional
        :param hf: determines if half precision is used.
        :type hf: bool, optional
        :param dynamic: determines if the model runs with dynamic input. If it is set to False, Nanodet Plus head with
         legacy_post_process=False runs faster. Otherwise, the inference is not affected.
        :type dynamic: bool, optional
        :param ch_l: determines if inference will run in channel-last format.
        :type ch_l: bool, optional
        :return: list of bounding boxes of last image of input or last frame of the video
        :rtype: opendr.engine.target.BoundingBoxList
        """

        ch_l = ch_l and self.jit_model is not None
        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                                       iou_thresh=iou_threshold, nms_max_num=nms_max_num, hf=hf, dynamic=dynamic,
                                       ch_l=ch_l)

        if not isinstance(input, Image):
            input = Image(input)
        _input = input.opencv()

        _input, *metadata = self.predictor.preprocessing(_input)

        if self.trt_model:
            if self.jit_model or self.ort_session:
                warnings.warn(
                    "Warning: More than one optimization types are initialized, "
                    "inference will run in TensorRT mode by default.\n"
                    "To run in a specific optimization please delete the self.ort_session, self.jit_model or "
                    "self.trt_model like: detector.ort_session = None.")
            preds = self.trt_model(_input)
        elif self.jit_model:
            if self.ort_session:
                warnings.warn(
                    "Warning: Both JIT and ONNX models are initialized, inference will run in JIT mode by default.\n"
                    "To run in JIT please delete the self.ort_session like: detector.ort_session = None.")
            self.jit_model = self.jit_model.half() if hf else self.jit_model.float()

            preds = self.jit_model(_input, *metadata)
        elif self.ort_session:
            preds = self.ort_session.run(['output'], {'data': _input.cpu().numpy()})
            preds = torch.from_numpy(preds[0]).to(self.device, torch.half if hf else torch.float32)
        else:
            self.predictor.model = self.predictor.model.half() if hf else self.predictor.model.float()
            preds = self.predictor(_input)
        res = self.predictor.postprocessing(preds, _input, *metadata)

        bounding_boxes = []
        if res.numel() != 0:
            for box in res:
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

    def benchmark(self, repetitions=1000, warmup=100, conf_threshold=0.35, iou_threshold=0.6, nms_max_num=100,
                  hf=False, ch_l=False, fuse=False, dataset=None, zeros=True):
        """
        Performs inference
        :param repetitions: input image to perform inference on
        :type repetitions: opendr.data.Image
        :param warmup: confidence threshold
        :type warmup: float, optional
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int
        """
        if dataset:
            self.benchmark_dataset(
                dataset,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                nms_max_num=nms_max_num,
                hf=hf, ch_l=ch_l, fuse=fuse,
            )
            return
        import numpy as np

        dummy_input = self.__cv_dumy_input()
        self.model.float()
        self.cfg.defrost()
        self.cfg.model.arch.ch_l = ch_l
        self.cfg.model.arch.fuse = fuse
        self.cfg.freeze()

        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                                       iou_thresh=iou_threshold, nms_max_num=nms_max_num, hf=hf)

        def bench_loop(input, function, repetitions, warmup, sing_inputs=True, onnx_fun=False):
            import numpy as np
            output = None
            timings = np.zeros((repetitions, 1))
            torch.cuda.synchronize()
            if onnx_fun:
                for run in range(warmup):
                    output = function(['output'], {'data': input})
                torch.cuda.synchronize()
                for run in range(repetitions):
                    starter = time.perf_counter()
                    output = function(['output'], {'data': input})
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - starter
                    timings[run] = dt
                return output, timings
            if sing_inputs:
                for run in range(warmup):
                    output = function(input)
                torch.cuda.synchronize()
                for run in range(repetitions):
                    starter = time.perf_counter()
                    output = function(input)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - starter
                    timings[run] = dt
                return output, timings
            for run in range(warmup):
                output = function(*input)
            torch.cuda.synchronize()
            for run in range(repetitions):
                starter = time.perf_counter()
                output = function(*input)
                torch.cuda.synchronize()
                dt = time.perf_counter() - starter
                timings[run] = dt
            return output, timings

        def bench_loop2(input, metadatab, function1, function2, repetitions, warmup, onnx_fun=False):
            import numpy as np
            timings = np.zeros((repetitions, 1))
            torch.cuda.synchronize()
            if onnx_fun:
                for run in range(warmup):
                    output1 = function1(['output'], {'data': input})
                    output2 = function2(torch.from_numpy(output1[0]), input, *metadatab)
                torch.cuda.synchronize()
                for run in range(repetitions):
                    starter = time.perf_counter()
                    output1 = function1(['output'], {'data': input})
                    output2 = function2(torch.from_numpy(output1[0]), input, *metadatab)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - starter
                    timings[run] = dt
                return output2, timings
            for run in range(warmup):
                output1 = function1(input)
                output2 = function2(output1, input, *metadatab)
            torch.cuda.synchronize()
            for run in range(repetitions):
                starter = time.perf_counter()
                output1 = function1(input)
                output2 = function2(output1, input, *metadatab)
                torch.cuda.synchronize()
                dt = time.perf_counter() - starter
                timings[run] = dt
            return output2, timings

        # Preprocess measurement
        (_input, *metadata), preprocess_timings = bench_loop(dummy_input, self.predictor.preprocessing, 10,
                                                             1, sing_inputs=False)
        # Onnx measurements
        onnx_infer_timings = None
        if self.ort_session:
            # Inference
            preds, onnx_infer_timings = bench_loop(dummy_input, self.ort_session.run, repetitions, warmup,
                                                   sing_inputs=False, onnx_fun=True)

        # Jit measurements
        jit_infer_timings = None
        if self.jit_model:
            if hf:
                self.jit_model = self.jit_model.half()
            try:
                self.jit_model = torch.jit.optimize_for_inference(self.jit_model)
            except:
                print("")
            post_out_jit, jit_infer_timings = bench_loop((_input, *metadata), self.jit_model, repetitions, warmup,
                                                         sing_inputs=False)

        # trt measurements
        trt_infer_timings = None
        trt_post_timings = None
        trt_infer_post_timings = None
        if self.trt_model:
            preds_trt, trt_infer_timings = bench_loop(_input, self.trt_model, repetitions, warmup, sing_inputs=True)
            post_out_trt, trt_post_timings = bench_loop((preds_trt, _input, *metadata), self.jit_postprocessing,
                                                        repetitions, warmup, sing_inputs=False)

            post_out_trt, trt_infer_post_timings = bench_loop2(_input, metadatab=metadata, function1=self.trt_model,
                                                               function2=self.jit_postprocessing,
                                                               repetitions=repetitions, warmup=warmup)

        # Original Python measurements
        if hf:
            self.predictor.model = self.predictor.model.half()
        preds, infer_timings = bench_loop(_input, self.predictor, repetitions, warmup, sing_inputs=True)

        # Post-processing measurements
        post_out, post_timings = bench_loop((preds, _input, *metadata), self.predictor.postprocessing, repetitions, warmup,
                                            sing_inputs=False)
        post_out, infer_post_timings = bench_loop2(_input, metadatab=metadata, function1=self.predictor,
                                                   function2=self.predictor.postprocessing, repetitions=repetitions,
                                                   warmup=warmup)

        # Measure std and mean of times
        std_preprocess_timings = np.std(preprocess_timings)
        std_infer_timings = np.std(infer_timings)
        std_post_timings = np.std(post_timings)
        std_infer_post_timings = np.std(infer_post_timings)

        mean_preprocess_timings = np.mean(preprocess_timings)
        mean_infer_timings = np.mean(infer_timings)
        mean_post_timings = np.mean(post_timings)
        mean_infer_post_timings = np.mean(infer_post_timings)

        if self.jit_model:
            mean_jit_infer_timings = np.mean(jit_infer_timings)
        if self.trt_model:
            mean_trt_infer_timings = np.mean(trt_infer_timings)
            mean_trt_post_timings = np.mean(trt_post_timings)
            mean_trt_infer_post_timings = np.mean(trt_infer_post_timings)
        if self.ort_session:
            mean_onnx_infer_timings = np.mean(onnx_infer_timings)

        # mean times to fps, torch measures in milliseconds
        fps_preprocess_timings = 1/mean_preprocess_timings
        fps_infer_timings = 1/mean_infer_timings
        fps_post_timings = 1/mean_post_timings
        fps_ifer_post_timings = 1 / mean_infer_post_timings

        if self.jit_model:
            fps_jit_infer_timings = 1/mean_jit_infer_timings

        if self.trt_model:
            fps_trt_infer_timings = 1 / mean_trt_infer_timings
            fps_trt_postprocessing_timings = 1 / mean_trt_post_timings
            fps_trt_infer_post_timings = 1 / mean_trt_infer_post_timings
        if self.ort_session:
            fps_onnx_infer_timings = 1/mean_onnx_infer_timings
            fps_onnx_infer_post_timings = 1/(mean_onnx_infer_timings + mean_post_timings)

        # Print measurements
        print(f"\n\nMeasure of model: {self.cfg.check_point_name} \nHalf precision: {hf}\nFuse Convs: {fuse}\nChannel last: {ch_l}\nData: {'zeros' if zeros else 'rand'}")
        print(f"\n=== Python measurements === \n"
              f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
              f"infer          fps = {fps_infer_timings} evn/s\n"#)
              f"postprocessing fps = {fps_post_timings} evn/s\n"
              f"infer + postpr fps = {fps_ifer_post_timings} evn/s")
        if self.jit_model:
            print(f"\n\n=== JIT measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer fps = {fps_jit_infer_timings} evn/s")

        if self.trt_model:
            print(f"\n\n=== TRT measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer          fps = {fps_trt_infer_timings} evn/s\n"
                  f"postprocessing fps = {fps_trt_postprocessing_timings} evn/s\n"
                  f"infer + postpr fps = {fps_trt_infer_post_timings} env/s")

        if self.ort_session:
            print(f"\n\n=== ONNX measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer          fps = {fps_onnx_infer_timings} evn/s\n"
                  f"postprocessing fps = {fps_post_timings} evn/s\n"
                  f"infer + postpr fps = {fps_onnx_infer_post_timings} evn/s")

        print(f"\n\n++++++ STD OF TIMES ++++++")
        print(f"std pre: {std_preprocess_timings}")
        print(f"std infer: {std_infer_timings}")
        print(f"std post: {std_post_timings}")
        print(f"std infer post: {std_infer_post_timings}")

        if self.jit_model:
            std_jit_infer_timings = np.std(jit_infer_timings)
            print(f"std JIT infer+post: {std_jit_infer_timings}")

        if self.trt_model:
            std_trt_infer_timings = np.std(trt_infer_timings)
            std_trt_post_timings = np.std(trt_post_timings)
            std_trt_infer_post_timings = np.std(trt_infer_post_timings)
            print(f"std trt infer: {std_trt_infer_timings}\n"
                  f"std trt post : {std_trt_post_timings}\n"
                  f"std trt infer post: {std_trt_infer_post_timings}")

        if self.ort_session:
            std_onnx_infer_timings = np.std(onnx_infer_timings)
            print(f"std ONNX infer: {std_onnx_infer_timings}")

        return

    def benchmark_dataset(self, dataset, conf_threshold=0.35, iou_threshold=0.6, nms_max_num=100, hf=True, ch_l=False, fuse=True):
        """
        Performs inference
        :param nms_max_num: determines the maximum number of bounding boxes that will be retained following the nms.
        :type nms_max_num: int
        """

        import numpy as np
        from tqdm import tqdm

        dummy_input = self.__cv_dumy_input()
        self.model.float()
        self.cfg.defrost()
        self.cfg.model.arch.ch_l = ch_l
        self.cfg.model.arch.fuse = fuse
        self.cfg.freeze()

        if not self.predictor:
            self.predictor = Predictor(self.cfg, self.model, device=self.device, conf_thresh=conf_threshold,
                                       iou_thresh=iou_threshold, nms_max_num=nms_max_num, hf=hf)

        if self.jit_model:
            if hf:
                self.jit_model = self.jit_model.half()
            try:
                self.jit_model = torch.jit.optimize_for_inference(self.jit_model)
            except:
                print("")

        if hf:
            self.predictor.model = self.predictor.model.half()

        def profile(input, function, sing_inputs=True, onnx_fun=False):
            torch.cuda.synchronize()
            if onnx_fun:
                starter = time.perf_counter()
                output = function(['output'], {'data': input})
                torch.cuda.synchronize()
                dt = time.perf_counter() - starter
                return output, dt
            if sing_inputs:
                starter = time.perf_counter()
                output = function(input)
                torch.cuda.synchronize()
                dt = time.perf_counter() - starter
                return output, dt
            starter = time.perf_counter()
            output = function(*input)
            torch.cuda.synchronize()
            dt = time.perf_counter() - starter
            return output, dt

        # Onnx measurements
        onnx_infer_timings = []
        if self.ort_session:
            # warmup
            (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
            _input = _input.cpu().numpy()
            _preds = self.ort_session.run(['output'], {'data': _input})
            _preds = torch.from_numpy(_preds[0]).to(self.device, torch.half if hf else torch.float32)
            _out = self.predictor.postprocessing(_preds, _input, *_metadata)

            # bench
            for (_img, _) in tqdm(dataset, total=len(dataset)):
                _img = _img.opencv()
                (_input, *_metadata), _ = profile(
                    _img, self.predictor.preprocessing
                )
                _input = _input.cpu().numpy()
                _preds, onnx_infer_time = profile(_input, self.ort_session.run, onnx_fun=True)
                _preds = torch.from_numpy(_preds[0]).to(self.device, torch.half if hf else torch.float32)
                onnx_infer_timings.append(onnx_infer_time)
        onnx_infer_timings = np.array(onnx_infer_timings)

        # Jit measurements
        jit_infer_timings = []
        if self.jit_model:
            # warmup
            (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
            _out = self.jit_model(_input, *_metadata)
            (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
            _out = self.jit_model(_input, *_metadata)

            # bench
            for (_img, _) in tqdm(dataset, total=len(dataset)):
                _img = _img.opencv()
                (_input, *_metadata), _ = profile(
                    _img, self.predictor.preprocessing
                )
                _preds, jit_infer_time = profile((_input, *_metadata), self.jit_model, sing_inputs=False)
                jit_infer_timings.append(jit_infer_time)
        jit_infer_timings = np.array(jit_infer_timings)

        # trt measurements
        trt_infer_timings = []
        trt_post_timings = []
        if self.trt_model:
            # warmup
            (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
            _preds = self.trt_model(_input)
            _out = self.predictor.postprocessing(_preds, _input, *_metadata)
            (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
            _preds = self.trt_model(_input)
            _out = self.predictor.postprocessing(_preds, _input, *_metadata)

            # bench
            for (_img, _) in tqdm(dataset, total=len(dataset)):
                _img = _img.opencv()
                (_input, *_metadata), _ = profile(
                    _img, self.predictor.preprocessing
                )
                _preds, trt_infer_time = profile(_input, self.trt_model, sing_inputs=True)
                trt_infer_timings.append(trt_infer_time)

                _out, trt_post_time = profile((_preds, _input, *_metadata), self.predictor.postprocessing, sing_inputs=False)
                trt_post_timings.append(trt_post_time)
        trt_infer_timings = np.array(trt_infer_timings)
        trt_post_timings = np.array(trt_post_timings)


        # Original Python measurements
        preprocess_timings = []
        infer_timings = []
        post_timings = []
        (_input, *_metadata) = self.predictor.preprocessing(dummy_input)
        _preds = self.predictor(_input)
        _out = self.predictor.postprocessing(_preds, _input, *_metadata)
        for (_img, _) in tqdm(dataset, total=len(dataset)):
            _img = _img.opencv()
            (_input, *_metadata), preprocess_time = profile(
                _img, self.predictor.preprocessing
            )
            preprocess_timings.append(preprocess_time)

            _preds, infer_time = profile(_input, self.predictor, sing_inputs=True)
            infer_timings.append(infer_time)

            # Post-processing measurements
            _out, post_time = profile((_preds, _input, *_metadata), self.predictor.postprocessing, sing_inputs=False)
            post_timings.append(post_time)
        preprocess_timings = np.array(preprocess_timings)
        infer_timings = np.array(infer_timings)
        post_timings = np.array(post_timings)
        full_run_timing = infer_timings + post_timings

        if self.trt_model:
            trt_full_run_timings = trt_infer_timings + trt_post_timings
        if self.ort_session:
            onnx_full_run_timings = onnx_infer_timings + post_timings

        # Measure std and mean of times
        std_preprocess_timings = np.std(preprocess_timings)
        std_infer_timings = np.std(infer_timings)
        std_post_timings = np.std(post_timings)

        mean_preprocess_timings = np.mean(preprocess_timings)
        mean_infer_timings = np.mean(infer_timings)
        mean_post_timings = np.mean(post_timings)
        mean_full_infer_timings = np.mean(full_run_timing)

        if self.jit_model:
            mean_jit_infer_timings = np.mean(jit_infer_timings)
        if self.trt_model:
            mean_trt_infer_timings = np.mean(trt_infer_timings)
            mean_trt_post_timings = np.mean(trt_post_timings)
            mean_trt_full_run_timings = np.mean(trt_full_run_timings)
        if self.ort_session:
            mean_onnx_infer_timings = np.mean(onnx_infer_timings)
            mean_onnx_full_run_timings = np.mean(onnx_full_run_timings)

        # mean times to fps, torch measures in milliseconds
        fps_preprocess_timings = 1/mean_preprocess_timings
        fps_infer_timings = 1/mean_infer_timings
        fps_post_timings = 1/mean_post_timings
        fps_full_infer_timings = 1/mean_full_infer_timings

        if self.jit_model:
            fps_jit_infer_timings = 1/mean_jit_infer_timings
        if self.trt_model:
            fps_trt_infer_timings = 1 / mean_trt_infer_timings
            fps_trt_postprocessing_timings = 1 / mean_trt_post_timings
            fps_trt_full_run_timings = 1 / mean_trt_full_run_timings
        if self.ort_session:
            fps_onnx_infer_timings = 1 / mean_onnx_infer_timings
            fps_onnx_full_run_timings = 1 / mean_onnx_full_run_timings

        # Print measurements
        print(f"\n\nMeasure of model: {self.cfg.check_point_name} \nHalf precision: {hf}\nFuse Convs: {fuse}\nChannel last: {ch_l}\nData: Dataset")
        print(f"\n=== Python measurements === \n"
              f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
              f"infer          fps = {fps_infer_timings} evn/s\n"
              f"postprocessing fps = {fps_post_timings} evn/s\n"
              f"infer + postpr fps = {fps_full_infer_timings} evn/s")
        if self.jit_model:
            print(f"\n\n=== JIT measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer fps = {fps_jit_infer_timings} evn/s")

        if self.trt_model:
            print(f"\n\n=== TRT measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer          fps = {fps_trt_infer_timings} evn/s\n"
                  f"postprocessing fps = {fps_trt_postprocessing_timings} evn/s\n"
                  f"infer + postpr fps = {fps_trt_full_run_timings} evn/s")

        if self.ort_session:
            print(f"\n\n=== ONNX measurements === \n"
                  f"preprocessing  fps = {fps_preprocess_timings} evn/s\n"
                  f"infer          fps = {fps_onnx_infer_timings} evn/s\n"
                  f"postprocessing fps = {fps_post_timings} evn/s\n"
                  f"infer + postpr fps = {fps_onnx_full_run_timings} evn/s")

        print(f"\n\n++++++ STD OF TIMES ++++++")
        print(f"std pre: {std_preprocess_timings}")
        print(f"std infer: {std_infer_timings}")
        print(f"std post: {std_post_timings}")

        if self.jit_model:
            std_jit_infer_timings = np.std(jit_infer_timings)
            print(f"std JIT infer+post: {std_jit_infer_timings}")

        if self.trt_model:
            std_trt_infer_timings = np.std(trt_infer_timings)
            std_trt_post_timings = np.std(trt_post_timings)
            print(f"std trt infer: {std_trt_infer_timings}\n"
                  f"std trt post : {std_trt_post_timings}")

        if self.ort_session:
            std_onnx_infer_timings = np.std(onnx_infer_timings)
            print(f"std ONNX infer: {std_onnx_infer_timings}")

        return

    def _info(self, msg, verbose=True):
        if self.logger and verbose:
            self.logger.info(msg)
        elif verbose:
            print(msg)
