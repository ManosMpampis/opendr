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

import logging
import os
import time

import numpy as np

try:
    from pytorch_lightning.loggers import Logger as LightningLoggerBase
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase

from pytorch_lightning.loggers import TensorBoardLogger
try:
    from pytorch_lightning.loggers.logger import rank_zero_experiment
except ImportError:
    from pytorch_lightning.loggers.base import rank_zero_experiment

from pytorch_lightning.utilities import rank_zero_only
try:
    from lightning_fabric.utilities.cloud_io import get_filesystem
except ImportError:
    from pytorch_lightning.utilities.cloud_io import get_filesystem

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.path import mkdir


class NoneWriter:
    def __init__(self, **kwargs):
        return

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass


class Logger:
    def __init__(self, local_rank, save_dir="./", use_tensorboard=True, profiling=True):
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        fmt = "[%(name)s] [%(asctime)s] %(levelname)s: %(message)s"
        if save_dir is None:
            logging.basicConfig(
                level=logging.INFO,
                filename=None,
                filemode="w",
            )
            self.log_dir = os.path.join("./", "logs")
        else:
            logging.basicConfig(
                level=logging.INFO,
                filename=os.path.join(save_dir, "logs.txt"),
                filemode="w",
            )
            self.log_dir = os.path.join(save_dir, "logs")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                ) from None
            if self.rank < 1:
                logging.info(
                    "Using Tensorboard, logs will be saved in {}".format(self.log_dir)
                )
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def info(self, string):
        if self.rank < 1:
            logging.info(string)

    def warning(self, string):
        if self.rank < 1:
            logging.warning(string)

    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class NanoDetLightningLogger(LightningLoggerBase):
    def __init__(self, save_dir="./", verbose_only=False, **kwargs):
        super().__init__()

        self.verbose_only = verbose_only
        self._name = "NanoDet"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._save_dir = os.path.join(save_dir, f"logs-{self._version}")
        # self.log_dir = os.path.join(save_dir, f"logs-{self._version}")

        self._fs = get_filesystem(save_dir)
        if not self.verbose_only:
            self._fs.makedirs(self._save_dir, exist_ok=True)
            # self._fs.makedirs(self.log_dir, exist_ok=True)
        self._init_logger(verbose_only)

        self._experiment = None
        self._kwargs = kwargs

    @property
    def name(self):
        return self._name

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                "the dependencies to use torch.utils.tensorboard "
                "(applicable to PyTorch 1.1 or higher)"
            ) from None

        if self.verbose_only:
            self._experiment = NoneWriter(log_dir=self._save_dir, **self._kwargs)
        else:
            self._experiment = SummaryWriter(log_dir=self._save_dir, **self._kwargs)
            # self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @property
    def version(self):
        return self._version

    def _init_logger(self, verbose_only=False):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        if verbose_only is False:
            # fh = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
            fh = logging.FileHandler(os.path.join(self._save_dir, "logs.txt"))
            fh.setLevel(logging.INFO)
            # set file formatter
            f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
            file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
            fh.setFormatter(file_formatter)
            # add the handlers to the logger
            self.logger.addHandler(fh)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = "[%(name)s] [%(asctime)s] %(levelname)s: %(message)s"
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(ch)
        self.logger.propagate = False

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def log(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node):
        # with open(os.path.join(self.log_dir, "train_cfg.yml"), "w") as f:
        with open(os.path.join(self._save_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)
        if self.verbose_only is False:
            text = cfg_node.dump()
            for _ in range(10):
                text = text.replace(" -", "-")
            text = text.replace(":\n-", ":[")
            for i in range(10):
                text = text.replace(f"\n- {i}", f", {i}")
            text = text.replace("\n--", "], [")
            text = text.replace("-", "[")
            for i in range(10):
                text = text.replace(f"{i}\n", f"{i}]\n")
            text = text.replace("\n", "\n\t  ")
            self.experiment.add_text("config", f"\t{text}")
        return
    #         dict_cfg = cfg_node.convert_to_dict(cfg_node, [])
    #         text = self.pretty_json(dict_cfg)
    #         for i in range(10):
    #             text = text.replace(f" ]", f"]")
    #         for i in range(10):
    #             text = text.replace(f"\n\t]", f"]")
    #         for _ in range(20):
    #             for i in range(10):
    #                 text = text.replace(f" {i}", f"{i}")
    #         for i in range(10):
    #             text = text.replace(f"\n\t{i}", f"{i}")
    #         text = text.replace('"', '')
    #         text = text.replace("{\n\t  save_dir", "save_dir")
    #         text = text.replace("{", "")
    #         for _ in range(20):
    #             text = text.replace(" }", "}")
    #         text = text.replace("\n\t},", "")
    #         text = text.replace("},", "")
    #         text = text.replace("\n\t}", "")
    #         self.experiment.add_text("config", text)
    #
    # def pretty_json(self, hp):
    #     import json
    #     json_hp = json.dumps(hp, indent=2)
    #     return "".join("\t" + line for line in json_hp.splitlines(True))

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info(f"hyperparams: {params}")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        if self.verbose_only is False:
            for k, v in metrics.items():
                self.experiment.add_scalar("Val_metrics/" + k, v, step)

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        if not self.verbose_only:
            self.experiment.flush()
            self.experiment.close()
        self.save()


class NanoDetLightningTensorboardLogger(TensorBoardLogger):
    def __init__(self, save_dir="./", verbose_only=False, sub_dir=None, **kwargs):

        self.verbose_only = verbose_only
        self._name = "NanoDetTensorboard"
        self._version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._save_dir = os.path.join(save_dir, f"logs-{self._version}")

        self._fs = get_filesystem(save_dir)

        if not self.verbose_only:
            self._fs.makedirs(self._save_dir, exist_ok=True)
        self._init_logger(verbose_only)

        self._kwargs = kwargs

        # saved in /save_dir/name/version{automated}/sub_dir/
        super().__init__(save_dir=self._save_dir, name=self._name, sub_dir=sub_dir)

    @property
    def name(self):
        return self._name

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        return super().experiment

    @property
    def version(self):
        return self._version

    def _init_logger(self, verbose_only=False):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        if verbose_only is False:
            # fh = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
            fh = logging.FileHandler(os.path.join(self._save_dir, "logs.txt"))
            fh.setLevel(logging.INFO)
            # set file formatter
            f_fmt = "[%(name)s][%(asctime)s]%(levelname)s: %(message)s"
            file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
            fh.setFormatter(file_formatter)
            # add the handlers to the logger
            self.logger.addHandler(fh)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = "[%(name)s] [%(asctime)s] %(levelname)s: %(message)s"
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(ch)
        self.logger.propagate = False

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def log(self, string):
        self.logger.info(string)

    @rank_zero_only
    def dump_cfg(self, cfg_node):
        # with open(os.path.join(self.log_dir, "train_cfg.yml"), "w") as f:
        with open(os.path.join(self._save_dir, "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)
        if self.verbose_only is False:
            text = cfg_node.dump()
            for _ in range(10):
                text = text.replace(" -", "-")
            text = text.replace(":\n-", ":[")
            for i in range(10):
                text = text.replace(f"\n- {i}", f", {i}")
            text = text.replace("\n--", "], [")
            text = text.replace("-", "[")
            for i in range(10):
                text = text.replace(f"{i}\n", f"{i}]\n")
            text = text.replace("\n", "\n\t  ")
            self.experiment.add_text("config", f"\t{text}")
        return

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        super().log_hyperparams(params, metrics)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.info(f"Val_metrics: {metrics}")
        for k, v in metrics.items():
            self.experiment.add_scalar("Val_metrics/" + k, v, step)

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        if not self.verbose_only:
            self.experiment.flush()
            self.experiment.close()
        self.save()
