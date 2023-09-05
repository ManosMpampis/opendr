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

import copy
import json
import os
import warnings
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.batch_process import stack_batch_img
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util\
    import convert_avg_params, gather_results, mkdir, rank_filter
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.check_point import save_model_state
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.weight_averager import build_weight_averager

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.optim import build_optimizer


class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, model, evaluator=None, accumulate=1, qat=False):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.accumulate = accumulate
        self.classes = cfg.class_names
        self.model = model
        self.val_losses = {}
        self.qat = qat
        if qat:
            self.model.eval()
            self.model.qconfig = torch.quantization.get_default_qconfig(self.cfg.qat.qconfig)
            self.model.fuse()
            self.model = torch.quantization.prepare_qat(self.model.train())
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        if "weight_averager" in self.cfg.model:
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        preds = self.forward(batch["img"])
        results = self.model.head.post_process(preds, batch, "eval")
        return results

    @rank_filter
    def _save_current_model(self, path, verbose):
        save_model_state(path=path, model=self.model, weight_averager=self.weight_averager, verbose=verbose)

    def save_current_model(self, path, verbose):
        model = self.model
        if self.qat:
            self.model.eval()
            model = torch.quantization.convert(self.model)
        save_model_state(path=path, model=model, weight_averager=self.weight_averager, verbose=verbose)
        if self.qat:
            self.model.train()

    @torch.jit.unused
    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        preds, loss, loss_states = self.model.forward_train(batch)

        if self.qat:
            if (self.global_step + 1) > self.cfg.qat.freeze_quantizer_parameters:
                self.model.apply(torch.quantization.disable_observer)
            if (self.global_step + 1) > self.cfg.qat.freeze_bn:
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # log train losses
        if (self.global_step + 1) % self.cfg.log.interval == 0:
            memory = (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch,
                self.cfg.schedule.total_epochs,
                (self.global_step+1),
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Experimen_Variables/Learning Rate", lr, (self.global_step + 1))
            self.scalar_summary("Experimen_Variables/Epoch", self.current_epoch, (self.global_step + 1))
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    loss_states[loss_name].mean().item(),
                    (self.global_step+1),
                )
            if self.logger:
                self.info(log_msg)

        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # save models in schedule epoches
        if (self.current_epoch + 1) % self.cfg.schedule.val_intervals == 0:
            checkpoint_save_path = os.path.join(self.cfg.save_dir, "checkpoints")
            mkdir(self.local_rank, checkpoint_save_path)
            print("===" * 10)
            print("checkpoint_save_path: {} \n epoch: {}".format(checkpoint_save_path, self.current_epoch))
            print("===" * 10)
            self.trainer.save_checkpoint(
                os.path.join(checkpoint_save_path, "model_iter_{}.ckpt".format(self.current_epoch))
            )
        # self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch)
        else:
            preds, loss, loss_states = self.model.forward_train(batch)

        # zero all losses
        if batch_idx == 0:
            self.val_losses = {}
            for loss_name in loss_states:
                if not (loss_name in self.val_losses):
                    self.val_losses[loss_name] = 0
        # update losses
        for loss_name in loss_states:
            self.val_losses[loss_name] += loss_states[loss_name].mean().item()

        if batch_idx % self.cfg.log.interval == 0:
            memory = (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch,
                self.cfg.schedule.total_epochs,
                (self.current_epoch + 1) * batch_idx,
                batch_idx,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            if self.logger:
                self.info(log_msg)

        if (batch_idx + 1) == sum(self.trainer.num_val_batches):
            memory = (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch,
                self.cfg.schedule.total_epochs,
                self.global_step,
                memory,
                lr,
            )
            for loss_name in self.val_losses:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name]
                )
                self.scalar_summary(
                    "Val_loss/" + loss_name,
                    self.val_losses[loss_name] / sum(self.trainer.num_val_batches),
                    (self.global_step+1),
                )

            if self.logger:
                self.info(log_msg)

        dets = self.model.head.post_process(preds, batch, "eval")
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results, self.device)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir)
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                verbose = True if (self.logger is not None) else False
                # TODO: save only if local_rank is < 0
                # self._save_current_model(self.local_rank, os.path.join(best_save_path, "nanodet_model_state_best.pth"),
                #                          verbose=verbose)
                self.save_current_model(os.path.join(best_save_path, "nanodet_model_state_best.pth"), verbose=verbose)
                metadata = {"model_paths": ["nanodet_model_state_best.pth"], "framework": "pytorch", "format": "pth",
                            "has_data": False,
                            "inference_params": {"input_size": self.cfg.data.val.input_size, "classes": self.classes},
                            "optimized": False, "optimizer_info": {}}
                with open(os.path.join(best_save_path, "nanodet_model_state_best.json"), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)

                txt_path = os.path.join(best_save_path, "eval_results.txt")
                with open(txt_path, "a") as f:
                    f.write("Epoch:{}\n".format(self.current_epoch + 1))
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            if self.logger:
                self.logger.log_metrics(eval_results, (self.global_step+1))
        else:
            if self.logger:
                self.info("Skip val on rank {}".format(self.local_rank))
        return

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results, self.device)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            if self.logger:
                self.info("Skip test on rank {}".format(self.local_rank))
        return

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """

        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)
        # name = optimizer_cfg.pop("name")
        # build_optimizer = getattr(torch.optim, name)
        # optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        # self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)
        # return optimizer

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.current_epoch < self.cfg.schedule.warmup.steps:
            warmup_batches = (self.cfg.schedule.warmup.steps * self.trainer.num_training_batches)
            if self.cfg.schedule.warmup.name == "constant":
                k = self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == "linear":
                k = 1 - (1 - self.trainer.global_step / warmup_batches) *\
                    (1 - self.cfg.schedule.warmup.ratio)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (1 - self.trainer.current_epoch / warmup_batches)
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # update params
        if (batch_idx + 1) % self.accumulate == 0:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
        else:
            optimizer_closure()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def text_summary(self, tag, text, step):
        """
            Write Tensorboard text log.
            Args:
                tag: Name for the tag
                text: Text to record
                step: Step value to record
        """
        if self.logger:
            self.logger.experiment.add_text(tag, text, global_step=step)

    def scalar_summary(self, tag, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            value: Value to record
            step: Step value to record

        """
        # if self.local_rank < 1:
        if self.logger:
            #if not self.logger.verbose_only:
            self.logger.experiment.add_scalar(tag, value, global_step=step)

    def info(self, string):
        # if self.logger:
        self.logger.info(string)

    # ------------Hooks-----------------

    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            if self.logger:
                self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self) -> None:
        self.model.set_epoch(self.current_epoch)

    # def on_train_start(self) -> None:
    #     if self.current_epoch > 0:
    #         self.lr_scheduler.last_epoch = self.current_epoch - 1

    # def on_epoch_start(self):
    #     self.model.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self) -> None:
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                if self.logger:
                    self.logger.info(
                        "Weight averaging is enabled but average state does not"
                        "match the model"
                    )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                if self.logger:
                    self.logger.info("Loaded average state from checkpoint.")
