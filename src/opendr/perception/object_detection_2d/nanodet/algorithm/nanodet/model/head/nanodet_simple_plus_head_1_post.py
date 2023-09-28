import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Dict

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util\
    import bbox2distance, distance2bbox, multi_apply
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.data.transform.warp \
    import warp_boxes, scriptable_warp_boxes
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss import DistributionFocalLoss,\
    QualityFocalLoss, GIoULoss
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv \
    import Conv, DWConv, ConvQuant, DWConvQuant
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.conv import fuse_modules

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.activation import act_layers

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.init_weights import normal_init
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.module.nms import multiclass_nms, batched_nms
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.assigner.dsl_assigner \
    import DynamicSoftLabelAssigner
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.gfl_head import Integral, reduce_mean


def _load_hparam(model: str):
    from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util import (load_config, cfg)
    import os
    from pathlib import Path

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
    path = Path(__file__).parent.parent.parent.parent.parent / "algorithm" / "config"
    wanted_file = "nanodet_{}.yml".format(model)
    for root, dir, files in os.walk(path):
        if wanted_file in files:
            full_path.append(os.path.join(root, wanted_file))
    assert (len(full_path) == 1), f"You must have only one nanodet_{model}.yaml file in your config folder"
    load_config(cfg, full_path[0])
    return cfg


class SimplifierNanoDetPlusHead_1(nn.Module):
    dynamic = False
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        use_depthwise=True,
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        fork=False,
        quant=False,
        **kwargs
    ):
        super(SimplifierNanoDetPlusHead_1, self).__init__()
        self.fork = fork
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation

        self.center_priors_1 = torch.empty(0)
        self.center_priors_0 = torch.empty(0)
        # for idx in range(len(strides)):
        #     self.register_buffer(f"center_priors_{idx}", torch.empty(0))

        if use_depthwise:
            self.ConvModule = DWConvQuant if quant else DWConv
        else:
            self.ConvModule = ConvQuant if quant else Conv
        # self.ConvModule = DWConv if use_depthwise else Conv

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        try:
            self.loss_qfl = QualityFocalLoss(
                beta=self.loss_cfg.loss_qfl.beta,
                loss_weight=self.loss_cfg.loss_qfl.loss_weight,
                cost_function=self.loss_cfg.loss_qfl.cost_function,
            )
        except AttributeError:
            self.loss_qfl = QualityFocalLoss(
                beta=self.loss_cfg.loss_qfl.beta,
                loss_weight=self.loss_cfg.loss_qfl.loss_weight
            )

        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs0 = self._buid_not_shared_head()
        self.cls_convs1 = self._buid_not_shared_head()

        self.gfl_cls0 = nn.Conv2d(self.feat_channels, self.num_classes + 4 * (self.reg_max + 1), 1, padding=0)
        self.gfl_cls1 = nn.Conv2d(self.feat_channels, self.num_classes + 4 * (self.reg_max + 1), 1, padding=0)

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    k=self.kernel_size,
                    s=1,
                    p=self.kernel_size // 2,
                    act=act_layers(self.activation),
                )
            )
        cls_convs = nn.Sequential(*cls_convs)
        return cls_convs

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            fuse_modules(m)
        return self

    def init_weights(self):
        # for m in self.cls_convs.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        # bias_cls = -4.595
        # for i in range(len(self.strides)):
        #     normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        for m in self.cls_convs0.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.cls_convs1.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        normal_init(self.gfl_cls0, std=0.01, bias=bias_cls)
        normal_init(self.gfl_cls1, std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        self.center_priors_0 = fn(self.center_priors_0)
        self.center_priors_1 = fn(self.center_priors_1)
        return self

    @torch.jit.unused
    def forward(self, feats: List[Tensor]):
        x1 = feats[0]
        x1 = self.cls_convs0(x1)
        x1 = self.gfl_cls0(x1)
        # post 1
        bs, _, ny, nx = x1.shape
        x1 = x1.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        cls, reg = x1.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        project = self.distribution_project(reg)
        if self.dynamic or self.center_priors_0.shape != project.shape:
            self.center_priors_0 = (
                self.get_single_level_center_priors(
                    bs, (ny, nx), self.strides[0], dtype=project.dtype, device=project.device
                )
            )
        dis_preds = project * self.center_priors_0[..., 2, None]
        decoded_bboxes = distance2bbox(self.center_priors_0[..., :2], dis_preds)
        x1 = torch.cat((cls, reg, decoded_bboxes), dim=2)

        x2 = feats[1]
        x2 = self.cls_convs1(x2)
        x2 = self.gfl_cls1(x2)
        # post 2
        bs, _, ny, nx = x2.shape
        x2 = x2.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        cls, reg = x2.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        project = self.distribution_project(reg)
        if self.dynamic or self.center_priors_1.shape != project.shape:
            self.center_priors_1 = (
                self.get_single_level_center_priors(
                    bs, (ny, nx), self.strides[1], dtype=project.dtype, device=project.device
                )
            )
        dis_preds = project * self.center_priors_1[..., 2, None]
        decoded_bboxes = distance2bbox(self.center_priors_1[..., :2], dis_preds)
        x2 = torch.cat((cls, reg, decoded_bboxes), dim=2)
        outputs = torch.cat((x1, x2), dim=1)
        return outputs

    @torch.jit.unused
    def graph_forward(self, feats: List[Tensor]):
        x1 = feats[0]
        x1 = self.cls_convs0(x1)
        x1 = self.gfl_cls0(x1)
        # post 1
        bs, _, ny, nx = x1.shape
        x1 = x1.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        cls, reg = x1.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        project = self.distribution_project(reg)
        if self.dynamic or self.center_priors_0.shape != project.shape:
            self.center_priors_0 = (
                self.get_single_level_center_priors(
                    bs, (ny, nx), self.strides[0], dtype=project.dtype, device=project.device
                )
            )

        dis_preds = project * self.center_priors_0[..., 2, None]
        decoded_bboxes = distance2bbox(self.center_priors_0[..., :2], dis_preds)
        x1 = torch.cat((cls.sigmoid(), decoded_bboxes), dim=2)

        x2 = feats[1]
        x2 = self.cls_convs1(x2)
        x2 = self.gfl_cls1(x2)
        # post 2
        bs, _, ny, nx = x2.shape
        x2 = x2.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        cls, reg = x2.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        project = self.distribution_project(reg)
        if self.dynamic or self.center_priors_1.shape != project.shape:
            self.center_priors_1 = (
                self.get_single_level_center_priors(
                    bs, (ny, nx), self.strides[1], dtype=project.dtype, device=project.device
                )
            )
        dis_preds = project * self.center_priors_1[..., 2, None]
        decoded_bboxes = distance2bbox(self.center_priors_1[..., :2], dis_preds)
        x2 = torch.cat((cls.sigmoid(), decoded_bboxes), dim=2)
        outputs = torch.cat((x1, x2), dim=1)
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]

        cls_preds, reg_preds, decoded_bboxes = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1), 4], dim=-1
        )

        center_priors = torch.cat([self.center_priors_0, self.center_priors_1], dim=1)
        if aux_preds is not None:
            # use auxiliary head to assign
            # aux_cls_preds, aux_reg_preds, aux_decoded_bboxes = aux_preds.split(
            #     [self.num_classes, 4 * (self.reg_max + 1), 4], dim=-1
            # )
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                    self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
            self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)
        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0

        assign_result = self.assigner.assign(
            cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = bbox2distance(center_priors[pos_inds, :2],
                                                      pos_gt_bboxes) / center_priors[pos_inds, None, 2]
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
        return (
            labels,
            label_scores,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process_old(self, preds, meta: Dict[str, Tensor], mode: str = "infer", conf_thresh: float = 0.3,
                     iou_thresh: float = 0.6, nms_max_num: int = 100):
        """Prediction results postprocessing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
            mode (str): Determines if it uses batches and numpy or tensors for scripting.
            conf_thresh (float): Determines the confidence threshold.
            iou_thresh (float): Determines the iou threshold.
            nms_max_num (int): Determines the maximum number of bounding boxes that will be retained following the nms.
        """
        if mode == "eval" and not torch.jit.is_scripting():
            # Inference do not use batches and tries to have
            # tensors exclusively for better optimization during scripting.
            return self._eval_post_process(preds, meta)

        cls_scores, bboxes = preds.split(
            (self.num_classes, 4), dim=-1
        )

        labels = torch.arange(self.num_classes, device=self.center_priors_0.device).unsqueeze(1).unsqueeze(1)
        det_results = []
        for i in range(cls_scores.shape[0]):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = cls_scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)

            det_bboxes, det_labels = multiclass_nms(bbox, score, score_thr=conf_thresh,
                                                    nms_cfg=dict(iou_threshold=iou_thresh), max_num=nms_max_num)

            det_bboxes[:, :4] = scriptable_warp_boxes(
                det_bboxes[:, :4],
                torch.linalg.inv(meta["warp_matrix"][i]), meta["width"][i], meta["height"][i]
            )
            for i in range(self.num_classes):
                inds = det_labels == i
                class_det_bboxes = det_bboxes[inds]
                if class_det_bboxes.shape[0] != 0:
                    det = torch.cat(
                        (
                            class_det_bboxes,
                            labels[i].repeat(class_det_bboxes.shape[0], 1)
                        ),
                        dim=1,
                    )
                    det_results.append(det)
        return det_results

    def post_process(self, preds, meta: Dict[str, Tensor], mode: str = "infer", conf_thresh: float = 0.3,
                     iou_thresh: float = 0.6, nms_max_num: int = 100):
        """Prediction results postprocessing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
            mode (str): Determines if it uses batches and numpy or tensors for scripting.
            conf_thresh (float): Determines the confidence threshold.
            iou_thresh (float): Determines the iou threshold.
            nms_max_num (int): Determines the maximum number of bounding boxes that will be retained following the nms.
        """
        if mode == "eval" and not torch.jit.is_scripting():
            # Inference do not use batches and tries to have
            # tensors exclusively for better optimization during scripting.
            return self._eval_post_process(preds, meta)

        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        bs = preds.shape[0]
        xc = (preds[..., :self.num_classes] > conf_thresh).any(dim=-1)
        det_results = [torch.zeros((0, 6), device=preds.device, dtype=preds.dtype)] * bs
        for i, (pred) in enumerate(preds):
            valid_mask = xc[i]
            pred = pred[valid_mask]
            if not pred.shape[0]:
                continue

            max_scores, labels = torch.max(pred[:, :self.num_classes], dim=1)
            keep = max_scores.argsort(descending=True)[:max_nms]
            pred = pred[keep]  # sort by confidence and remove excess boxes
            labels = labels[keep]
            bboxes = pred[:, self.num_classes:]
            cls_scores = max_scores[keep]
            # cls_scores, bboxes = pred.split((self.num_classes, 4), dim=-1)

            det_bboxes, keep = batched_nms(bboxes, cls_scores, labels, nms_cfg=dict(iou_threshold=iou_thresh, nms_max_num=float(nms_max_num)))
            det_labels = labels[keep]
            det_bboxes[:, :4] = scriptable_warp_boxes(
                det_bboxes[:, :4],
                torch.linalg.inv(meta["warp_matrix"][i]), meta["width"][i], meta["height"][i]
            )
            det = torch.cat((det_bboxes, det_labels[:, None]), dim=1)
            det_results[i] = det
        return det_results

    def _eval_post_process(self, preds, meta):
        # TODO: get_bboxes must run in loop and can be used only tensors for better performance
        cls_scores, _, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1), 4], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds)
        det_results = {}
        warp_matrixes = meta["warp_matrix"]
        img_heights = (
            meta["height"].cpu().numpy()
            if isinstance(meta["height"], torch.Tensor)
            else meta["height"]
        )
        img_widths = (
            meta["width"].cpu().numpy()
            if isinstance(meta["width"], torch.Tensor)
            else meta["width"]
        )
        img_ids = (
            meta["id"].cpu().numpy()
            if isinstance(meta["id"], torch.Tensor)
            else meta["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
                result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def get_bboxes(self, cls_preds, bboxes, conf_threshold: float = 0.05,
                   iou_threshold: float = 0.6, nms_max_num: int = 100):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            bboxes (Tensor): Shape (num_imgs, num_points, 4).
            conf_threshold (float): Determines the confident threshold.
            iou_threshold (float): Determines the iou threshold in nms.
            nms_max_num (int): Determines the maximum number of bounding boxes that will be retained following the nms.
        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        cls_preds = cls_preds.sigmoid()

        # add a dummy background class at the end of all labels
        result_list = []
        for i in range(cls_preds.shape[0]):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = cls_preds[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=conf_threshold,
                nms_cfg=dict(iou_threshold=iou_threshold),
                max_num=nms_max_num,
            )
            result_list.append(results)
        return result_list

    def get_single_level_center_priors(
            self,
            batch_size: int,
            featmap_size: Tuple[int, int],
            stride: int,
            dtype: torch.dtype,
            device: torch.device,
            flatten: bool = True
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
            flatten (bool): flatten the x and y tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        if torch.jit.is_scripting() or not torch.__version__[:4] == "1.13":
            y, x = torch.meshgrid(y_range, x_range)
        else:
            y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        if flatten:
            y = y.flatten()
            x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


if __name__ == '__main__':
    import copy
    cfg = _load_hparam("test")
    model_cfg = copy.deepcopy(cfg.model)
    head_cfg = copy.deepcopy(model_cfg.arch.head)
    name = head_cfg.pop("name")
    assert name == "SimplifierNanoDetPlusHead_1"

    model = SimplifierNanoDetPlusHead_1(
        **head_cfg
    ).eval()

    print(model)
    print("=========================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("=========================================")
    model.fuse()
    print(model)
    print("=========================================")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("=========================================")
