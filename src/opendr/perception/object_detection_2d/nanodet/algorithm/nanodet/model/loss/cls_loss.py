import torch
import torch.nn as nn
import torch.nn.functional as F

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss.utils import weighted_loss


@weighted_loss
def cross_entropy_loss(pred, label, label_smoothing=0.0):
    r""" Classic cross entropy.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).
        label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing
            the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a
            uniform distribution. Default: 0.0
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    loss = F.cross_entropy(pred, label, label_smoothing=label_smoothing, reduction="none")
    return loss


@weighted_loss
def hinge_loss(pred, label, margin=1.0):
    r""" Classic hinge entropy.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).
        margin (float, optional) – Has a default value of 1.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    loss = F.hinge_embedding_loss(pred, label, margin, reduction="none")
    return loss


class CrossEntropyLoss(nn.Module):
    r""" Original Cross Entropy Loss from Pytorch

        Args:
            reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed. Default: 'mean'
            label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing
                the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a
                uniform distribution . Default: 0.0
        """
    def __init__(self, reduction='mean', loss_weight=1.0, label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    @torch.jit.unused
    def forward(
            self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * cross_entropy_loss(
            pred, target, weight, self.label_smoothing, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls


class HingeLoss(nn.Module):
    r""" Original Cross Entropy Loss from Pytorch

        Args:
            reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed. Default: 'mean'
            margin (float, optional) – Has a default value of 1.
        """
    def __init__(self, reduction='mean', loss_weight=1.0, margin=1.0):
        super(HingeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.margin = margin

    @torch.jit.unused
    def forward(
            self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * hinge_loss(
            pred, target, weight, self.margin, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
