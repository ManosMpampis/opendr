from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss.gfocal_loss import QualityFocalLoss,\
    DistributionFocalLoss
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss.cls_loss import CrossEntropyLoss,\
    HingeLoss
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.loss.iou_loss import IoULoss, BoundedIoULoss,\
    GIoULoss, DIoULoss, CIoULoss

__all__ = ['QualityFocalLoss', 'DistributionFocalLoss', 'CrossEntropyLoss', 'HingeLoss',
           'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss']
