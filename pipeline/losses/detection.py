import torch
import torch.nn as nn
from pipeline.modules.cuda import device
from pipeline.metrics.iou import box_iou


def get_loss(config: str, **kwargs):
    if config == 'default':
        return DefaultDetectorLoss(**kwargs)
    elif config == 'best_detector':
        return BestDetectorEverLoss(**kwargs)
    else:
        raise BaseException('Loss not supported')


# User's Custom Losses

class DefaultDetectorLoss(nn.Module):
    def __init__(self, class_weight=None):
        super(DefaultDetectorLoss, self).__init__()
        self.coordinate_loss = nn.BCELoss()
        self.bbox_size_loss = nn.L1Loss()
        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weight)

    def __call__(self, bbox_, cls_, bbox, cls):
        l_coords = self.coordinate_loss(bbox_[:, :2], bbox[:, :2])
        l_bbox_size = self.bbox_size_loss(torch.log(bbox_[:, 2:4]), torch.log(bbox[:, 2:4]))
        l_class = self.classification_loss(cls_, cls.long() - 1)
        # -1 because cats' class is 1, dogs' class is 2 and 0 not used
        total = l_coords + l_bbox_size + l_class
        return total


class BestDetectorEverLoss(nn.Module):
    """
    Loss function for detector with 7x7x3 anchor boxes
    """
    def __init__(self, class_weight=None):
        super(BestDetectorEverLoss, self).__init__()

        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float).to(device)
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weight)

        self.probability_loss = nn.BCELoss()
        self.coordinate_loss = nn.BCELoss(reduction='sum')
        self.bbox_size_loss = nn.L1Loss(reduction='sum')

    def __call__(self, bbox_, cls_, bbox, cls):
        # Classification loss
        total_loss = self.classification_loss(cls_, cls.long() - 1)

        # Calculate IoU for anchors
        probs = bbox[:, :1]
        n = probs.shape[0]
        d = probs.shape[3]
        m = probs.view(n, -1).argmax(1).view(-1, 1)
        max_indices = torch.cat((m // d, m % d), dim=1)

        target_prob = torch.zeros((bbox.shape[0], 3, bbox.shape[2], bbox.shape[3])).to(device)

        for batch, (i, j) in enumerate(max_indices):
            target_bbox = bbox[batch:batch + 1, 1:, i, j].clone()
            target_bbox[0, 0] = (target_bbox[0, 0] + j) / 7.
            target_bbox[0, 1] = (target_bbox[0, 1] + i) / 7.

            anchorA_bbox = bbox_[batch:batch + 1, 1:5, i, j].clone()
            anchorA_bbox[0, 0] = (anchorA_bbox[0, 0] + j) / 7.
            anchorA_bbox[0, 1] = (anchorA_bbox[0, 1] + i) / 7.

            anchorB_bbox = bbox_[batch:batch + 1, 6:10, i, j].clone()
            anchorB_bbox[0, 0] = (anchorB_bbox[0, 0] + j) / 7.
            anchorB_bbox[0, 1] = (anchorB_bbox[0, 1] + i) / 7.

            anchorC_bbox = bbox_[batch:batch + 1, 11:15, i, j].clone()
            anchorC_bbox[0, 0] = (anchorC_bbox[0, 0] + j) / 7.
            anchorC_bbox[0, 1] = (anchorC_bbox[0, 1] + i) / 7.

            anchors_iou = torch.tensor([box_iou(anchorA_bbox, target_bbox),
                                        box_iou(anchorB_bbox, target_bbox),
                                        box_iou(anchorC_bbox, target_bbox)])
            best_anchor = torch.argmax(anchors_iou)

            # Probability loss
            target_prob[batch, best_anchor] = bbox[batch, 0]

            # Coordinate & size loss
            total_loss += self.coordinate_loss(bbox_[batch:batch+1, best_anchor*5 + 1: best_anchor*5 + 3, i, j],
                                               bbox[batch:batch+1, 1:3, i, j]) + \
                          self.bbox_size_loss(torch.log(bbox_[batch:batch+1, best_anchor*5 + 3: best_anchor*5 + 5, i, j]),
                                              torch.log(bbox[batch:batch+1, 3:5, i, j]))

        # Probability loss
        total_loss += self.probability_loss(bbox_[:, 0], target_prob[:, 0]) + \
                      self.probability_loss(bbox_[:, 5], target_prob[:, 1]) + \
                      self.probability_loss(bbox_[:, 10], target_prob[:, 2])

        return total_loss
