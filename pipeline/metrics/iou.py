import torch


def box_area(boxes):
    """
    Computes the area of boxes in batch.
    :boxes  - tensor with box coords.
    :return - tensor with box areas:
    """
    return (boxes[:, 2:3] - boxes[:, 0:1]) * (boxes[:, 3:4] - boxes[:, 1:2])


def convert_coords(boxes):
    """
    Convert coords from [xc, yc, w, h] to [x1, y1, x2, y2]
    """
    return torch.cat([boxes[:, 0:1] - boxes[:, 2:3] / 2.,
                      boxes[:, 1:2] - boxes[:, 3:4] / 2.,
                      boxes[:, 0:1] + boxes[:, 2:3] / 2.,
                      boxes[:, 1:2] + boxes[:, 3:4] / 2.], dim=1)


def box_iou(boxA, boxB, convert=True):
    """
    Computes the IoU of boxes with batch support.
    :boxesA, boxesB - Tensors with box coords.
    :return - Tensor with IoU score by batch.
    """
    assert boxA.shape == boxB.shape
    if convert:
        boxA = convert_coords(boxA)
        boxB = convert_coords(boxB)
    boxI = torch.cat([torch.max(boxA[:, :2], boxB[:, :2]), torch.min(boxA[:, 2:4], boxB[:, 2:4])], dim=1)

    areaA, areaB, areaI = box_area(boxA), box_area(boxB), box_area(boxI)

    iou = areaI / (areaA + areaB - areaI)
    return iou
