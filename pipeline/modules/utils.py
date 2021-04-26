import torch


def get_best_anchor(y_):
    """
    Function gets best bbox relatively to anchor's object probability.
    Batches allowed.
    Used by BestDetectorEver model
    :param y_: anchors with object probabilities
    :return: coordinates of best bbox (xc, yc, w, h)
    """
    # Get max indices for all anchors
    max_indices_A = get_max_indices(y_[:, :1])
    max_indices_B = get_max_indices(y_[:, 5:6])
    max_indices_C = get_max_indices(y_[:, 10:11])

    best_bboxes = list()
    for batch, indices in enumerate(zip(max_indices_A, max_indices_B, max_indices_C)):
        # Get indices of max elements
        (iA, jA), (iB, jB), (iC, jC) = indices
        # Find index of max from them
        max_anchor_num = torch.argmax(torch.tensor([y_[batch, 0, iA, jA],
                                                    y_[batch, 5, iB, jB],
                                                    y_[batch, 10, iC, jC]])).item()

        # Batch in y_: [pA, xcA, ycA, wA, hA, pB, xcB, ycB, wB, hB, pC, xcC, ycC, wC, hC]
        # where pA, xcA... [7, 7] tensors for group A anchors, etc...
        # Get coordinates of best bbox
        bbox = y_[batch:batch + 1, max_anchor_num * 5 + 1:max_anchor_num * 5 + 5,
               indices[max_anchor_num][0], indices[max_anchor_num][1]]
        # Recalculate xc, yc 0...1 value relatively to cell corner to 0...1 value relatively to image corner
        # indices[max_anchor_num] - coordinates of cell in best anchor
        bbox[0, 0] = (bbox[0, 0] + indices[max_anchor_num][1]) / 7.
        bbox[0, 1] = (bbox[0, 1] + indices[max_anchor_num][0]) / 7.
        best_bboxes.append(bbox)
    out_tensor = torch.cat(best_bboxes, dim=0)
    return out_tensor


def get_best_bbox(y):
    """
    Auxiliary function to get bbox from target tensor for rpn network
    :param y: target [batch, 5, 7, 7] tensor
    :return: [batch, 4]
    """
    probs = y[:, :1]
    n = probs.shape[0]
    d = probs.shape[3]
    m = probs.view(n, -1).argmax(1).view(-1, 1)
    max_indices = torch.cat((m // d, m % d), dim=1)

    best_bboxes = list()
    for batch, (i, j) in enumerate(max_indices):
        bbox = y[batch:batch + 1, 1:, i, j]
        bbox[0, 0] = (bbox[0, 0] + j) / 7.
        bbox[0, 1] = (bbox[0, 1] + i) / 7.
        best_bboxes.append(bbox)
    out_tensor = torch.cat(best_bboxes, dim=0)
    return out_tensor


def get_max_indices(tensor):
    """
    This function find indices of maximum element in every batch.
    :param tensor: [batch, 1, 7, 7] tensor
    :return: [batch, 2] tensor
    """
    n = tensor.shape[0]
    d = tensor.shape[3]
    m = tensor.view(n, -1).argmax(1).view(-1, 1)
    max_indices = torch.cat((m // d, m % d), dim=1)
    return max_indices
