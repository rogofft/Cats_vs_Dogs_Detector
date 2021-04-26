import torch
from PIL import Image


def get_transform_function(config: str):
    if config == 'baseline':
        return load_img_normalize_coord
    elif config == 'rpn_style':
        return load_img_rpn_style
    else:
        return None


# User data transformations:


def load_img_normalize_coord(x: str, y: list):
    """
    This function loads image by path, normalize coordinates xc, yc, w, h to 0...1
    :param x: string path to image
    :param y: coordinates of bbox and class label. Format: [xc, yc, w, h, class_id]
    :return: Pil_image, Tensor_y(same format)
    """
    # load image and convert to tensor
    img = Image.open(x)
    # get image shape
    img_x, img_y = img.size
    # normalize coordinates to 0...1
    meta = torch.as_tensor(y, dtype=torch.float) / torch.as_tensor([img_x, img_y, img_x, img_y, 1.], dtype=torch.float)
    return img, (meta[:4], meta[4])


def load_img_rpn_style(x: str, y: list):
    img, (coords, cls) = load_img_normalize_coord(x, y)

    # Make 7x7 target cells for RPN network training
    # Calculate number of cell, where object is
    cell_x, cell_y = int(torch.floor(coords[:2] * 7)[0]), int(torch.floor(coords[:2] * 7)[1])
    I = torch.zeros((1, 7, 7))
    xc = torch.zeros((1, 7, 7))
    yc = torch.zeros((1, 7, 7))
    w = torch.ones((1, 7, 7))
    h = torch.ones((1, 7, 7))

    # Set probability of object to 1 to cell
    I[0, cell_y, cell_x] = 1.
    # Normalize coordinates of object center according to cell
    xc[0, cell_y, cell_x] = coords[0] * 7 - cell_x
    yc[0, cell_y, cell_x] = coords[1] * 7 - cell_y
    # Set object size to cell
    w[0, cell_y, cell_x] = coords[2]
    h[0, cell_y, cell_x] = coords[3]
    # make [5, 7, 7] target cells
    bbox = torch.cat([I, xc, yc, w, h], dim=0)

    return img, (bbox, cls)
