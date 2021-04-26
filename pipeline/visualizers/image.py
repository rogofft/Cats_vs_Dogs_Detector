import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image_with_bbox(img, cls, bbox):
    """
    Function for visualizing bbox on image
    :param img: PIL Image
    :param bbox: [x1, y1, x2, y2] bbox
    """
    cls_dict = {1:'Cat', 2:'Dog'}

    f, ax = plt.subplots()
    ax.imshow(img)

    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(cls_dict[cls])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
