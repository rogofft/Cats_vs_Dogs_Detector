import os
from PIL import Image


def get_file_list(config: dict, *args, **kwargs):
    if config['name'] == 'default':
        return os.listdir(*args)
    if config['name'] == 'baseline':
        # get jpgs
        data_file_list = get_file_list_by_extension(*args, ext=config['data_extension'], **kwargs)
        # remove grayscale imgs
        remove_grayscale_imgs(data_file_list)
        return data_file_list


# User's data path processing


def get_file_list_by_extension(path: str, ext: str) -> list:
    return [os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith('.' + ext)]


def remove_grayscale_imgs(file_list: list):
    for img_path in file_list:
        if Image.open(img_path).mode != 'RGB':
            file_list.remove(img_path)
