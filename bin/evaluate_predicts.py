import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
from pipeline import box_iou, accuracy

# First argument - path to dataset
try:
    data_path = sys.argv[1]
except IndexError:
    raise BaseException('Config path not found!')

# Second argument - path to predicts
try:
    predicts_path = sys.argv[2]
except IndexError:
    raise BaseException('Model weights file not found!')

filename_list = os.listdir(predicts_path)
filename_list.remove('__netinfo.txt')

# Make tensors for using metrics functions
predict_list = []
data_list = []

for filename in filename_list:

    # Predicts
    with open(os.path.join(predicts_path, filename)) as f:
        data = list(map(int, f.readline().strip().split(' ')))
        predict_list.append(data)

    # Dataset
    with open(os.path.join(data_path, filename)) as f:
        data = list(map(int, f.readline().strip().split(' ')))
        data_list.append(data)

predicts = torch.tensor(predict_list)
data = torch.tensor(data_list)

mIoU = box_iou(predicts[:, 1:], data[:, 1:], convert=False).mean().item()
acc = accuracy(predicts[:, :1], data[:, :1])

with open(os.path.join(predicts_path, '__netinfo.txt'), 'a') as f:
    f.write(f', mIoU {(mIoU*100):.2f}% , classification accuracy {(acc*100):.2f}%')
