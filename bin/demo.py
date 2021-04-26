import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pipeline
from torch import load
from PIL import Image


# First argument - path to config file
try:
    config_path = sys.argv[1]
except IndexError:
    raise BaseException('Config path not found!')

# Second argument - path to model weigths file
try:
    model_dict_path = sys.argv[2]
except IndexError:
    raise BaseException('Model weights file not found!')

# Third argument - path to image file
try:
    img_path = sys.argv[3]
except IndexError:
    raise BaseException('Image file not found!')

# Get model configuration
model_config, _ = pipeline.load_config(config_path)

# Model
model = pipeline.get_model(model_config)
model.load_state_dict(load(model_dict_path))
model.eval()

# Image
img = Image.open(img_path)

# Predict
predict = model.predict([img, ])[0]
cls, bbox = predict[0], predict[1:]

# Calculate bboxes & visualize
pipeline.show_image_with_bbox(img, cls, bbox)
