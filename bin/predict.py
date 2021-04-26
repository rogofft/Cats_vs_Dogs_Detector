import os
import sys
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(app_path)

import torch
from tqdm import tqdm
from PIL import Image
import pipeline
import timeit


def make_predictions(model, data, save_path=os.path.join(app_path, 'predictions')):

    loop = tqdm(enumerate(data), total=len(data), leave=True)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for idx, img_path in loop:
        img = Image.open(img_path)
        with torch.no_grad():
            # get list of bbox prediction and class for every image
            predictions = model.predict([img, ])[0]

            with open(os.path.join(save_path, os.path.basename(img_path)[:-3] + 'txt'), "w") as f:
                f.write(' '.join([str(elem) for elem in predictions]))


def calculate_inference_time():
    setup_code_single = '''
from __main__ import model, data
from PIL import Image
img = [Image.open(path) for path in data[:1]]'''

    test_code = 'model.predict(img)'

    inference_time_single = min(timeit.repeat(stmt=test_code, setup=setup_code_single, repeat=5, number=10)) / 10

    return inference_time_single


if __name__ == '__main__':
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

    model_config, data_config = pipeline.load_config(config_path)

    # Data
    # Load data files' paths
    data_file_list = pipeline.get_file_list(data_config['data_file_processing'], data_config['data_path'])

    # Extract metadata
    data_extractor = pipeline.get_data_extractor(data_config['data_extractor']['name'])
    data = data_extractor(data_file_list)

    # We need only paths to file
    data = [block[0] for block in data]
    # Get validation indices
    train_indices, val_indices = pipeline.get_train_val_indices(len(data),
                                                                data_config['dataloaders']['params']['validation_part'],
                                                                data_config['dataloaders']['params']['seed'])
    data = [data[idx] for idx in val_indices]

    # Model
    # Get model
    model = pipeline.get_model(model_config)
    model.load_state_dict(torch.load(model_dict_path))
    model.eval()

    # Make predictions
    make_predictions(model, data)

    # Calculate inference time
    inference_time = calculate_inference_time()

    # Generate net info file
    with open(os.path.join(app_path, 'predictions', '__netinfo.txt'), 'w') as f:
        f.write(f'{model_config["model"]["name"]}: {(inference_time*1000):.2f} ms, '
                f'{len(train_indices)} train, {len(val_indices)} valid')
