import sys
import os
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(app_path)

import pipeline


if __name__ == '__main__':
    # First argument - path to config file
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise BaseException('Config path not found!')

    model_config, data_config = pipeline.load_config(config_path)

    # Data
    dataset = pipeline.make_dataset(data_config)

    # Make train and val dataloaders
    train_loader, val_loader = pipeline.get_dataloaders(data_config, dataset)

    # Model

    # Get model
    model = pipeline.get_model(model_config)

    # Get train function
    train = pipeline.get_train_function(model_config)

    # Train model
    model, stats, name = train(model, train_loader, val_loader, model_config,
                               save_path=os.path.join(app_path, model_config['save_path']))

    pipeline.view_graphs(stats, save_path=os.path.join(app_path, 'plots', 'train_plots.png'))
