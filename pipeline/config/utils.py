import yaml


def load_config(config_path: str):
    # Load Main config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model config
    with open(config['model_config']) as f:
        model_config = yaml.safe_load(f)

    # Load data config
    with open(config['data_config']) as f:
        data_config = yaml.safe_load(f)

    return model_config, data_config
