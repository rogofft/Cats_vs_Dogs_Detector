import pipeline


def make_dataset(data_config: dict):
    # Load data files' paths
    data_file_list = pipeline.get_file_list(data_config['data_file_processing'], data_config['data_path'])

    # Extract metadata
    data_extractor = pipeline.get_data_extractor(data_config['data_extractor']['name'])
    raw_data = data_extractor(data_file_list)

    # Make a dataset
    dataset_class = pipeline.get_dataset_type(data_config['dataset']['name'])

    dataset_transforms = pipeline.get_transform_function(data_config['dataset']['transforms'])
    dataset_augmentations = pipeline.get_augmentations(data_config['dataset']['augmentations'])

    dataset = dataset_class(raw_data, transform=dataset_transforms, augmentation=dataset_augmentations)

    return dataset
