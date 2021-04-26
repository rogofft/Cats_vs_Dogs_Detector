from torch.utils.data import DataLoader
from pipeline.data.generators import get_train_val_subsets


def get_dataloaders(config: dict, dataset):
    if config['dataloaders']['name'] == 'default':
        return make_default_loaders(dataset, **config['dataloaders']['params'])
    else:
        raise BaseException('Data loaders config not found!')


# User's dataloaders config


def make_default_loaders(dataset, validation_part=0.3,
                         train_batch=4,
                         val_batch=1,
                         seed=None):

    val_length = int(validation_part * len(dataset))
    train_length = len(dataset) - val_length

    # Use random seed to reproduce results
    train_dataset, val_dataset = get_train_val_subsets(dataset, validation_part, seed)

    # Make loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch,
                            shuffle=False, pin_memory=True, drop_last=False)
    return train_loader, val_loader
