from torch import randperm, Generator
from torch.utils.data import Subset


def get_train_val_indices(n, val_part=0.2, seed=None):
    if seed:
        generator = Generator().manual_seed(seed)
    else:
        generator = None
    mixed_indices = randperm(n, generator=generator)
    train_count = round((1. - val_part) * len(mixed_indices))
    train_indices, val_indices = mixed_indices[:train_count], mixed_indices[train_count:]
    return train_indices, val_indices


def get_train_val_subsets(dataset, val_part=0.2, seed=None):
    train_indices, val_indices = get_train_val_indices(len(dataset), val_part, seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
