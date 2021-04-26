from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


def get_dataset_type(config: str):
    if config == 'default':
        return DetectionDataset
    else:
        raise BaseException('Dataset type not found!')


# User's Dataset types


class DetectionDataset(Dataset):
    """
    Dataset for training models
    """
    def __init__(self, data, transform=None, augmentation=ToTensor()):
        self.data = data
        self.length = len(data)
        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, idx):
        x, y = self.data[idx][0], self.data[idx][1:]
        if self.transform:
            x, y = self.transform(x, y)
        if self.augmentation:
            x = self.augmentation(x)
        return x, y

    def __len__(self):
        return self.length
