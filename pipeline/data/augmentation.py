import torchvision.transforms as T


def get_augmentations(config: str):
    if config == 'baseline':
        return BaseAugmentation()
    elif config == 'transfer_resnet':
        return ResNetNormalize()
    elif config == 'transfer_inception_v3':
        return InceptionV3Normalize()
    else:
        return None


# User's augmentation presets


class BaseAugmentation:
    def __init__(self):
        self.augs = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __call__(self, *args, **kwargs):
        return self.augs(*args)


class ResNetNormalize:
    def __init__(self):
        self.augs = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args, **kwargs):
        return self.augs(*args)


class InceptionV3Normalize:
    def __init__(self):
        self.augs = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args, **kwargs):
        return self.augs(*args)
