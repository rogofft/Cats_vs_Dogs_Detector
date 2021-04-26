import torch.optim as optim


def get_optimizer(config: str, *args, **kwargs):
    if config == 'adam':
        return optim.Adam(*args, **kwargs)
    elif config == 'sgd':
        return optim.SGD(*args, **kwargs)
    else:
        raise BaseException('Optimizer not found!')


def get_lr_sheduler(config: str, *args, **kwargs):
    if config == 'step_lr':
        return optim.lr_scheduler.StepLR(*args, **kwargs)
    else:
        raise BaseException('LR_Sheduler not found!')
