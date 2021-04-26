from torch import eq


def accuracy(tensorA, tensorB):
    return eq(tensorA, tensorB).float().mean().item()
