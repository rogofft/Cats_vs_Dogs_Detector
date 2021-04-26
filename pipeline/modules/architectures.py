from pipeline.modules.detectors import BaselineModel, TransferInceptionV3, TransferResnet50, BestDetectorEver
from pipeline.modules.cuda import device


def get_model(config: dict):
    model_name = config['model']['name']
    if model_name == 'baseline':
        model = BaselineModel().float().to(device)
    elif model_name == 'transfer_inception_v3':
        model = TransferInceptionV3().float().to(device)
    elif model_name == 'transfer_resnet50':
        model = TransferResnet50().float().to(device)
    elif model_name == 'best_detector':
        model = BestDetectorEver().float().to(device)
    else:
        raise BaseException('Model architecture not found!')
    return model
