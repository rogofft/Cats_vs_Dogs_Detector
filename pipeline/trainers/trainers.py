import os
import pipeline
import torch
from datetime import datetime
from tqdm import tqdm
from pipeline import device
from pipeline.modules.utils import get_best_anchor, get_best_bbox


def get_train_function(config: dict):
    name = config['trainer']['name']
    if name == 'default':
        return train
    elif name == 'train_fine_tune':
        return train_fine_tune
    elif name == 'train_fine_tune_inception_v3':
        return train_fine_tune_inception_v3
    else:
        raise BaseException('Train function not found')


# User's train functions


def train(model, train_loader, val_loader, config: dict, save_path):
    # Load modules
    criterion = pipeline.get_loss(config['loss']['name'], **config['loss']['params'])
    optimizer = pipeline.get_optimizer(config['optimizer']['name'], model.parameters(), **config['optimizer']['params'])
    lr_sheduler = pipeline.get_lr_sheduler(config['lr_sheduler']['name'], optimizer, **config['lr_sheduler']['params'])
    early_stop = pipeline.get_early_stop_detector(config['early_stop']['name'], **config['early_stop']['params'])

    # accumulate statistics
    train_loss, train_iou, train_acc = [], [], []
    val_loss, val_iou, val_acc = [], [], []

    epochs = config['epochs']

    for epoch in range(epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
            running_loss = 0.
            running_iou = 0.
            running_acc = 0.

            for idx, (img, (bbox, cls)) in loop:
                img, bbox, cls = img.to(device), bbox.to(device), cls.to(device)

                optimizer.zero_grad()

                # Forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    bbox_, cls_ = model(img)
                    loss = criterion(bbox_, cls_, bbox, cls)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.detach().sum().item()
                    running_acc += pipeline.accuracy(torch.argmax(torch.softmax(cls_.detach().cpu(), dim=1), dim=1),
                                                     cls.cpu() - 1)
                    if model.predict_anchors:
                        running_iou += pipeline.box_iou(
                            get_best_anchor(bbox_.detach()).cpu(),
                            get_best_bbox(bbox).cpu()).mean().item()
                    else:
                        running_iou += pipeline.box_iou(bbox_, bbox).mean().item()

                    loop.set_description(f'Epoch [{epoch + 1}/{epochs}] {phase}')
                    loop.set_postfix(loss=running_loss / (idx + 1),
                                     mIoU=running_iou / (idx + 1),
                                     acc=running_acc / (idx + 1))

            epoch_loss = running_loss / len(data_loader)
            epoch_iou = running_iou / len(data_loader)
            epoch_acc = running_acc / len(data_loader)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_iou.append(epoch_iou)
                train_acc.append(epoch_acc)
                lr_sheduler.step()
            else:
                val_loss.append(epoch_loss)
                val_iou.append(epoch_iou)
                val_acc.append(epoch_acc)
                if early_stop.check_for_best_score(epoch_iou):
                    # Save best score model weights
                    torch.save(model.state_dict(),
                               os.path.join(save_path, config['model']['name']) + '.pt')

        if early_stop.check_for_stop():
            break

    name = os.path.join(save_path, datetime.now().strftime("%d-%m-%Y_%H.%M") + '_' + config['model']['name']) + '.pt'
    os.rename(os.path.join(save_path, config['model']['name']) + '.pt', name)

    return model, [train_loss, train_iou, train_acc, val_loss, val_iou, val_acc], name


def train_fine_tune(model, train_loader, val_loader, config: dict, save_path):
    """
    Function for train model with fine tune last layers of backbone
    A very long time but gives more accuracy
    """
    model, stats, name = train(model, train_loader, val_loader, config, save_path)

    # load best model
    model.load_state_dict(torch.load(name))

    # enable grad for backbone last layers
    for param in model.backbone[-2].parameters():
        param.requires_grad = True

    for param in model.backbone[-1].parameters():
        param.requires_grad = True

    # reduce learning rate
    config['optimizer']['params']['lr'] = 0.00005

    # train with new parameters
    model, fine_tune_stats, name = train(model, train_loader, val_loader, config, save_path)

    # add fine tune statistic
    for i in range(len(stats)):
        stats[i] += fine_tune_stats[i]

    # load best model
    model.load_state_dict(torch.load(name))

    # enable grad for another backbone layer
    for param in model.backbone[-3].parameters():
        param.requires_grad = True

    # reduce learning rate
    config['optimizer']['params']['lr'] = 0.00002

    # train with new parameters
    model, fine_tune_stats, name = train(model, train_loader, val_loader, config, save_path)

    # add fine tune statistic
    for i in range(len(stats)):
        stats[i] += fine_tune_stats[i]

    return model, stats, name


def train_fine_tune_inception_v3(model, train_loader, val_loader, config: dict, save_path):
    """
    Function for train model with fine tune last layers of backbone
    A very long time but gives more accuracy
    """
    model, stats, name = train(model, train_loader, val_loader, config, save_path)

    # load best model
    model.load_state_dict(torch.load(name))

    # enable grad for backbone last layers

    for param in model.backbone.Mixed_7a.parameters():
        param.requires_grad = True

    # reduce learning rate
    config['optimizer']['params']['lr'] = 0.00005

    # train with new parameters
    model, stats, name = train(model, train_loader, val_loader, config, save_path)

    return model, stats, name
