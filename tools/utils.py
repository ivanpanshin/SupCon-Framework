import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import torch.nn.functional as F
import time
import albumentations as A
from albumentations import pytorch as AT
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import os
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, f1_score

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from .backbones import BACKBONES
from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedulers import SCHEDULERS
from .datasets import SupConDataset
from .models import SupConModel


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    writer.add_scalar(tag, message, index)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        return [self.crop_transform(image=x), self.crop_transform(image=x)]


def build_transforms(second_stage):
    if second_stage:
        train_transforms = A.Compose([
            A.Flip(),
            A.Rotate(),
            A.Resize(224, 224),
            A.Normalize(),
            AT.ToTensorV2()
        ])
        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            "valid_transforms": valid_transforms,
        }
    else:
        train_transforms = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.15, 1.)),
            A.Rotate(),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.9),
            A.ToGray(p=0.2),
            A.Normalize(),
            AT.ToTensorV2(),
        ])

        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            'valid_transforms': valid_transforms,
        }

    return transforms_dict


def build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=False):

    train = pd.read_csv("{}/annotations/train.csv".format(data_dir))
    valid = pd.read_csv('{}/annotations/valid.csv'.format(data_dir))

    if second_stage:
        train_features_dataset = SupConDataset(data_dir, "train", transforms['train_transforms'], train, True)
    else:
        # train_features_dataset is used for evaluation -> hence, we don't need TwoCropTransform
        train_features_dataset = SupConDataset(data_dir, "train", transforms['valid_transforms'], train, True)
        train_supcon_dataset = SupConDataset(data_dir, "train", TwoCropTransform(transforms['train_transforms']), train)

    valid_dataset = SupConDataset(data_dir, 'train', transforms['valid_transforms'], valid, True)

    train_supcon_loader = DataLoader(
        train_supcon_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True)
    train_features_loader = DataLoader(
        train_features_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_sizes['valid_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    if second_stage:
        return {'train_features_loader': train_features_loader, 'valid_loader': valid_loader}
    return {'train_supcon_loader': train_supcon_loader, 'train_features_loader': train_features_loader, 'valid_loader': valid_loader}


def build_model(backbone, second_stage=False, num_classes=None, pretrained=None):
    model = SupConModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes)

    if pretrained:
        model.load_state_dict(torch.load(pretrained)['model_state_dict'], strict=False)

    return model


def build_optim(model, optimizer_params, scheduler_params, loss_params):

    if 'params' in loss_params:
        criterion = LOSSES[loss_params['name']](**loss_params['params'])
    else:
        criterion = LOSSES[loss_params['name']]()

    optimizer = OPTIMIZERS[optimizer_params["name"]](model.parameters(), **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler}


def compute_embeddings(loader, model, scaler):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, model.embedding_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)

    for idx, (images, labels) in enumerate(loader):
        images = images.cuda()
        bsz = labels.shape[0]
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
        else:
            embed = model(images)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del images, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)


def train_epoch_constructive(train_loader, model, criterion, optimizer, scaler):
    model.train()
    train_loss = []

    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0]['image'], images[1]['image']], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
                embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(embed, labels)

        else:
            embed = model(images)
            f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
            embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(embed, labels)

        del images, labels, embed
        torch.cuda.empty_cache()

        train_loss.append(loss.item())

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return {'loss': np.mean(train_loss)}


def train_epoch_ce(train_loader, model, criterion, optimizer, scaler):
    model.train()
    train_loss = []

    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
                train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        del data, target, output
        torch.cuda.empty_cache()

    return {"loss": np.mean(train_loss)}


def validation_ce(model, criterion, valid_loader, scaler):
    model.eval()
    val_loss = []
    valid_bs = valid_loader.batch_size
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    y_pred, y_true = np.zeros(len(valid_loader)*valid_loader.batch_size), np.zeros(len(valid_loader)*valid_loader.batch_size)
    correct_samples = 0

    for batch_i, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if criterion:
                        loss = criterion(output, target)
                        val_loss.append(loss.item())
            else:
                output = model(data)
                if criterion:
                    loss = criterion(output, target)
                    val_loss.append(loss.item())

            correct_samples += (
                target.detach().cpu().numpy() == np.argmax(output.detach().cpu().numpy(), axis=1)
            ).sum()
            y_pred[batch_i * valid_bs : (batch_i + 1) * valid_bs] = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_true[batch_i * valid_bs : (batch_i + 1) * valid_bs] = target.detach().cpu().numpy()

            del data, target, output
            torch.cuda.empty_cache()

    valid_loss = np.mean(val_loss)
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    accuracy_score = correct_samples / (len(valid_loader.dataset))

    metrics = {"loss": valid_loss, "accuracy": accuracy_score, "f1_scores": f1_scores, 'f1_score_macro': f1_score_macro}
    return metrics


def validation_constructive(valid_loader, train_loader, model, scaler):
    calculator = AccuracyCalculator(k=1)
    model.eval()

    query_embeddings, query_labels = compute_embeddings(valid_loader, model, scaler)
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model, scaler)

    acc_dict = calculator.get_accuracy(
        query_embeddings,
        reference_embeddings,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source=False
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()

    return acc_dict
