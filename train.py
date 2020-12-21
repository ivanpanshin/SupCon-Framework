import argparse
import logging
import os
import time
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from tools import utils

scaler = torch.cuda.amp.GradScaler()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="configs/train/train_supcon_resnet18-product-version3.yml",
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    hyperparams = parse_config()

    backbone = hyperparams["train"]["backbone"]
    pretrained = hyperparams['train']['ckpt_pretrained']
    amp = hyperparams['train']['amp']
    n_epochs = hyperparams["train"]["n_epochs"]
    logging_name = hyperparams['train']['logging_name']
    target_metric = hyperparams['train']['target_metric']
    stage = hyperparams['train']['stage']
    data_dir = hyperparams["dataset"]["data_dir"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = hyperparams["scheduler"]
    criterion_params = hyperparams["criterion"]

    if not amp: scaler = None

    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        'valid_batch_size': hyperparams['dataloaders']['valid_batch_size']
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]

    utils.seed_everything()

    transforms = utils.build_transforms(second_stage=(stage is 'first'))
    loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=(stage is 'first'))
    model = utils.build_model(backbone, second_stage=(stage is 'first'), pretrained=pretrained).cuda()

    optim = utils.build_optim(model, optimizer_params, scheduler_params, criterion_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )
    if logging_name is None:
        logging_name = "stage_{}_model_{}_dataset_{}".format(stage, backbone, data_dir.split("/")[-1])

    os.makedirs(
        "logs/{}".format(logging_name),
        exist_ok=True,
    )

    writer = SummaryWriter("runs/{}".format(logging_name))
    logging_dir = "logs/{}".format(logging_name)
    logging_path = os.path.join(logging_dir, "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    metric_best = 0
    for epoch in range(n_epochs):
        utils.add_to_logs(logging, "{}, epoch {}".format(time.ctime(), epoch))

        start_training_time = time.time()
        if stage == 'first':
            train_metrics = utils.train_epoch_constructive(loaders['train_supcon_loader'], model, criterion, optimizer, scaler)
        else:
            train_metrics = utils.train_epoch_ce(loaders['train_loader'], model, criterion, optimizer, scaler)
        end_training_time = time.time()

        start_validation_time = time.time()
        if stage == 'first':
            valid_metrics = utils.validation_constructive(loaders['valid_loader'], loaders['train_features_loader'], model, scaler)
        else:
            valid_metrics = utils.validation_ce(model, criterion, loaders['valid_loader'], scaler)
        end_validation_time = time.time()

        print('epoch {}, train time {:.2f} valid time {:.2f} train loss {:.2f} valid acc dict {}'.format(epoch,
                                                                                                         end_training_time - start_training_time,
                                                                                                         end_validation_time - start_validation_time,
                                                                                                         train_metrics['loss'], valid_metrics))

        utils.add_to_tensorboard_logs(writer, train_metrics['loss'], "Loss/train", epoch)

        utils.add_to_logs(
            logging,
            "Epoch {}, train loss: {:.4f} valid metrics: {}".format(
                epoch,
                train_metrics['loss'],
                valid_metrics
            ),
        )

        if valid_metrics[target_metric] > metric_best:
            utils.add_to_logs(
                logging,
                "{} increased ({:.6f} --> {:.6f}).  Saving model ...".format(target_metric,
                    metric_best, valid_metrics[target_metric]
                ),
            )

            os.makedirs(
                "weights/{}".format(logging_name),
                exist_ok=True,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "weights/{}/epoch{}".format(
                    logging_name, epoch
                ),
            )
            metric_best = valid_metrics[target_metric]

        scheduler.step()

    writer.close()