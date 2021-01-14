import os
from collections import OrderedDict
import torch
import argparse
import yaml

from tools import utils

scaler = torch.cuda.amp.GradScaler()

def swa(paths):
    state_dicts = []
    for path in paths:
        state_dicts.append(torch.load(path)["model_state_dict"])

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = sum([state_dict[k] for state_dict in state_dicts]) / len(state_dicts)

    return average_dict


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="configs/train/swa_supcon_resnet18-product-version3_stage1.yml"
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == "__main__":
    hyperparams = parse_config()

    backbone = hyperparams["model"]["backbone"]
    num_classes = hyperparams['model']['num_classes']
    top_k_checkoints = hyperparams['model']['top_k_checkoints']
    amp = hyperparams['train']['amp']
    weights_dir = hyperparams['train']['weights_dir']
    stage = hyperparams['train']['stage']
    data_dir = hyperparams["dataset"]
    batch_sizes = {
        "train_batch_size": hyperparams["dataloaders"]["train_batch_size"],
        'valid_batch_size': hyperparams['dataloaders']['valid_batch_size']
    }
    num_workers = hyperparams["dataloaders"]["num_workers"]

    if not amp: scaler = None

    utils.seed_everything()

    if os.path.exists(os.path.join(weights_dir, "swa")):
        os.remove(os.path.join(weights_dir, "swa"))

    transforms = utils.build_transforms(second_stage=(stage == 'second'))
    loaders = utils.build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == 'second'))
    model = utils.build_model(backbone, second_stage=(stage == 'second'), num_classes=num_classes, ckpt_pretrained=None).cuda()

    list_of_epochs = sorted([int(x.split('epoch')[1]) for x in os.listdir(weights_dir)])
    best_epochs = list_of_epochs[-top_k_checkoints::]
    model_prefix = "epoch"

    checkpoints_paths = ["{}/{}{}".format(weights_dir, model_prefix, epoch) for epoch in best_epochs]
    average_dict = swa(checkpoints_paths)

    torch.save({"model_state_dict": average_dict}, os.path.join(weights_dir, "swa"))
    model.load_state_dict(torch.load(os.path.join(weights_dir, "swa"))['model_state_dict'])

    if stage == 'first':
        valid_metrics = utils.validation_constructive(loaders['valid_loader'], loaders['train_features_loader'], model,
                                                      scaler)
    else:
        valid_metrics = utils.validation_ce(model, None, loaders['valid_loader'], scaler)

    print('swa stage {} validation metrics: {}'.format(stage, valid_metrics))
