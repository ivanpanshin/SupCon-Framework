import torch.optim as optim


SCHEDULERS = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
}

