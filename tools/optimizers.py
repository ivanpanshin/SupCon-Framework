import torch.optim as optim
import torch_optimizer as jettify_optim


OPTIMIZERS = {
    "Adam": optim.Adam,
    'AdamW': optim.AdamW,
    "SGD": optim.SGD,
    'LookAhead': jettify_optim.Lookahead,
    'Ranger': jettify_optim.Ranger,
    'RAdam': jettify_optim.RAdam,
}