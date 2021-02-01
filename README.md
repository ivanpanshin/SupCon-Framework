

# SupCon-Framework
<p align="center"><img src="https://github.com/ivanpanshin/SupCon-Framework/blob/main/images/logo.png?raw=true" width="800"></p>


The repo is an implementation of Supervised Contrastive Learning. It's based on another [implementation](https://github.com/HobbitLong/SupContrast), but with several differencies: 
- Fixed bugs (incorrect ResNet implementations, which leads to a very small max batch size), 
- Offers a lot of additional functionality (first of all, rich validation). 

To be more precise, in this implementations you will find:
- Augmentations with [albumentations](https://github.com/albumentations-team/albumentations)
- Hyperparameters are moved to .yml configs
- [t-SNE](https://github.com/DmitryUlyanov/Multicore-TSNE) visualizations
- 2-step validation (for features before and after the projection head) using metrics like AMI, NMI, mAP, precision_at_1, etc with [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning).
- [Exponential Moving Average](https://github.com/fadel/pytorch_ema) for a more stable training, and Stochastic Moving Average for a better generalization and just overall performance.
- Automatic Mixed Precision (torch version) training in order to be able to train with a bigger batch size (roughly by a factor of 2).
- LabelSmoothing loss, and [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) for the second stage of the training (FC).
- TensorBoard logs, checkpoints


## Install

1. Clone the repo:
```
git clone https://github.com/ivanpanshin/SupCon-Framework
```

2. Create a clean virtual environment 
```
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
````
python -m pip install –upgrade pip
pip install -r requirements.txt
````

## Training

In order to execute Cifar10 training run:
```
    python train.py --config_name configs/train/train_supcon_resnet18_cifar10_stage1.yml
    python swa.py --config_name configs/train/swa_supcon_resnet18_cifar100_stage1.yml
    python train.py --config_name configs/train/train_supcon_resnet18_cifar10_stage2.yml
    python swa.py --config_name configs/train/swa_supcon_resnet18_cifar100_stage2.yml
```

The process of training Cifar100 is exactly the same, just change config names from *cifar10* to *cifar100*. 

## Visualizations 

This repo is supplied with t-SNE visualizations so that you can check embeddings you get after the training. Check `t-SNE.ipynb` for details. 

<p align="center"><img src="https://github.com/ivanpanshin/SupCon-Framework/blob/main/images/t-SNE-cifar10.png?raw=true" width="700"></p>

Those are t-SNE visualizations for Cifar10 for validation and train with SupCon (top), and validation and train with CE (bottom).

## FAQ

- Q: What hyperparameters I should try to change? 

  A: First of all, learning rate. Second of all, try to change the augmentation policy. SupCon is build around "cropping + color jittering" scheme, so you can try changing the cropping size or the intensity of jittering. Check `tools.utils.build_transforms` for that.
- Q: What backbone and batch size should I use? 
  
  A: This is quite simple. Take the biggest backbone you can, and after that take the highest batch size your GPU can offer. The reason for that: SupCon is more prone (than regular classification training with CE/LabelSmoothing/etc) to improving with stronger backbones. Moverover, it has a property of explicit hard positive and negative mining. It means that the higher the batch size - the more difficult and helpful samples you supply to your model.   
  
- Q: Do I need the second stage of the training? 

  A: Not necessarily. You can do classification based only on embeddings. In order to do that compute embeddings for the train set, and at inference time do the following: take a sample, compute its embedding, take the closest one from the training, take its class. To make this fast and efficient, you something like [faiss](https://github.com/facebookresearch/faiss) for similarity search. Note that this is actually how validation is done in this repo. Moveover, during training you will see a metric `precision_at_1`. This is actually just accuracy based solely on embeddings. 
 
- Q: Should I use AMP?

  A: If your GPU has tensor cores (like 2080Ti) - yes. If it doesn't (like 1080Ti) - check the speed with AMP and without. If the speed dropped slightly (or even increased by a bit) - use it, since SupCon works better with bigger batch sizes. 
  
- Q: How should I use EMA? 

  A: You only need to choose the `ema_decay_per_epoch` parameter in the config. The heuristic is fairly simple. If your dataset is big, then something as small as 0.3 will do just fine. And as your dataset gets smaller, you can increase `ema_decay_per_epoch`. Thanks to bonlime for this idea. I advice you to check his great [pytorch tools](https://github.com/bonlime/pytorch-tools) repo, it's a hidden gem. 
 
