from torch.utils.data import Dataset
import torchvision
import os
import cv2


class SupConDataset(torchvision.datasets.CIFAR10):
    def __init__(self, data_dir, train, transform, second_stage):
        super().__init__(root=data_dir, train=train, download=True, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        return image, label


