from torch.utils.data import Dataset
import os
import cv2


class SupConDataset(Dataset):
    def __init__(self, data_dir, mode, transform, csv=None, second_stage=False):
        self.data_dir = data_dir
        self.mode = mode
        self.csv = csv
        self.second_stage = second_stage
        self.transform = transform

        if self.mode == "train":
            self.labels = csv.labels.values

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, "images", self.csv.iloc[idx, 1])

        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.second_stage:
            image = self.transform(image=img)['image']
        else:
            image = self.transform(img)

        if self.mode == "train":
            return image, self.labels[idx]
        return image