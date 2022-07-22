
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from tqdm import tqdm

import glob
import os
import time
# from IPython import display as ipd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

labels = {
    'bacterial_leaf_blight': 0,
    'bacterial_leaf_streak': 1,
    'bacterial_panicle_blight': 2,
    'blast': 3,
    'brown_spot': 4,
    'dead_heart': 5,
    'downy_mildew': 6,
    'hispa': 7,
    'normal': 8,
    'tungro': 9
    
}
num_classes = len(labels.keys())
one_hot_encoding = F.one_hot(torch.arange(0, num_classes) % num_classes, num_classes=num_classes)
one_hot_encoding

class PaddyDiseaseClassificationDataset(Dataset):
    def __init__(self, data, dataset_name='', transforms=None):
        self.image_paths = data
        self.transforms = transforms
        self.name = dataset_name
        self.distribution = self.get_distribution()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        img = cv2.imread(img_path)[:, :, ::-1]  # convert it to rgb
        img = img.astype('float32')
        #img /= 255  # scale img to [0, 1]
        

        # print(img_path.split("/")[-2])
        # print(img_path)
        label = labels[img_path.split("/")[-2]]
        if self.transforms is not None:
            if label == 3 or label ==7:
                transform = self.transforms[0]
            else:
                transform = self.transforms[-1]
            img = transform(image=img)['image']

        encoded_label = one_hot_encoding[label]
        encoded_label = encoded_label.type(torch.FloatTensor)

        return img, encoded_label

    def get_distribution(self):
        distribution = {}
        splitted_paths = [path.split("/") for path in self.image_paths]
        for splitted_path in splitted_paths:
            if splitted_path[-2] not in distribution:
                distribution[splitted_path[-2]] = 0
            else:
                distribution[splitted_path[-2]] += 1
        print(f"Distribution of {self.name} dataset: {distribution}")
        return distribution