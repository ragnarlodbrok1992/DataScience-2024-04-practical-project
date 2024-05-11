import os
import time
import wget
import zipfile

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from PIL import Image

import pycuda.driver as drv
import pycuda.autoinit

# Global driver initialization
drv.init()


def gpu_info():
    print(f"Number of devices: {drv.Device.count()}")

    for i in range(drv.Device.count()):
        dev = drv.Device(i)
        print(f"Device {i}: {dev.name()}")
        print(f"  Compute capability: {dev.compute_capability()}")
        print(f"  Total memory: {dev.total_memory() / 1024**2} MB")

def is_cuda_available():
    return torch.cuda.is_available()

@dataclass
class Config:
    # Hyperparameters
    random_seed: int = 1
    learning_rate: float = 0.001
    num_epochs: int = 10

    # Architecture
    num_features: int = 128 * 128
    num_classes: int = 2
    batch_size: int = 128
    device: str = 'cuda:0'
    grayscale: bool = False

    # Dataset configuration
    data_root: str = 'datasets'
    base_url: str = 'https://graal.ift.ulaval.ca/public/celeba/'
    file_list: list = field(default_factory=lambda: [
        'img_align_celeba.zip',
        'list_attr_celeba.txt',
        'identity_CelebA.txt',
        'list_bbox_celeba.txt',
        'list_landmarks_align_celeba.txt',
        'list_eval_partition.txt'
    ])

    def get_config_dict(self):
        temp_dict = {}
        for key, value in self.__dict__.items():
            temp_dict[key.upper()] = value
        return temp_dict

class DatasetDownloader:
    def __init__(self, conf: Config):
        self.data_root = conf.data_root
        self.base_url = conf.base_url
        self.file_list = conf.file_list

    def download(self):
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        for file in self.file_list:
            if not os.path.exists(os.path.join(self.data_root, file)):
                print(f"Downloading {file}...")
                wget.download(self.base_url + file, self.data_root)
        return self

    def extract(self):
        for file in self.file_list:
            if file.endswith('.zip') and not os.path.exists(os.path.join(self.data_root, file.replace('.zip', ''))):
                print(f"Extracting {file}...")
                with zipfile.ZipFile(os.path.join(self.data_root, file), 'r') as zip_ref:
                    zip_ref.extractall(self.data_root)
        return self

if __name__ == "__main__":
    gpu_info()
    print(f"Is CUDA available: {is_cuda_available()}")

    # Dataset downloader
    conf = Config()
    # print(conf.get_config_dict())

    downloader = DatasetDownloader(conf)
    downloader.download().extract()
