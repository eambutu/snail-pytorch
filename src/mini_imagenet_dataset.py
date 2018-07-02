# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import errno
import os
from PIL import Image
import torch
import shutil
import pickle


class MiniImagenetDataset(data.Dataset):
    def __init__(self, mode='train', root='../dataset/mini-imagenet', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. Follow instructions in README to download mini-imagenet.')

        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        f = open(pickle_file, 'rb')
        self.data = pickle.load(f)

        self.x = [np.transpose(x, (2, 0, 1)) for x in self.data['image_data']]
        self.x = [torch.FloatTensor(x) for x in self.x]
        self.y = [-1 for _ in range(len(self.x))]
        class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = class_idx[class_name]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(self.root)

def index_classes(items):
    idx = {}
    for i in items:
        if (not i in idx):
            idx[i] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx
