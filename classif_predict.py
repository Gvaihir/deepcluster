# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from util import load_model
from util import AverageMeter, Logger, UnifLabelSampler
import seaborn as sn
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import cv2
import json




from util import load_model, DataOrganizer

parser = argparse.ArgumentParser(description="""Use classifier to predict cellular phenotypes in the wells""")

parser.add_argument('--data', type=str, help='path to original dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')

parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', action='store_true', help='chatty')


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ImageOuput(object):
    """
    Object with full image name and location and predicted class

    """
    def __init__(self):
        self.img_name = []
        self.class_name = []

    def test(self, model, dataloader):
        # monitor test loss and accuracy
        img_name = []
        class_name = []

        with torch.no_grad():
            for batch_idx, (data, labels, path) in enumerate(dataloader):
                # move to GPU
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                self.img_name.extend(path)
                self.class_name.extend(preds)










def main():
    global args
    args = parser.parse_args()

    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    ### IMPORT DATA ###
    data_dir = args.data

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]
    dataset = ImageFolderWithPaths(data_dir, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=True,
                                             pin_memory=True)

    ### IMPORT MODEL ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model)
    # freeze some layers
    for param in model.features.parameters():
        param.requires_grad = False

    model.cuda()


    ### Create object ###
    obj = ImageOuput()
    obj.test(model, dataloader)


    ### Export JSON ###
    with open(os.path.join(args.exp, "json"), "w") as file:
        json.dump(obj.__dict__, file)



if __name__ == '__main__':
    main()
