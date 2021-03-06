# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json



from util import load_model, DataOrganizer

parser = argparse.ArgumentParser(description="""Use classifier to predict cellular phenotypes in the wells""")

parser.add_argument('--data', type=str, help='path to original dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--class_labels', type=str, help='abs path to class labels')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
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

    def test(self, model, dataloader, classes):

        with torch.no_grad():
            for batch_idx, (data, labels, path) in enumerate(dataloader):
                # move to GPU
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                self.img_name.extend(path)
                res = [classes[x] for x in preds.tolist()]
                self.class_name.extend(res)










def main():
    global args
    args = parser.parse_args()
    since = time.time()
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

    ### Class Labels
    with open(args.class_labels, 'r') as f:
        classes = f.read().splitlines()

    ### Create object ###
    obj = ImageOuput()
    obj.test(model, dataloader, classes)

    out_file = ".".join([os.path.basename(os.path.dirname(data_dir)), "json"])




    ### Export JSON ###
    with open(os.path.join(args.exp, out_file), "w") as file:
        json.dump(obj.__dict__, file)

    time_elapsed = time.time() - since
    print('Prediction complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()
