# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
from glob import glob

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import models


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)




class DataOrganizer():
    """ Class to organize given data into train, validation, test
    Methods:
        - makeDirTree()

    """

    def __init__(self, path):
        self.dirs = glob(os.path.join(path, '*'))
        self.val_subdirs = []
        self.test_subdirs = []
        self.train_subdirs = []

    def makeDirTree(self, make_test = False, verbose=False):
        parent_path = os.path.abspath(os.path.join(self.dirs[0], "../.."))

        # val dir
        val_dir = os.path.join(parent_path, "val")
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        for subdir in self.dirs:
            base = os.path.join(val_dir, os.path.basename(subdir))
            self.val_subdirs.append(base)
            if verbose:
                print(base)
            if not os.path.exists(base):
                os.makedirs(base)


        # do we need test directory?
        if make_test:
            test_dir = os.path.join(parent_path, "test")
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            for subdir in self.dirs:
                base = os.path.join(test_dir, os.path.basename(subdir))
                self.test_subdirs.append(base)
                if verbose:
                    print(base)
                if not os.path.exists(base):
                    os.makedirs(base)

        # train dir
        train_dir = os.path.join(parent_path, "train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        for subdir in self.dirs:
            base = os.path.join(train_dir, os.path.basename(subdir))
            self.train_subdirs.append(base)
            if verbose:
                print(base)
            if not os.path.exists(base):
                os.makedirs(base)

    def splitAndSymlink(self, val_prob=0.2, test_prob=0):

        # split data
        for i in range(0,len(self.dirs)):
            image_list = glob(os.path.join(self.dirs[i], '*.tif'))
            numb_images = len(image_list)

            # validation set
            val_list = np.random.choice(image_list, round(numb_images * val_prob), replace=False)
            [os.symlink(x, os.path.join(self.val_subdirs[i], os.path.basename(x))) for x in val_list]

            # remove val images
            [image_list.remove(x) for x in val_list]

            # check if test set is needed
            if test_prob > 0:
                test_list = np.random.choice(image_list, round(numb_images * test_prob), replace=False)
                [os.symlink(x, os.path.join(self.test_subdirs[i], os.path.basename(x))) for x in test_list]

                # remove test images
                [image_list.remove(x) for x in test_list]

            # train set = remaining images
            [os.symlink(x, os.path.join(self.train_subdirs[i], os.path.basename(x))) for x in image_list]








