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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

from util import load_model, DataOrganizer

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, help='path to original dataset')
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--exp', type=str, default='', help='exp folder')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                    help='weight decay pow (default: -4)')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--make_test', type=bool, default=False, help='prepare test set?')
parser.add_argument('--val_prob', type=float, default=0.2, help='proportion of data for validation set')
parser.add_argument('--test_prob', type=float, default=0, help='proportion of data for test set')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    #fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # monitoring with WanB
    wandb.init(config=args)

    ### LOAD DATA ###

    # split data
    organizer = DataOrganizer(path=args.data)
    organizer.makeDirTree(make_test=args.make_test, verbose=False)
    organizer.splitAndSymlink(val_prob=args.val_prob, test_prob=args.test_prob)


    # data loading code
    traindir = os.path.abspath(os.path.join(organizer.train_subdirs[0], "../"))
    valdir = os.path.abspath(os.path.join(organizer.val_subdirs[0], "../"))


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transformations_val = [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize]

    transformations_train = [transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.RandomCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize]

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=transforms.Compose(transformations_train)
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transform=transforms.Compose(transformations_val)
    )

    image_datasets = {'train': train_dataset, 'val': val_dataset}




    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(args.batch_size/2),
                                             shuffle=False,
                                             num_workers=args.workers)

    dataloaders = {'train': train_loader, 'val': val_loader}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #### LOAD MODEL ###
    model = load_model(args.model)
    model.top_layer = nn.Linear(model.top_layer.weight.size(1), len(organizer.train_subdirs))


    # freeze some layers
    for param in model.features.parameters():
        param.requires_grad = False
    # unfreeze Linear scaling
    if args.train_batchnorm:
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # set optimizer
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    # decay of lr
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.cuda()
    cudnn.benchmark = True

    # training
    model_training = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)







def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            is_best = epoch_acc > best_acc
            if phase == 'val' and is_best:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if is_best:
                filename = 'model_best.pth.tar'
            else:
                filename = 'checkpoint.pth.tar'
            torch.save({
                'epoch': epoch + 1,
                'arch': 'alexnet',
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.exp, filename))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    main()
