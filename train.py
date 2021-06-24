import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import copy

from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--num_epochs', dest='num_epochs', default='20')
    parser.add_argument('--device', action='store', default='device')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()


def train_model(model, criteria, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs=20, device='cuda'):
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
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
                    loss = criteria(outputs, labels)

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
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
def main():
     
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dirs = {'train': train_dir, 
        'valid': valid_dir,
        'test': test_dir}
    
    # TODO: Define your transforms for the training, validation, and testing sets
    size = 224
    data_transforms  = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
     ]),
                    } 

# TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']} 

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid',       'test']}
    dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criteria = nn.NLLLoss() # defining criterion and optimizer 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    num_epochs = int(args.num_epochs)
    class_index = image_datasets['train'].class_to_idx
    sched = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    eps=20
    device = args.device # get the gpu settings
    train_model(model, criteria, optimizer, sched, dataset_sizes, dataloaders, num_epochs = 10, device='cuda')
    model.class_to_idx = class_index
    path = args.save_dir # path to new save location
    save_checkpoint(path, model, optimizer, args, classifier)
    
if __name__ == "__main__":
    main()    
    
    