import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import projectUtils

ap = argparse.ArgumentParser(description='Train.py')
# Command Line args

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/", type = str)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

# parse args and set to parameters
pa = ap.parse_args()
where = ''.join(pa.data_dir)
path = pa.save_dir
lr = pa.learning_rate
network = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

print("#### args start ####")
print(where)
print(path)
print(lr)
print(network)
print(dropout)
print(hidden_layer1)
print(power)
print(epochs)
print("#### args end ####")


dataloaders, dataloaders_validation, dataloaders_test = projectUtils.load_data(where)

model, optimizer, criterion = projectUtils.nn_setup(network, dropout, hidden_layer1, lr, power)

projectUtils.train_network(model, optimizer, criterion, dataloaders, dataloaders_validation, dataloaders_test, epochs, 5, power)

projectUtils.save_checkpoint(path, network, hidden_layer1, dropout, lr)

print("The model has been trained!")