import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from utility_functions import load_data
from functions import build_classifier, validation, train_model, test_model, save_model, load_checkpoint

parser = argparse.ArgumentParser(description='Image Classifier.')

parser.add_argument('data_directory', action = 'store',
                    help = 'path to training data.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'the pretrained model, default is VGG-11.'

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'my_checkpoint.pth',
                    help = 'the location to save checkpoint.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.5,
                    help = 'dropout for training the model, default is 0.5.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 4096,
                    help = 'Enter number of hidden units in classifier, default is 4096.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'number of epochs during training, default is 1.')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units, dropout)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

test_model(model, testloader, gpu_mode)

save_model(loaded_model, train_data, optimizer, save_dir, epochs)
