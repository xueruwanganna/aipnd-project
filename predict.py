import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

from utility_functions import load_data, process_image
from functions import load_checkpoint, predict, test_model


parser = argparse.ArgumentParser(description='class prediction')

parser.add_argument('--image_path', action='store',
                    default = '../aipnd-project/flowers/test/102/image_08004',
                    help='path to image.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'my_checkpoint.pth',
                    help='ocation to save checkpoint.')

parser.add_argument('--arch', action='store',
                    dest='pretrained_model', default='vgg11',
                    help='pretrained model to use, default is VGG11.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 3,
                    help='number of top most likely classes, default is 3.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='path to image.')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')

results = parser.parse_args()

save_dir = results.save_directory
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

loaded_model = load_checkpoint(model, save_dir, gpu_mode)

processed_image = process_image(image)

if gpu_mode == True: processed_image = processed_image.to('cuda')

probs, classes = predict(processed_image, loaded_model, top_k, gpu_mode)

print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[i]]
    
print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

