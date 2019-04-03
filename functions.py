import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

def build_classifier(model, input_units, hidden_units, dropout):
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model



def validation(model, validloader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0
    
    if gpu_mode == True: model.to('cuda')

    for images, labels in validloader:       
        if gpu_mode == True: images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy



def train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode):
    steps = 0
    print_every = 30

    if gpu_mode == True: model.to('cuda')
    
    for e in range(epochs):
        running_loss = 0

        for inputs, labels in trainloader:
            steps += 1           
            if gpu_mode == True: inputs, labels = inputs.to('cuda'), labels.to('cuda')
   
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
 
            if steps % print_every == 0:
               
                model.eval()
    
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu_mode)
                print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Training Loss: {:.3f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.3f} | ".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                model.train()
    
    return model, optimizer



def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0
    
    if gpu_mode == True: model.to('cuda')

    with torch.no_grad():
        for images, labels in testloader:
            
            if gpu_mode == True: images, labels = images.to('cuda'), labels.to('cuda')
   
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test accuracy of model for {total} images: {round(100 * correct / total,2)}%")


    
def save_model(model, train_data, optimizer, save_dir):
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict}

    return torch.save(my_checkpoint, save_dir)



def load_checkpoint(model, save_dir, gpu_mode):
        
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
            
    
   
def predict(processed_image, loaded_model, topk, gpu_mode):

    loaded_model.eval()
    
    if gpu_mode == True:
        loaded_model.to('cuda')
    else:
        loaded_model.cpu()
    
    with torch.no_grad():
        output = loaded_model.forward(processed_image)

    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    class_to_idx = loaded_model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list
