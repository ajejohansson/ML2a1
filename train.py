import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #, random_split
from torch.optim import Adam
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.io import read_image, decode_image#, transforms
import torchvision.transforms as transforms
from itertools import chain
import tqdm
import pickle
from preprocess import ThaiOCRData

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

class ThaiOCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, (5,5), padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1*72*48, 10)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(10, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, image):
        out = self.conv2d(image)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return self.log_softmax(out)

def train(filename='trained_model.pt', epochs=2, device='cpu'):
    with open('train_dataset.pkl', 'rb') as f:
        loaded_train = pickle.load(f)
    classes = loaded_train.classes
    loader = DataLoader(loaded_train, batch_size=32, shuffle=True)
    model = ThaiOCRModel(len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=0.1)
    criterion = nn.NLLLoss().to(device)
    

    for epoch in range(epochs):
        print("Training epoch", epoch)
        
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            #print(batch_id, batch)
            #print(batch)
            #for i in batch:
            #    print(i)
            #    print()

            #img_name, label, dpi, typeface, img_tensor = batch
            img_tensor, label = batch
            print(label)
            print(print(img_tensor[0].dtype))
            #X, y = batch
            label = label.to(device)
            print('passed')
            optimizer.zero_grad()
            out = model(img_tensor)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            print('passed batch')
        

    print(loader)
    print(model)

    torch.save(model.state_dict(), filename)

    
#source ML-env/bin/activate
if __name__ == "__main__":
    #trainset, testset, valset, classes = generate_split(training_dir)

    #classes = loaded_train.classes
    #model = train(classes, device=device)
    #print(classes)
    train(device=device)