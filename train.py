import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
#from PIL import Image
#import pandas as pd
from sklearn.model_selection import train_test_split
#from torchvision.io import read_image, decode_image#, transforms
import torchvision.transforms as transforms
#from itertools import chain
import tqdm
import pickle
from preprocess import ThaiOCRData
torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

class ThaiOCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, (5,5), padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1*72*48, 300)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(300, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, image):
        out = self.conv2d(image)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return self.log_softmax(out)

def train(filename='trained_model.pt', epochs=4, device='cpu'):
    with open('train_dataset.pkl', 'rb') as f:
        loaded_train = pickle.load(f)
    classes = loaded_train.classes
    loader = DataLoader(loaded_train, batch_size=32, shuffle=True)
    model = ThaiOCRModel(len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(epochs):
        print("Training epoch", epoch)
        total_loss = 0
        
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            img_tensor, label = batch
            label = label.to(device)
            optimizer.zero_grad()
            out = model(img_tensor)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print('Average loss for epoch:', round(total_loss/len(loader), 3))
        print()

    torch.save(model.state_dict(), filename)

if __name__ == "__main__":

    train(device=device)