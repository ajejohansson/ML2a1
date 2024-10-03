import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #, random_split
from torch.optim import Adam
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.io import read_image, decode_image#, transforms
import torchvision.transforms as transforms
from itertools import chain
import tqdm
import pickle
from preprocess import ThaiOCRData
from train import ThaiOCRModel

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

def print_metrics(y_pred, y_test, classes):
    # Note: This function has more or less the same structure as one I have used in a previous assignment.
    
    """
    Prints accuracy; precision, recall, f-score per class; and macro precision, recall, f-score based on inputs.
    Args:
        y_pred: set of class predictions made by model
        y_test: set of gold labels of evaluation set where y_test[i] corresponds to y_pred[i]
    """
    accuracy = accuracy_score(y_test, y_pred)
    p_scores = precision_score(y_test, y_pred, labels = classes, average = None, zero_division=0.0)
    r_scores = recall_score(y_test, y_pred, labels = classes, average = None, zero_division=0.0)
    f_scores = f1_score(y_test, y_pred, labels = classes, average = None, zero_division=0.0)
    macro_p = precision_score(y_test, y_pred, labels = classes, average = "macro", zero_division=0.0)
    macro_r = recall_score(y_test, y_pred, labels = classes, average = "macro", zero_division=0.0)
    macro_f = f1_score(y_test, y_pred, labels = classes, average = "macro", zero_division=0.0)

    print("accuracy is", accuracy)
    print()

    for label, precision in zip(classes, p_scores):
        print("precision for label '{}' is {}".format(label, precision))
    print("macro precision is", macro_p)
    print()

    for label, recall in zip(classes, r_scores):
        print("recall for label '{}' is {}".format(label, recall))
    print("macro recall is", macro_r)
    print()

    for label, f_score in zip(classes, f_scores):
        print("f-score for label '{}' is {}".format(label, f_score))
    print("macro f-score is", macro_f)
    

def test(device='cpu', eval_set='val'):
    #with open('val_dataset.pkl', 'rb') as f:
    with open(eval_set+'_dataset.pkl', 'rb') as f:
        loaded_eval = pickle.load(f)
    
    classes = loaded_eval.classes
    loader = DataLoader(loaded_eval, batch_size=1)
    model = ThaiOCRModel(len(classes)).to(device)
    model.load_state_dict(torch.load('trained_model.pt', weights_only=True))
    model = model.to(device)
    #optimizer = Adam(model.parameters(), lr=0.1)
    #criterion = nn.NLLLoss().to(device)

    y_test = []
    y_pred = []
    model.eval()
 
        
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        img_tensor, label = batch
        gold_label = label.to('cpu')
        pred = model(img_tensor)

        #could also append gold_label/pred[0], but extend makes it robust to larger batch sizes than 1 
        y_test.extend(gold_label)
        y_pred.extend(pred)
    
    print_metrics(y_pred, y_test, classes)


        






#source ML-env/bin/activate
if __name__ == "__main__":
    test(device=device, eval_set='val')
    #trainset, testset, valset, classes = generate_split(training_dir)

    #classes = loaded_train.classes
    #model = train(classes, device=device)
    #print(classes)
    #train(device=device)