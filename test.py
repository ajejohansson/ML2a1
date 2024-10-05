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
import builtins
import json

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

def print_metrics(y_pred, y_test, classes, lang, model, filelog=False):
    # Note: This function is modified from a previous assignment where we were also asked to give these metrics
    
    """
    Prints accuracy; precision, recall, f-score per class; and macro precision, recall, f-score based on inputs.
    Args:
        y_pred: set of class predictions made by model
        y_test: set of gold labels of evaluation set where y_test[i] corresponds to y_pred[i]
    """
    
    
    # from sethdandridge.com/blog/redefining-pythons-print-function
    # This code snippet works to make all prints in the function write to a file instead of printing,
    # so I can gather the results that I refer to. When print_metrics is called with filelog=True,
    # the code runs well and creates/appends to the desired file.
    # However, when filelog=False (and )
    #if filelog:
    #    def print(*args, **kwargs):
    #        with open('results_log.txt', 'a+') as f:
    #            return builtins.print(*args, file=f, **kwargs)

    # from stackoverflow.com/questions/11124093/redirect-python-print-output-to-logger
    if filelog:
        log = open(os.getcwd()+'/results/results_log'+str(len(os.listdir('results'))+1)+'.txt', 'a')
        sys.stdout = log

    
    accuracy = accuracy_score(y_test, y_pred)
    p_scores = precision_score(y_test, y_pred, average = None, zero_division=0.0)
    r_scores = recall_score(y_test, y_pred, average = None, zero_division=0.0)
    f_scores = f1_score(y_test, y_pred, average = None, zero_division=0.0)
    macro_p = precision_score(y_test, y_pred, average = "macro", zero_division=0.0)
    macro_r = recall_score(y_test, y_pred, average = "macro", zero_division=0.0)
    macro_f = f1_score(y_test, y_pred, average = "macro", zero_division=0.0)

    if lang:
        print('Results for ', lang)
    else:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("Results with the following test/eval configuration:")
        print(config)
    print()
    print("model params:")
    print(model)
    print()
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
    print()
    print('------------------------------------------------')
    if filelog:
        log.close()

def test(device='cpu', eval_set='val'):
    #with open('val_dataset.pkl', 'rb') as f:
    with open(eval_set+'_dataset.pkl', 'rb') as f:
        loaded_eval = pickle.load(f)
    
    classes = loaded_eval.classes
    lang = loaded_eval.lang
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
        gold_label = label#.to('cpu')
        pred = model(img_tensor).to('cpu')

        #could also append gold_label/pred[0], but extend makes it robust to larger batch sizes than 1
        # append label/pred[0] rather than extend label/pred means the code is not robust to batch sizes
        # >1, but there isn't really a reason to go >1 for testing, and this is just simpler than 
        # getting the argmax through a list comprehension or something.
        y_test.append(gold_label[0])
        y_pred.append(pred[0].argmax())
    
    #print('Dumping...')
    #with open('y.pkl', 'wb') as f:
    #    pickle.dump((y_pred, y_test, classes), f)
    #print(y_pred[0], y_test[0])
    #print(len(y_pred), len(y_test))

    #print('Printing metrics...')

    print_metrics(y_pred, y_test, classes, lang, model, filelog=True) #filelog=True



#source ML-env/bin/activate
if __name__ == "__main__":
    test(device=device, eval_set='test')
    #trainset, testset, valset, classes = generate_split(training_dir)

    #classes = loaded_train.classes
    #model = train(classes, device=device)
    #print(classes)
    #train(device=device)