import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
#from torchvision.io import read_image, decode_image 
import torchvision.transforms as transforms
from itertools import chain
import pickle
import json

with open("config.json", "r") as f:
    config = json.load(f)

def decode_classes(langdir_tup):
    """
    langdir_tup = iterable of paths to the language (or subset) specific directory, i.e 'English', 'Numeric'
                If only one such directory is targeted, must still be a list/tuple-like (of len 1) 

    Returns a translation key dictionary from class id code (e.g. '110') to more sensible character (e.g. 'p'),
    based on files in the input directory. Assumes the structure of assignment directory, where the only file
    at this directory level is a translation key text file.

    """
    #Not necessary for classification but useful for analysis.
    class_dict = {}
    for langdir in langdir_tup:
        onlyfiles = [f for f in os.listdir(langdir) if os.path.isfile(os.path.join(langdir, f))]
        classfile = os.path.join(langdir, onlyfiles[0])
        with open(classfile, encoding="TIS-620") as f:
            lines = f.readlines()
        for id_char in lines:
            if len(id_char) > 1: #filters out some whitespace lines
                id, char = id_char.split()
                class_dict[id] = char
    return class_dict

def generate_split(parent_dir):

    disjoint = True
    # 'disjoint' is a variable I use to determine whether the subsets need to be split
    # if all data is used or the model is trained and evaluated on exactly the same data subset, there needs to be a standard split,
    # but if the training and evaluation are done on disjoint subsets, there is no reason to split these subsets into even smaller sets

    if len(sys.argv) > 1:
        disjoint = False
        if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)]:
            langdir = os.path.join(parent_dir, sys.argv[1].title())
            langdir_tup = (langdir,)
            walk_tup = (os.walk(langdir),)
            print('Generating data splits for ' + sys.argv[1].title()+'...')

            #variable I save for evaluation printing purposes:
            lang = sys.argv[1].title()
        else:
            thaidir = os.path.join(parent_dir, 'Thai')
            engdir = os.path.join(parent_dir, 'English')
            langdir_tup = (thaidir, engdir)
            walk_tup = (tuple(os.walk(thaidir)), tuple(os.walk(engdir)))
            print('Generating data splits for combined Thai + English...')
            print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")
            lang = 'Thai + English'
    else:
        print('Generating datasets per configuration file:')
        print(config)
        if config['train'] == config['eval']: # bit of a shortcut and should work for experiments, but not foolproof: see readme
            disjoint = False
        
        if (config['train']['lang'].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)] 
        and config['train']['lang'].title() == config['eval']['lang'].title()):
            langdir = os.path.join(parent_dir, config['train']['lang'].title())
            langdir_tup = (langdir,)
            walk_tup = (os.walk(langdir),)
        else:
            thaidir = os.path.join(parent_dir, 'Thai')
            engdir = os.path.join(parent_dir, 'English')
            langdir_tup = (thaidir, engdir)
            walk_tup = (tuple(os.walk(thaidir)), tuple(os.walk(engdir)))
        lang = False #only 'false' in that I use the boolean to tell the eval a command line arg
                    #has not been given and to instead use the config file

    cl_id_to_char = decode_classes(langdir_tup)

    data_subsets = []

    #getting classes pre-split to make train/test not have a different number of them
    classes = set()
    if disjoint:
        train = []
        eval = []  

    # walk_tup is often only one item
    # the following construction (and some later code in this function)
    # is only for being agnostic to whether or not this is the case:
    for subset in walk_tup:
        data_list = []
        for item in subset:
            parent_path = item[0]
            split_path = parent_path.split('/')
            label, dpi, typeface = split_path[-3:]

            # Some label IDs are missing in the label text files even though they occur in the data (e.g. '195')
            # I want to use the more interpretable characters for analysis if possible, but will use the id if not
            if label in cl_id_to_char:
                label = cl_id_to_char[label]
            
            pics = item[2]
            for pic in pics:
                if pic.endswith('.bmp'):
                    #using list of tups (not dict) to be compatible with sklearn's split function:
                    if len(sys.argv) > 1:
                        data_list.append((pic, label, parent_path))
                        classes.add(label)
                    elif not disjoint:
                        if (dpi in config['train']['dpi'] and typeface in config['train']['typeface']):
                            data_list.append((pic, label, parent_path))
                            classes.add(label)
                    else:
                        if (dpi in config['train']['dpi'] and typeface in config['train']['typeface']):
                            train.append((pic, label, parent_path))
                            classes.add(label)
                        if (dpi in config['eval']['dpi'] and typeface in config['eval']['typeface']):
                            eval.append((pic, label, parent_path))
                            classes.add(label)

        data_subsets.append(data_list)

    if disjoint:
        test, val = train_test_split(eval, train_size=0.5, random_state=1, shuffle=True)
        return train, test, val, classes, lang

    subset_splits = []

    # splits each data subset (e.g. 'Thai', 'English') into separate train, test, val sets, which are then concatenated
    for subset in data_subsets:
        train, test_val = train_test_split(subset, train_size=0.8, random_state=1, shuffle=True)
        test, val = train_test_split(test_val, train_size=0.5, random_state=1, shuffle=True)
        subset_splits.append([train, test, val])

    train, test, val = [list(chain(*split)) for split in zip(*subset_splits)]

    return train, test, val, classes, lang

class ThaiOCRData(Dataset):
    def __init__(self, subset, classes, lang):

        '''
        Subclass of pytorch Dataset

        args:
            subset: dataset tuple with structure of outputs 0,1,2 of generate_split()
            classes: set of classes from data
            lang: string of language data is based on, or False boolean if language is received from config file
        '''
        data_dict = {}
        indices = []
        
        for pic, label, parent_path in subset:
         
            data_dict[pic] = {'label': label, 'fullpath': os.path.join(parent_path, pic)}
            indices.append(pic)

        self.parent_dir = training_dir
        self.data_dict = data_dict
        self.indices = indices
        self.classes = list(classes)
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}

        #partially from https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((72, 48))
            ])
        self.device = device
        self.lang = lang    

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_name = self.indices[idx]

        label_str = self.data_dict[img_name]['label']
        label = torch.as_tensor(self.class_to_idx[label_str])
        pil_image = Image.open(self.data_dict[img_name]['fullpath'])
        img_tensor = self.transform(pil_image).to(self.device)

        return img_tensor, label

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:3')
    else:
        device = torch.device('cpu')

    training_dir = os.path.abspath("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet")
    traindata, testdata, valdata, classes, lang = generate_split(training_dir)
    trainset = ThaiOCRData(traindata, classes, lang)
    testset = ThaiOCRData(testdata, classes, lang)
    valset = ThaiOCRData(valdata, classes, lang)

    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open('test_dataset.pkl', 'wb') as f:
        pickle.dump(testset, f)
    with open('val_dataset.pkl', 'wb') as f:
        pickle.dump(valset, f)
