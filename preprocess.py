

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

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

training_dir = os.path.abspath("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet")
#not only the training dir for our purposes, but close enough

def generate_split(parent_dir):
    if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)]:
        walk_tup = (os.walk(parent_dir + '/' + sys.argv[1].title()),)
        print('Generating data splits for ' + sys.argv[1].title()+'...')
    else:
        walk_tup = (tuple(os.walk(parent_dir + '/' + 'Thai')), tuple(os.walk(parent_dir + '/' + 'English')))
        print('Generating data splits for combined Thai + English...')
        print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")

    data_subsets = []
    #getting classes pre-split to make train/test not have a different number of them
    classes = set()

    # walk_tup is often only one item
    # the following construction (and some later code in this function)
    # is only for being agnostic to whether or not this is the case
    for subset in walk_tup:
        data_list = []
        for item in subset:
            parent_path = item[0]
            split_path = parent_path.split('/')
            label, dpi, typeface = split_path[-3:]
            pics = item[2]
            for pic in pics:
                if pic.endswith('.bmp'):
                    #using list of tups (not dict) to be compatible with sklearn's split function:
                    data_list.append((pic, label, dpi, typeface, parent_path))
                    classes.add(label)
        data_subsets.append(data_list)

    subset_splits = []
    # splits each data subset (e.g. 'Thai', 'English') into separate train, test, val sets, which are then concatenated
    # the upside of split->concat (as opposed to the other way around) is that the datasets have a controlled subset balance.
    # the downside is that the datasets technically aren't fully randomly sampled
    # (of course, this is only for language/special/numeric; it is still random with regards dpi, typeset ) 
    for subset in data_subsets:
        train, test_val = train_test_split(subset, train_size=0.8, random_state=1, shuffle=True)
        test, val = train_test_split(test_val, train_size=0.5, random_state=1, shuffle=True)
        subset_splits.append([train, test, val])

    train, test, val = [list(chain(*split)) for split in zip(*subset_splits)]
    return train, test, val, classes



def generate_datafiles(train, test, val, classes):
    current_directory = os.getcwd()
    datadir = os.path.join(current_directory, 'data')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    
    print(type(train))
    #def write_subset(subset, subset_name):
    #    with open(os.path.join(datadir, subset_name), 'w') as f:
    #        f.writelines(subset)
    #write_subset(train, 'train')

    #with open('saved_train.pkl', 'wb') as f:
    #    pickle.dump(train, f)
        
    #with open('saved_train.pkl', 'rb') as f:
    #    loaded_train = pickle.load(f)

    #print(type(loaded_train))

class ThaiOCRData(Dataset):
    def __init__(self, subset, classes):
        '''
        I input classes from the preprocess (as opposed to collecting them in-class) since
        I want all data subsets to have all classes, even if a particular one does not occur
        in a given subset.
        '''
        #walking = os.walk(training_dir + '/' + subdir.title())
        #print('Generating data splits for ' + subdir.title()+'...')
        #if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)]:
        #    walking = os.walk(parent_dir + '/' + sys.argv[1].title())
        #    print('Generating data splits for ' + sys.argv[1].title()+'...')
        #else:
        #    walking = tuple(os.walk(parent_dir + '/' + 'Thai')) + tuple(os.walk(parent_dir + '/' + 'English'))
        #    print('Generating data splits for combined Thai + English...')
        #    print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")
        data_dict = {}
        indices = []
        #classes = set()
        #for item in walking:
        #    full_path = item[0]
        #    split_path = full_path.split('/')
        #    label, dpi, typeface = split_path[-3:]
        #    pics = item[2]
        #    for pic in pics:
        for pic, label, dpi, typeface, parent_path in subset:
            #if pic.endswith('.bmp'):    
            data_dict[pic] = {'label': label, 'dpi': dpi, 'typeface': typeface, 'fullpath': os.path.join(parent_path, pic)}
            indices.append(pic)
        #        classes.add(label)

        self.parent_dir = training_dir
        self.data_dict = data_dict
        #self.subdir = subdir
        self.indices = indices
        self.classes = classes

        #from https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
        #read/decode_image from pytorch does not seem to handle bmp images
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.device = device

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_name = self.indices[idx]
        label = self.data_dict[img_name]['label']
        dpi =  self.data_dict[img_name]['dpi']
        typeface = self.data_dict[img_name]['typeface']
        pil_image = Image.open(self.data_dict[img_name]['fullpath'])
        #image = read_image(self.data_dict[img_name]['fullpath'])
        img_tensor = self.transform(pil_image).to(self.device) # read_image(pil_image)

        return img_name, label, dpi, typeface, img_tensor     
    
    



if __name__ == "__main__":
    traindata, testdata, valdata, classes = generate_split(training_dir)
    #generate_datafiles(train, test, val, classes)
    trainset = ThaiOCRData(traindata, classes)
    #print(trainset[0])
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
        
    with open('train_dataset.pkl', 'rb') as f:
        loaded_train = pickle.load(f)