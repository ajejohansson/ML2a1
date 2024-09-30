import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.io import read_image, decode_image#, transforms
import torchvision.transforms as transforms
from itertools import chain

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

language = sys.argv[1]
#print(len(sys.argv))
#print(sys.argv)
#print(language)

#source ML-env/bin/activate

# path = os.path.abspath("/scratch/lt2326-2926-h24/ThaiOCR/")
training_dir = os.path.abspath("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet")
#training_dir = os.path.abspath("/scratch/lt2326-2926-h24/ThaiOCR")
#print(os.path.isdir(path))
#"/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/"
#print(training_dir)
#print([dirname.lower() for dirname in os.listdir(training_dir)])
#walk_tup = tuple(os.walk(training_dir))
#print(os.listdir(training_dir))
#print('thai'.title())

def data_preprocess(training_directory):
    
    if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(training_directory)]:
        walk_tup = tuple(os.walk(training_directory + '/' + sys.argv[1].title()))
        print('Generating data splits for ' + sys.argv[1].title()+'...')
    else:
        walk_tup = tuple(os.walk(training_directory + '/' + 'Thai')) + tuple(os.walk(training_directory + '/' + 'English'))
        print('Generating data splits for combined Thai + English...')
        print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")

    

    #walk_tup = tuple(os.walk(lang_dir))
    
    train, test_valid = train_test_split(walk_tup, train_size=0.8, random_state=1, shuffle=True)
    test, valid = train_test_split(test_valid, train_size=0.5, random_state=1, shuffle=True)
    return (train, test, valid)

#data_split = data_preprocess(training_dir)

#print(len(split_test))

#for sp in data_split:
#    print(len(sp))

def get_data_split(data_dict):

    df = pd.DataFrame(data_dict)

    #walk_tup = tuple(os.walk(lang_dir))
    
    train, test_valid = train_test_split(df, train_size=0.8, random_state=1, shuffle=True)
    test, valid = train_test_split(test_valid, train_size=0.5, random_state=1, shuffle=True)
    return (train, test, valid)



def generate_split(parent_dir):
    #splits data 
        #walking = os.walk(dset)
    if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)]:
        walk_tup = (os.walk(parent_dir + '/' + sys.argv[1].title()),)
        print('Generating data splits for ' + sys.argv[1].title()+'...')
    else:
        walk_tup = (tuple(os.walk(parent_dir + '/' + 'Thai')), tuple(os.walk(parent_dir + '/' + 'English')))
        print('Generating data splits for combined Thai + English...')
        print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")
    #data_dict = {}   

    #indices = []

    data_subsets = []
    
    # walk_tup is often only one item
    # the following construction (and some later code in this function)
    # is only to be agnostic to whether or not this is the case
    for subset in walk_tup:
        data_list = []
        for item in subset:
            full_path = item[0]
            split_path = full_path.split('/')
            label, dpi, typeface = split_path[-3:]
            pics = item[2]
            for pic in pics:
                if pic.endswith('.bmp'):
                    #using list of tups (not dict) to be compatible with sklearn's split function:
                    data_list.append((pic, label, dpi, typeface))
                    
                    #data_dict[pic] = {'label': label, 'dpi': dpi, 'typeface': typeface}
                    #indices.append(pic)
        data_subsets.append(data_list)

    #df = pd.DataFrame(data_dict)
    #print(df)
    #print(data_dict)
    #dict_items = data_dict.items()
    #print(len(dict_items))


    subset_splits = []
    # splits each data subset (e.g. 'Thai', 'English') into separate train, test, val sets, which are then concatenated
    # the upside of split->concat (as opposed to the other way around) is that the datasets have a controlled subset balance.
    # the downside is that the datasets technically aren't fully randomly sampled
    # (of course, this is only for language/special/numeric; it is still random with regards dpi, typeset ) 
    for subset in data_subsets:
        train, test_val = train_test_split(subset, train_size=0.8, random_state=1, shuffle=True)
        test, val = train_test_split(test_val, train_size=0.5, random_state=1, shuffle=True)
        subset_splits.append((train, test, val))
    
    print(len(subset_splits[0]))

    #concat_split = []
    zipped = list(zip(*subset_splits))
    train_test_val = [tuple(chain(*split)) for split in zip(*subset_splits)]
    #train_test_val = list(zip(subset_splits))
    #print(train_test_val)
    print(len(train_test_val))
    for item in train_test_val:
        print(len(item))
    #print(len(train_test_split[0]))
    #for item in train_test_val:
    #    print(item)
    #train, test, val = []
    
    #return (train, test, valid)

#train, test, valid = generate_split(training_dir)
#df, train, walking = generate_split(training_dir)
#print(df)
#print('PASSED')

generate_split(training_dir)

class ThaiOCRData(Dataset):
    def __init__(self, subdir):
        walking = os.walk(training_dir + '/' + subdir.title())
        print('Generating data splits for ' + subdir.title()+'...')
        '''if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(parent_dir)]:
            walking = os.walk(parent_dir + '/' + sys.argv[1].title())
            print('Generating data splits for ' + sys.argv[1].title()+'...')
        else:
            walking = tuple(os.walk(parent_dir + '/' + 'Thai')) + tuple(os.walk(parent_dir + '/' + 'English'))
            print('Generating data splits for combined Thai + English...')
            print("If a single subset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")'''
        data_dict = {}
        indices = []
        classes = set()
        for item in walking:
            full_path = item[0]
            split_path = full_path.split('/')
            label, dpi, typeface = split_path[-3:]
            pics = item[2]
            for pic in pics:
                if pic.endswith('.bmp'):    
                    data_dict[pic] = {'label': label, 'dpi': dpi, 'typeface': typeface, 'fullpath': os.path.join(full_path, pic)}
                    indices.append(pic)
                    classes.add(label)

        self.parent_dir = training_dir
        self.data_dict = data_dict
        self.subdir = subdir
        self.indices = indices

        #from https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
        #read/decode_image from pytorch does not seem to handle bmp images
        self.transform = transforms.Compose([transforms.PILToTensor()])


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_name = self.indices[idx]
        label = self.data_dict[img_name]['label']
        dpi =  self.data_dict[img_name]['dpi']
        typeface = self.data_dict[img_name]['typeface']
        pil_image = Image.open(self.data_dict[img_name]['fullpath'])
        #image = read_image(self.data_dict[img_name]['fullpath'])
        img_tensor = self.transform(pil_image) # read_image(pil_image)

        return img_name, label, dpi, typeface, img_tensor     

    
            #one = os.path.basename(item[0])
            #two = os.path.basename(one)
            #print(one, two)
            #print(item[0])
            #print()
            #id = 
            #print(img[0])
            #print(img)
            #break

#thai = ThaiOCRData('thai')
#counter = 1
'''print(len(thai))
for item in thai:
    print(item)
    print(counter)
    counter +=1'''
#print(thai[2])
#gen= os.walk("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/")


#unsplit = ThaiOCRData(training_dir)

#train_dataset, test_dataset, val_dataset = random_split(unsplit, [0.8, 0.1, 0.1])
#print(unsplit.data_dict)

#print(train_dataset.data_dict, test_dataset.data_dict, val_dataset.data_dict)


#        self.data_dict = data_dict
#        self.dataset = parent_dir
#        self.indices = indices
#count = 0
#print(len(train.data_dict))
#print(len(train), len(test), len(val))

#thai = ThaiOCRData("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/")1
#d = {"s": 2, "t":3}
#train, test_valid = train_test_split(d, train_size=0.8, random_state=1, shuffle=True)
#print(thai.data_dict)
#print()
#print(thai.lang_dir)
#print()
#print(thai.indices)
#print(gen(next))






class ThaiOCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 3, (5,5), padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3*3*3, 10)
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

def train(epochs=2):
    train_data = ThaiOCRData('thai')
    loader = DataLoader(train_data, batch_size=8, shuffle=True)
    print(loader)





train()
'''def list_files(training_dir):
    for root, dirs, files in os.walk(training_dir):
        level = root.replace(training_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))'''


#list_files(training_dir)

#print(os.listdir(training_dir+'/Thai/199/200/normal'))
#print(os.listdir(training_dir+'/Thai'))
"""for item in os.listdir(training_dir+'/Thai'):
    #if os.path.isdir(item):
    print(os.path.isdir(training_dir+'/Thai/'+item))"""
    #print(item)
                       
#print(os.listdir(training_dir))




"""filepath = os.path.join(os.path.dirname(__file__), '../data/'+filename)
    with open(filepath) as f:
        loaded_data = f.read()
    return loaded_data.split("\n")"""



import itertools
def load_data(filename):
    return None
    """
    Reads text file with data instances separate by newline.
    Assumes directory structure per assignment 2 but does not strictly have to be 
run from src directory.
    Arg:
        filename: name of file to be read, with no additional path information.
    Returns:
        content of input file in the form of a list, each element representing a 
file line.
    """
    """filepath = os.path.join(os.path.dirname(__file__), '../data/'+filename)
    with open(filepath) as f:
        loaded_data = f.read()
    return loaded_data.split("\n")"""


#print(device)

#source ML-env/bin/activate