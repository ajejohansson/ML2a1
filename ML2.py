import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

language = sys.argv[1]
print(len(sys.argv))
print(sys.argv)
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

def splitter(training_directory):
    
    if sys.argv[1].lower() in [dirname.lower() for dirname in os.listdir(training_directory)]:
        walk_tup = tuple(os.walk(training_directory + '/' + sys.argv[1].title()))
        print('Generating data splits for ' + sys.argv[1].title()+'...')
    else:
        walk_tup = tuple(os.walk(training_directory + '/' + 'Thai')) + tuple(os.walk(training_directory + '/' + 'English'))
        print('Generating data splits for combined Thai + English...')
        print("If single subset dataset is sought, run with 'Thai', 'English', 'Special', or 'Numeric' as arg 1.")

    

    #walk_tup = tuple(os.walk(lang_dir))
    train, test_valid = train_test_split(walk_tup, train_size=0.8, random_state=1, shuffle=True)
    test, valid = train_test_split(test_valid, train_size=0.5, random_state=1, shuffle=True)
    return train, test, valid

split_test = splitter(training_dir)

print(len(split_test))



class ThaiOCRData(Dataset):
    def __init__(self, lang_dir):
        walking = os.walk(lang_dir)
        data_dict = {}
        indices = []
        for item in walking:
            split_path = item[0].split('/')
            label, dpi, typeface = split_path[-3:]
            pics = item[2]
            for pic in pics:
                data_dict[pic] = {'label': label, 'dpi': dpi, 'typeface': typeface}
                indices.append(pic)

            #print(item)

            #print(label, dpi, typeface)

            #print(pic)

        self.data_dict = data_dict
        self.lang_dir = lang_dir
        self.indices = indices


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        img_name = self.indices[idx]
        label = self.data_dict[img_name]['label']
        dpi =  self.data_dict[img_name]['dpi']
        typeface = self.data_dict[img_name]['typeface']
        image = read_image(img_name)

        return img_name, label, dpi, typeface, image     

    
            #one = os.path.basename(item[0])
            #two = os.path.basename(one)
            #print(one, two)
            #print(item[0])
            #print()
            #id = 
            #print(img[0])
            #print(img)
            #break


#gen= os.walk("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/")

thai = ThaiOCRData("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/")

#print(thai.data_dict)
#print()
#print(thai.lang_dir)
#print()
#print(thai.indices)
#print(gen(next))



def list_files(training_dir):
    for root, dirs, files in os.walk(training_dir):
        level = root.replace(training_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


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

#source ML-env/bin/activatesource ML-env/bin/activate