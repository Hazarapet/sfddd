"""
Created on Sat May  7 18:12:17 2016

@author: hazarapet
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
from PIL import Image
from scipy import misc

TRAIN_PATH = "data/train/";
TEST_PATH  = "data/test/"

TRAIN_FOLDERS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

def load_small_train(data_count = 10, shuffle = False, image_size={'width': 640,'height': 480}):
    data = []
    labels = [];
    
    # indexes to read files
    indexes = np.arange(data_count);
    
    #shuffle if needs
    if shuffle:
        np.random.shuffle(indexes);
    
    # loop on classes folders
    for (cl, fl) in TRAIN_FOLDERS:
        files = glob.glob(TRAIN_PATH + cl + "" + fl + "/*.jpg")
        files = np.array(files)
        for i in files[indexes]:
            img = np.asarray(Image.open(i).convert('RGB').resize((image_size['width'], image_size['height'])))            
            #img = img.reshape(1, image_size['height'], image_size['width'])
            
            # show resized image
            #plt.imshow(img); 
            #plt.show()
            
            img = img.transpose(2, 1, 0)
            data.append(img)
            labels.append(fl)
            
    sep_ind = np.arange(len(data));
    np.random.shuffle(sep_ind);
    
    data = np.array(data, dtype='int32')/256;
    labels = np.array(labels, dtype='int32');
     
    x_train = data[sep_ind[:-data_count*10/3]];
    y_train = labels[sep_ind[:-data_count*10/3]];

    x_val = data[sep_ind[-data_count*10/3:]];
    y_val = labels[sep_ind[-data_count*10/3:]];      
           
    return x_train, y_train, x_val, y_val
    
        
        
def loadVGG19():
    with open('data/vgg19.pkl', 'rb') as fileP:   
        weights = pickle.load(fileP);
        return weights['param values'];























    
    