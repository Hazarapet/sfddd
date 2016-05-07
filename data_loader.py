"""
Created on Sat May  7 18:12:17 2016

@author: hazarapet
"""
import Image
import numpy as np
import glob
from scipy import misc

TRAIN_PATH = "data/train/";
TEST_PATH  = "data/test/"

TRAIN_FOLDERS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

def load_small_train(data_count = 10, shuffle = False):
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
            img = misc.imread(i)
            data.append(img)
            labels.append(fl)
            
            
    return np.array(data, dtype='int32'), np.array(labels, dtype='int32')
    
        
        

    
    