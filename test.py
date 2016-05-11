# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:02:35 2016

@author: hazarapet
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import csv

TEST_PATH  = "data/test/"

def test(f=None, image_size={'width': 640, 'height': 480}):
    
    files = glob.glob(TEST_PATH + "/*.jpg")
    files = np.array(files)
    with open('python_result.csv', 'w') as csvfile:
        fieldnames = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in files:
            fileName = i[len(TEST_PATH):]
            img = np.asarray(Image.open(i).convert('L').resize((image_size['width'], image_size['height'])))            
            img = img.reshape(-1, 1, image_size['height'], image_size['width'])
            img = img.transpose(0, 1, 3, 2);

            #plt.imshow(img[0,0]); 
            #plt.show()            
            
            result = f(img)
            
            #print result[0]            
            #print round(result[0], 1)
            rounded_result = np.round(result[0], 1);
            writer.writerow({'img': fileName,
                             'c0': rounded_result[0],
                             'c1': rounded_result[1],
                             'c2': rounded_result[2],
                             'c3': rounded_result[3],
                             'c4': rounded_result[4],
                             'c5': rounded_result[5],
                             'c6': rounded_result[6],
                             'c7': rounded_result[7],
                             'c8': rounded_result[8],
                             'c9': rounded_result[9]})
        
    print "Csv Write is Done"
    
    
    
#test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    