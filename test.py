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

def test(f, image_size={'width': 640, 'height': 480}):
    
    files = glob.glob(TEST_PATH + "/*.jpg")
    files = np.array(files)
    with open('python_result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
    
        for i in files:
            fileName = i[len(TEST_PATH):]
            img = np.asarray(Image.open(i).convert('L').resize((image_size['width'], image_size['height'])))            
            img = img.reshape(1, 1, image_size['height'], image_size['width'])
            img = img.transpose(0, 1, 3, 2);

            plt.imshow(img[0,0]); 
            plt.show()            
            
            result = f(img)
            writer.writerow(fileName, 
                    format(result[0], '.1f'),format(result[1], '.1f'), 
                    format(result[2], '.1f'),format(result[3], '.1f'), 
                    format(result[4], '.1f'),format(result[5], '.1f'),
                    format(result[6], '.1f'),format(result[7], '.1f'),
                    format(result[8], '.1f'),format(result[9], '.1f'))
        
    print "Csv Write is Done"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    