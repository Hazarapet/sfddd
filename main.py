"""
Created on Fri May  6 20:53:25 2016

@author: hazarapet
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import theano
import time
import lasagne
import lasagne.regularization as reg
import theano.tensor as T

import data_loader

#small_train_data, small_train_labels = data_loader.load_small_train()

st_time = time.time()
N_epoch = 100
Lambda = 1e-3

X = T.tensor4('X')
y = T.vector('y', dtype='int32')

l_in = lasagne.layers.InputLayer((480, 640, 3));

l_hidden = lasagne.layers.DenseLayer(l_in, num_units=100)

l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10, 
              nonlinearity=T.nnet.softmax)


train_prediction = lasagne.layers.get_output(l_out)

train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
train_loss = train_loss.mean();

params = lasagne.layers.get_all_params(l_out, trainable = True)
updates = lasagne.updates.nesterov_momentum(train_loss,
            params, learning_rate=0.01, momentum=0.9)



















print("{:.3f}s Runtime".format(time.time() - st_time));





















