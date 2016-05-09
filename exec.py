# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:52:14 2016

@author: hazarapet
"""

import pandas as pd
import theano
import time
import csv
import lasagne
import lasagne.regularization as reg
import numpy as np
import theano.tensor as T
import pylab

def load_train_data():
    with open('train.csv', 'rb') as train_csv:
        fileData = csv.reader(train_csv)
        labels = []
        data = [];
        
        i = 0
        for d in fileData:
            if i > 0:
                labels.append(np.int(d[0]))
                data.append(np.array(d[1:], dtype='int32'))
            i = i + 1;

    data = np.array(data, dtype='int32').reshape(-1, 1, 28, 28);
    labels = np.array(labels, dtype='int32');
    
    indexes = np.arange(len(data));
    np.random.shuffle(indexes);
    
    train_ind = indexes[:-10000];
    val_ind = indexes[-10000:];
    
    train_set = data[train_ind];
    val_set = data[val_ind];
    
    train_labels = labels[train_ind];
    val_labels = labels[val_ind];
    
    return train_set, train_labels, val_set, val_labels
    
def load_test_data():
    with open('test.csv', 'rb') as test_csv:
        fileData = csv.reader(test_csv)
        data = [];
        
        i = 0
        for d in fileData:
            if i > 0:
                data.append(d)
            i = i + 1;
            
    return np.array(data, dtype='int32').reshape(-1, 1, 28, 28)
    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




st_time = time.time()
N_epoch = 100
Lambda = 1e-3

X_train, y_train, X_val, y_val = load_train_data()

X = T.tensor4('X')
y = T.vector('y', dtype='int32')


# ########################## Building Network ################################
# ############################################################################
l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=X)

l_in_drop = lasagne.layers.DropoutLayer(l_in, p = 0.2)

l_hidden1 = lasagne.layers.DenseLayer(l_in, num_units=600,
             nonlinearity=lasagne.nonlinearities.sigmoid)

l_hid1_drop = lasagne.layers.DropoutLayer(l_hidden1, p = 0.5)       
                              
l_hidden2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=600,
            nonlinearity=lasagne.nonlinearities.sigmoid)

l_out = lasagne.layers.DenseLayer(l_hidden2, num_units=10,
                                  nonlinearity=lasagne.nonlinearities.softmax)
                                  
# get the prediction of network
train_prediction = lasagne.layers.get_output(l_out)
#f = theano.function([X], prediction)
# Loss function for train
train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
train_loss = train_loss.mean()

# Regularization
layer1_reg = reg.regularize_layer_params(l_hidden1, reg.l1)*Lambda
layer2_reg = reg.regularize_layer_params(l_hidden2, reg.l1)*Lambda

train_loss = train_loss + layer1_reg + layer2_reg

# train params and updates
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.nesterov_momentum(
            train_loss, params, learning_rate=0.01, momentum=0.9)

# train function
train_fn = theano.function([X, y], train_loss, updates=updates)

# ##############################################################
# Test side
# Test prediction
test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
f = theano.function([X], T.argmax(test_prediction, axis=1))
# Loss function for train
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
test_loss = train_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)
                      
val_fn = theano.function([X, y], [test_loss, test_acc])


for epoch in range(N_epoch):
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, N_epoch, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))



X_test = load_test_data();

print "-----------------Testing----------------"
#print f(X_test)


with open('python_result.csv', 'w') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    result = f(X_test)
    
    writer.writeheader()
    for i in range(len(result)):
        writer.writerow({'ImageId': i + 1, 'Label': result[i]})
        
    print "Csv Write is Done"
        










print("{:.3f}s Runtime".format(time.time() - st_time));
