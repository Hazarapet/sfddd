"""
Created on Fri May  6 20:53:25 2016

@author: hazarapet
"""

import numpy as np
import matplotlib.pyplot as plt
import theano
import time
import lasagne
import lasagne.regularization as reg
import theano.tensor as T

import data_loader

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
N_EPOCH = 10
LAMBDA = 1e-3
IMAGE_SIZE = {};
IMAGE_SIZE['height'] = 120;
IMAGE_SIZE['width'] = int(120*1.33);

X = T.tensor4('X')
y = T.vector('y', dtype='int32')

#############################################################
###################### Network Building #####################
print "Preparing to build Network ...";

l_in = lasagne.layers.InputLayer(shape=(None, 3, IMAGE_SIZE['width'], IMAGE_SIZE['height']), input_var = X);

l_in_drop = lasagne.layers.dropout(l_in, p = 0.2)

l_hidden1 = lasagne.layers.DenseLayer(l_in_drop, num_units = 200,
                 nonlinearity=lasagne.nonlinearities.sigmoid)

l_hidden1_drop = lasagne.layers.dropout(l_hidden1, p=0.3);

l_hidden2 = lasagne.layers.DenseLayer(l_hidden1_drop, num_units = 200,
                 nonlinearity=lasagne.nonlinearities.sigmoid)

l_hidden2_drop = lasagne.layers.dropout(l_hidden2, p=0.3);

l_out = lasagne.layers.DenseLayer(l_hidden2_drop, num_units = 10, 
              nonlinearity=T.nnet.softmax)

print "Network has been built"

train_prediction = lasagne.layers.get_output(l_out);

train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
train_loss = train_loss.mean();

train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), y), 
                   dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(l_out, trainable = True)
updates = lasagne.updates.nesterov_momentum(train_loss,
            params, learning_rate=0.01, momentum=0.9)
            
# train loss function    
train_fn = theano.function([X, y], [train_loss, train_acc], updates=updates)

print "Train 'Prediction', 'Loss' 'Acc', 'Updates' and 'Train_fn' are ready"

# ##############################################################
# Test side
# Test prediction
test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
#f = theano.function([X], T.argmax(test_prediction, axis=1))

# Loss function for train
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
test_loss = train_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)
                      
val_fn = theano.function([X, y], [test_loss, test_acc])

print "Test 'Prediction', 'Loss', 'Acc' and 'Val_fn' are ready"
#############################################################
####################### Loading Sets ########################

print "Loading X_train, y_train, X_val, y_val datasets ..."

X_train, y_train, X_val, y_val = data_loader.load_small_train(
    data_count=20, image_size=IMAGE_SIZE);

print "Datasests are loaded"

##############################################################
######################## Training ############################

print "Starting Training..."

for epoch in range(N_EPOCH):
    train_err = 0;
    train_acc = 0;    
    train_batches = 0;
    start_time = time.time();
    
    for batch in iterate_minibatches(X_train, y_train, 10, shuffle=True):
        inputs, targets = batch
        t_err, t_acc = train_fn(inputs, targets);
        train_err += t_err;
        train_acc += t_acc;
        train_batches +=1;
    
    val_err = 0;
    val_acc = 0;
    val_batches = 0;
    
    for batch in iterate_minibatches(X_val, y_val, 10, shuffle=False):
        inputs, targets = batch;
        v_err, v_acc = val_fn(inputs, targets);
        val_err += v_err;
        val_acc += v_acc;
        val_batches += 1;
        
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, N_EPOCH, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
        
        


























print("{:.3f}s Runtime".format(time.time() - st_time));





















