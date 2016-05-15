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
import test

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
IMAGE_SIZE = {'height': 50, 'width': int(50*1.33)};

X = T.tensor4('X')
y = T.vector('y', dtype='int32')

#############################################################
###################### Network Building #####################
print "Preparing to build Network ...";
network = {};

network['l_in'] = lasagne.layers.InputLayer(shape=(None, 3, IMAGE_SIZE['width'], IMAGE_SIZE['height']), input_var = X);


network['conv_1'] = lasagne.layers.Conv2DLayer(network['l_in'], 
                    num_filters=64, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)

network['conv_2'] = lasagne.layers.Conv2DLayer(network['conv_1'], 
                    num_filters=64, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)



network['conv_3'] = lasagne.layers.Conv2DLayer(network['conv_2'], 
                    num_filters=128, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)

network['conv_4'] = lasagne.layers.Conv2DLayer(network['conv_3'], 
                    num_filters=128, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)



network['conv_5'] = lasagne.layers.Conv2DLayer(network['conv_4'], 
                    num_filters=256, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_6'] = lasagne.layers.Conv2DLayer(network['conv_5'], 
                    num_filters=256, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_7'] = lasagne.layers.Conv2DLayer(network['conv_6'], 
                    num_filters=256, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_8'] = lasagne.layers.Conv2DLayer(network['conv_7'], 
                    num_filters=256, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    


network['conv_9'] = lasagne.layers.Conv2DLayer(network['conv_8'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_10'] = lasagne.layers.Conv2DLayer(network['conv_9'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_11'] = lasagne.layers.Conv2DLayer(network['conv_10'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_12'] = lasagne.layers.Conv2DLayer(network['conv_11'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_13'] = lasagne.layers.Conv2DLayer(network['conv_12'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_14'] = lasagne.layers.Conv2DLayer(network['conv_13'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_15'] = lasagne.layers.Conv2DLayer(network['conv_14'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['conv_16'] = lasagne.layers.Conv2DLayer(network['conv_15'], 
                    num_filters=512, filter_size=3, nonlinearity=lasagne.nonlinearities.tanh)


network['dense_1'] = lasagne.layers.DenseLayer(network['conv_16'], 
                    num_units=1000, nonlinearity=lasagne.nonlinearities.tanh)
                    
network['dense_2'] = lasagne.layers.DenseLayer(network['dense_1'], 
                    num_units=1000, nonlinearity=lasagne.nonlinearities.tanh)

network['l_out'] = lasagne.layers.DenseLayer(network['dense_2'], num_units = 10, 
              nonlinearity=T.nnet.softmax)

print "Network has been built"

print "loading VGG19 Pre-Trained weights"

VGG19_Weights = data_loader.loadVGG19();

print "VGG19 Pre-Trained Weights has been loaded"

train_prediction = lasagne.layers.get_output(network['l_out']);

train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, y)
train_loss = train_loss.mean();

train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), y), 
                   dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(network['l_out'], trainable = True)

updates = lasagne.updates.adadelta(train_loss,
            params, learning_rate=0.01, momentum=0.9)
            
# train loss function    
train_fn = theano.function([X, y], [train_loss, train_acc], updates=updates)

print "Train 'Prediction', 'Loss' 'Acc', 'Updates' and 'Train_fn' are ready"

# ##############################################################
# Test side
# Test prediction
test_prediction = lasagne.layers.get_output(network['l_out'], deterministic=True)
f = theano.function([X], test_prediction)

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

X_train, y_train, X_val, y_val = data_loader.load_small_train(shuffle=True,
    data_count=100, image_size=IMAGE_SIZE);

print "Datasests are loaded"

##############################################################
######################## Training ############################

print "Starting Training..."

for epoch in range(N_EPOCH):
    train_err = 0;
    train_acc = 0;    
    train_batches = 0;
    start_time = time.time();
    
    
    
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
    print("Epoch {} of {} took {:.2f}s".format(
        epoch + 1, N_EPOCH, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
        
        

#######################################################################
############################## Testing ################################
#######################################################################
#test.test(f=f, image_size=IMAGE_SIZE)
























print("{:.3f}s Runtime".format(time.time() - st_time));