# _*_coding:utf-8_*_
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
 
 
def W_init(shape, name=None):
    ''' Initiallize weights as in paper'''
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)
 
 
#  //TODO figure out how to initialize layer biases in keras
def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)
 
 
input_shape = (105, 105, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
# build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                   kernel_initializer=W_init, kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7, 7), activation='relu',
                   kernel_regularizer=l2(2e-4), kernel_initializer=W_init, bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                   bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init, kernel_regularizer=l2(2e-4),
                   bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(
    Dense(4096, activation="sigmoid", kernel_regularizer=l2(1e-3), kernel_initializer=W_init, bias_initializer=b_init))
# encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
# merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0] - x[1])
both = merge([encoded_l, encoded_r], mode=L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input, right_input], output=prediction)
# optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
 
optimizer = Adam(0.00006)
# //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
 
siamese_net.count_params()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
 
    def __init__(self, Xtrain, Xval):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes, self.n_examples, self.w, self.h = Xtrain.shape
        self.n_val, self.n_ex_val, _, _ = Xval.shape
 
    def get_batch(self, n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes, size=(n,), replace=False)
        pairs = [np.zeros((n, self.h, self.w, 1)) for i in range(2)]
        targets = np.zeros((n,))
        targets[n // 2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0, self.n_examples)
            pairs[0][i, :, :, :] = self.Xtrain[category, idx_1].reshape(self.w, self.h, 1)
            idx_2 = rng.randint(0, self.n_examples)
            # pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n // 2 else (category + rng.randint(1, self.n_classes)) % self.n_classes
            pairs[1][i, :, :, :] = self.Xtrain[category_2, idx_2].reshape(self.w, self.h, 1)
        return pairs, targets
 
    def make_oneshot_task(self, N):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = rng.choice(self.n_val, size=(N,), replace=False)
        indices = rng.randint(0, self.n_ex_val, size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_examples, replace=False, size=(2,))
        test_image = np.asarray([self.Xval[true_category, ex1, :, :]] * N).reshape(N, self.w, self.h, 1)
        support_set = self.Xval[categories, indices, :, :]
        support_set[0, :, :] = self.Xval[true_category, ex2]
        support_set = support_set.reshape(N, self.w, self.h, 1)
        pairs = [test_image, support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets
 
    def test_oneshot(self, model, N, k, verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k, N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct += 1
        percent_correct = (100.0 * n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, N))
        return percent_correct
    
    
evaluate_every = 7000
loss_every=300
batch_size = 32
N_way = 20
n_val = 550
siamese_net.load_weights("PATH")
best = 76.0
for i in range(900000):
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving")
            siamese_net.save('PATH')
            best=val_acc
 
    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))
