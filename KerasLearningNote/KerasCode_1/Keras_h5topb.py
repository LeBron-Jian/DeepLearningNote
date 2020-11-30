from keras.models import load_model
from tensorflow.python.framework import graph_util
from keras import backend as K
import tensorflow as tf
import os

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
 
h5_file = 'sample_model.h5'
 
h5_model = model.save(h5_file, custom_objects={'contrastive_loss': contrastive_loss})


