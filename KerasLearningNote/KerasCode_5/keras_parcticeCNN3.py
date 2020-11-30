from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.datasets import imdb
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
import keras
from keras.layers import Embedding, LSTM
 
max_features = 10000
# 该数据库含有IMDB的25000条影评，被标记为正面/负面两种评价，影评已被预处理为词下标构成的序列
# y_train和y_test  序列的标签，是一个二值 list
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(X_train.shape, y_train.shape)  # (25000,) (25000,)
 
main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])
 
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
 
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input, auxiliary_input],
              outputs=[main_output, auxiliary_output])
 
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy',
                    'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1, 'aux_output': 0.2})
 
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': headline_labels, 'aux_input': additional_label},
          epochs=50, batch_size=32, verbose=0)
 
model.predict({'main_input': headline_data, 'aux_input': additional_data})
