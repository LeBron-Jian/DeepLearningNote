import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape[0])
X_train, X_test = X_train.reshape(X_train.shape[0], 784), X_test.reshape(X_test.shape[0], 784)
X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
X_train /= 255
X_test /= 255
 
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
 
model = Sequential()
model.add(Dense(units=784, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
 
model.fit(X_train, y_train, epochs=5, batch_size=32)
 
score = model.evaluate(X_test, y_test, batch_size=128)
print('loss:', score[0])
print('accu:', score[1])
 
model.predict_classes(X_test, batch_size=128)
