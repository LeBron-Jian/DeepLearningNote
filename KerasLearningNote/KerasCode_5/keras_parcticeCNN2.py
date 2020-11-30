from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
 
learning_rate = 0.001
epoch = 10
decay = learning_rate / epoch
sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
 
model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
 
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=20, batch_size=32, verbose=1)
plt.figure(12)
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'val'], loc='upper left')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()
 
score = model.evaluate(X_test, y_test, verbose=1)
print(model.metrics_names)
print('loss:', score[0])
print('accu:', score[1])
 
prediction = model.predict_classes(X_test)
print(prediction[:10])
