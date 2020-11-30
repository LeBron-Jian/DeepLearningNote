class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
 
model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
 
history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])
 
print(history.losses)
# 输出
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('acc') > 0.95):
            print("\nReached 95% accuracy so cancelling training !")
            self.model.stop_training = True
