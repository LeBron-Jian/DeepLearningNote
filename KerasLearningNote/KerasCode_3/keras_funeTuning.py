# （1）导入bottleneck_features数据
train_data = np.load(open('bottleneck_features_train.npy'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 96)  # matt,打标签
 
validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 16)  # matt,打标签
 
# （2）设置标签，并规范成Keras默认格式
train_labels = keras.utils.to_categorical(train_labels, 5)
validation_labels = keras.utils.to_categorical(validation_labels, 5)
 
# （3）写“小网络”的网络结构
model = Sequential()
#train_data.shape[1:]
model.add(Flatten(input_shape=(4,4,512)))# 4*4*512
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))  # 二分类
model.add(Dense(5, activation='softmax'))  # matt,多分类
#model.add(Dense(1))
#model.add(Dense(5))
#model.add(Activation('softmax'))
 
# （4）设置参数并训练
model.compile(loss='categorical_crossentropy',  
# matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])
 
model.fit(train_data, train_labels,
          nb_epoch=50, batch_size=16,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')
