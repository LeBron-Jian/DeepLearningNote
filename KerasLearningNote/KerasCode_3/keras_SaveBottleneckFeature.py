generator = datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
# the predict_generator method returns the output of a model
# given a generator that yields batches of numpy data
bottleneck_feature_train = model.predict_generator(generator, 2000)
# save the output as a Numpy array
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_feature_train)
 
generator = datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
bottleneck_feature_validaion = model.predict_generator(generator, 2000)
# save the output as a Numpy array
np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_feature_validaion)


train_data = np.load(open('bottleneck_features_train.npy'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 1000 + [1] * 1000)
 
validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.array([0] * 400 + [1] * 400)
 
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, nb_epoch=50, batch_size=32,
          validation_data=(validation_data, validation_labels))
model.sample_weights('bottleneck_fc_model.h5')
