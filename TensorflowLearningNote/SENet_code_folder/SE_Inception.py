def build_SE_model(nb_classes, input_shape=(256, 256, 3)):
    inputs_dim = Input(input_shape)
    x = Inception(include_top=False, weights='imagenet', input_shape=None,
        pooling=max)(inputs_dim)
 
    squeeze = GlobalAveragePooling2D()(x)
 
    excitation = Dense(units=2048//16)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=2048)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 2048))(excitation)
 
    scale = multiply([x, excitation])
 
    x = GlobalAveragePooling2D()(scale)
    dp_1 = Dropout(0.3)(x)
    fc2 = Dense(nb_classes)(dp_1)
    # 此处注意，为Sigmoid函数
    fc2 = Activation('sigmoid')(fc2)
    model = Model(inputs=inputs_dim, outputs=fc2)
    return model
 
 
if __name__ == '__main__':
    model =build_model(nb_classes, input_shape=(im_size1, im_size2, channels))
    opt = Adam(lr=2*1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit（）
