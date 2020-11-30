from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from keras.models import Model
 
 
def build_model():
    # 构建模型
    # 载入Model 权重 + 网络
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    # print(vgg_model.output_shape[1:])  # (4, 4, 512)
 
    # 网络结构
    # build a classifier model to put on top of the convolution model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001), ))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid'))
    top_model.add(Dense(5, activation='softmax'))
 
    # note that it is neccessary to start with a fully-trained classifier
    # including the top classifier, in order to successfully to do fine-tuning
    # top_model_weights_path 是上一个模型的权重
    # 我们上个模型是将网络的卷积层部分，把全连接以上的部分抛掉，然后在我们自己的训练集
    # 和测试集上跑一边，将得到的输出，记录下来，然后基于输出的特征，我们训练一个全连接层
 
    top_model.load_weights(top_model_weights_path)
 
    # add the model on top of the convolutional base
    # vgg_model.add(top_model)
    # 上面代码是需要将 身子 和 头 组装到一起，我们利用函数式组装即可
    print(base_model.input, base_model.output)  # (?, 150, 150, 3)  (None, 4, 4, 512)
    vgg_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
 
    # 然后将最后一个卷积块前的卷积层参数冻结
    # 普通的模型需要对所有层的weights进行训练调整，但是此处我们只调整VGG16的后面几个卷积层，所以前面的卷积层要冻结起来
    for layer in vgg_model.layers[: 15]:  # # :25 bug   15层之前都是不需训练的
        layer.trainable = False
 
    # compile the model with a SGD/momentum optimizer
    # vgg_model.compile(loss='binary_crossentropy',
    #                   optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #                   metrics=['accuracy'])
    vgg_model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),  # 使用一个非常小的lr来微调
                      metrics=['accuracy'])
    vgg_model.summary()
    return vgg_model
 
 
def generate_data():
    # 然后以很低的学习率去训练
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 图片像素值为0-255，此处都乘以1/255，调整到0-1之间
        shear_range=0.2,  # 斜切
        zoom_range=0.2,  # 方法缩小范围
        horizontal_flip=True  # 水平翻转
    )
 
    # this is the augmentation configuration we will use for testing
    test_datagen = ImageDataGenerator(rescale=1. / 255)
 
    # this is a generator that will read pictures found in subfolders of 'data/train'
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        'data/mytrain',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'  # if we use binary_crossentropy loss, we need binary labels
    )
 
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        'data/mytest',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )
    # 开始用 train set 来微调模型的参数
    print("start to fine-tune my model")
    vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )
 
    # vgg_model.save_weights('vgg_try.h5')
 
 
if __name__ == '__main__':
    nb_train_samples = 500
    nb_validation_samples = 100
    epochs = 50
    batch_size = 4
    top_model_weights_path = 'bottleneck_fc_model.h5'
    vgg_model = build_model()
    generate_data()
