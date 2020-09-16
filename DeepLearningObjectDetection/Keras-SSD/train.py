from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing import image
from nets.ssd import SSD300
from nets.ssd_training import MultiboxLoss,Generator
from utils.utils import BBoxUtility
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import cv2
import keras
import os
import sys
if __name__ == "__main__":
    log_dir = "logs/"
    annotation_path = '2007_train.txt'
    
    NUM_CLASSES = 4
    input_shape = (300, 300, 3)
    priors = pickle.load(open('model_data/prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('model_data/ssd_weights.h5', by_name=True, skip_mismatch=True)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    BATCH_SIZE = 4
    gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES, do_crop=True)

    if True:
        model.compile(optimizer=Adam(lr=1e-4),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=5.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=5, 
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    if True:
        model.compile(optimizer=Adam(lr=5e-5),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=50, 
                initial_epoch=5,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])