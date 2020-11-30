# 用训练好的模型来预测新的样本
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model
 
 
def predict(model, img_path, target_size):
    img = cv2.imread(img_path)
    if img.shape != target_size:
        img = cv2.resize(img, target_size)
    # print(img.shape)
    x = image.img_to_array(img)
    x *= 1. / 255  # 相当于ImageDataGenerator(rescale=1. / 255)
    x = np.expand_dims(x, axis=0)  # 调整图片维度
    preds = model.predict(x)
    return preds[0]
 
 
if __name__ == '__main__':
    model_path = 'animal.h5'
    model = load_model(model_path)
    target_size = (150, 150)
    img_path = 'data/test/300.jpg'
    res = predict(model, img_path, target_size)
    print(res)
