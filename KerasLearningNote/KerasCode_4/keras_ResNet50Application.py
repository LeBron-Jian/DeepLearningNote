# 利用ResNet50网络进行 ImageNet 分类
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
 
model = ResNet50(weights='imagenet')
 
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
 
preds = model.predict(x)
# decode the results into a list of tuples(class description, probability)
# one such list for each sample in the batch
print('Predicted:', decode_predictions(preds, top=3)[0])
'''
Predicted: [('n01871265', 'tusker', 0.40863296),
('n02504458', 'African_elephant', 0.36055887),
('n02504013', 'Indian_elephant', 0.22416794)]
'''
