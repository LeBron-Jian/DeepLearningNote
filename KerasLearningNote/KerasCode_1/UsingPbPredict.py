import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
 

def predict_pb(pb_model_path, image_path1, image_path2, target_size):
    '''
      此处为孪生网络的预测代码
    '''
    sess = tf.Session()
    with gfile.FastGFile(pb_model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    # 输入  这里有两个输入
    input_x = sess.graph.get_tensor_by_name('input_2:0')
    input_y = sess.graph.get_tensor_by_name('input_3:0')
    # 输出
    op = sess.graph.get_tensor_by_name('lambda_1/Sqrt:0')
 
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    # 灰度化，并调整尺寸
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, target_size)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, target_size)
    data1 = np.array([image1], dtype='float') / 255.0
    data2 = np.array([image2], dtype='float') / 255.0
    y_pred = sess.run(op, {input_x: data1, input_y: data2})
    print(y_pred)
