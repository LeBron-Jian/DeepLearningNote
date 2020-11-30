import tensorflow as tf
# print(tensorflow.__version__)  # 1.14.0
from keras.models import load_model
from tensorflow.python.framework import graph_util
from tensorflow import lite
from keras import backend as K
import os
 
 
def h5_to_pb(h5_file, output_dir, model_name, out_prefix="output_"):
    h5_model = load_model(h5_file, custom_objects={'contrastive_loss': contrastive_loss})
    print(h5_model.input)
    # [<tf.Tensor 'input_2:0' shape=(?, 80, 80) dtype=float32>, <tf.Tensor 'input_3:0' shape=(?, 80, 80) dtype=float32>]
    print(h5_model.output)  # [<tf.Tensor 'lambda_1/Sqrt:0' shape=(?, 1) dtype=float32>]
    print(len(h5_model.outputs))  # 1
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        # print(out_nodes)  # ['output_1']
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    with tf.gfile.GFile(os.path.join(output_dir, model_name), "wb") as filemodel:
        filemodel.write(main_graph.SerializeToString())
    print("pb model: ", {os.path.join(output_dir, model_name)})
 
 
def pb_to_tflite(pb_file, tflite_file):
    inputs = ["input_1"]  # 模型文件的输入节点名称
    classes = ["output_1"]  # 模型文件的输出节点名称
    converter = tf.lite.TocoConverter.from_frozen_graph(pb_file, inputs, classes)
    tflite_model = converter.convert()
    with open(tflite_file, "wb") as f:
        f.write(tflite_model)
 
 
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
 
 
def h5_to_tflite(h5_file, tflite_file):
    converter = lite.TFLiteConverter.from_keras_model_file(h5_file,
                                                           custom_objects={'contrastive_loss': contrastive_loss})
    tflite_model = converter.convert()
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
 
 
if __name__ == '__main__':
    h5_file = 'screw_10.h5'
    tflite_file = 'screw_10.tflite'
    pb_file = 'screw_10.pb'
    # h5_to_tflite(h5_file, tflite_file)
    h5_to_pb(h5_file=h5_file, model_name=pb_file, output_dir='', )
    # pb_to_tflite(pb_file, tflite_file)
