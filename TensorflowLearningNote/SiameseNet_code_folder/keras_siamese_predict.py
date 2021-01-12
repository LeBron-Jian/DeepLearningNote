# load data
def get_image_data(imagePaths, label):
    data = []
    labels = []
    for image_name in os.listdir(imagePaths):
        imagePath = os.path.join(imagePaths, image_name)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, target_size)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype='float')
    data /= 255.0
    labels = np.array(labels)
    data, labels = shuffle(data, labels, random_state=0)
    return data, labels
 
 
def load_train_test_data():
    filelist = []
    for i in os.listdir(file_path):
        filelist.append(i)
    data = []
    labels = []
    for i in range(len(filelist)):
        filedir = filelist[i]
        allpath = os.path.join(file_path, filelist[i])
        data_i, labels_i = get_image_data(imagePaths=allpath, label=filedir)
        data_i, labels_i = list(data_i), list(labels_i)
        data.extend(data_i)
        labels.extend(labels_i)
    data, labels = np.array(data), np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=123456)
    return x_train, x_test, y_train, y_test, filelist
 
 
 
def load_data():
    x_train, x_test, y_train, y_test, filelist = load_train_test_data()
    print(x_train.shape, y_train.shape) 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255.0
    # x_test /= 255.0
    input_shape = x_train.shape[1:]  # (80, 80)
    digit_indices = [np.where(y_train == filelist[i])[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)
    digit_indices = [np.where(y_test == filelist[i])[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    # print(te_pairs.shape, te_y.shape)  # (980, 2, 80, 80) (980,)
    return input_shape, tr_pairs, tr_y, te_pairs, te_y
    

# predict data with model
def predict(model_path, image_path1, image_path2, target_size):
    saved_model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss})
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    # 灰度化，并调整尺寸
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, target_size)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.resize(image2, target_size)  # <class 'numpy.ndarray'>
    # print(image2.shape)  # (80, 80)
    # print(image2)
    # 对图像数据做scale操作
    data1 = np.array([image1], dtype='float') / 255.0 / 255.0
    data2 = np.array([image2], dtype='float') / 255.0 / 255.0
    print(data1.shape, data2.shape)  # (1, 80, 80) (1, 80, 80)
    pairs = np.array([data1, data2])
    print(pairs.shape)  # (2, 80, 80)
 
    y_pred = saved_model.predict([data1, data2])
    print(y_pred)
    # print(y_pred)  # [[4.1023154]]
    # pred = y_pred.ravel() < 0.5
    # print(pred)  # 如果没有 <0.5则为 [4.1023154]  有的话则是  [False]
    # y_true = [1]  # 1表示两个是一个类，0表示不同的类
    # if pred == y_true:
    #     print("是同一类")
    # else:
    #     print("不是同一类")

 
