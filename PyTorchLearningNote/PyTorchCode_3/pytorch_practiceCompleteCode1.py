#_*_coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import torch
# torchvision包的主要功能是实现数据的处理，导入和预览等
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
 
start_time = time.time()
# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
 
 
# 首先获取手写数字的训练集和测试集
# root 用于指定数据集在下载之后的存放路径
# transform 用于指定导入数据集需要对数据进行那种变化操作
# train是指定在数据集下载完成后需要载入那部分数据，
# 如果设置为True 则说明载入的是该数据集的训练集部分
# 如果设置为FALSE 则说明载入的是该数据集的测试集部分
data_train = datasets.MNIST(root="./data/",
                           transform = transform,
                            train = True,
                            download = True)
 
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                            train = False)
 
 
#数据预览和数据装载
# 下面对数据进行装载，我们可以将数据的载入理解为对图片的处理，
# 在处理完成后，我们就需要将这些图片打包好送给我们的模型进行训练 了  而装载就是这个打包的过程
# dataset 参数用于指定我们载入的数据集名称
# batch_size参数设置了每个包中的图片数据个数
#  在装载的过程会将数据随机打乱顺序并进打包
data_loader_train = torch.utils.data.DataLoader(dataset =data_train,
                                                batch_size = 64,
                                                shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset =data_test,
                                                batch_size = 64,
                                                shuffle = True)
 
# 在装载完成后，我们可以选取其中一个批次的数据进行预览
images,labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)
 
img = img.numpy().transpose(1,2,0)
 
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
 
img = img*std +mean
# print(labels)
print([labels[i] for i in range(64)])
# 由于matplotlab中的展示图片无法显示，所以现在使用OpenCV中显示图片
# plt.imshow(img)
# cv2.imshow('win',img)
# key_pressed=cv2.waitKey(0)
 
#模型搭建和参数优化
# 在顺利完成数据装载后，我们可以开始编写卷积神经网络模型的搭建和参数优化的代码
#卷积层使用torch.nn.Conv2d类来搭建
# 激活层使用torch.nn.ReLU 类方法来搭建
# 池化层使用torch.nn.MaxPool2d类方法来搭建
# 全连接层使用 torch.nn.Linear 类方法来搭建
 
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
 
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(1024,10)
        )
 
    # 我们通过继承torch.nn.Modeule来构造网络，因为手写数字
    # 识别比较简单，我们只是用了两个卷积层，一个最大池化层，两个全连接层。
    # 在向前传播过程中进行x.view(-1, 14 * 14 * 128)
    # 对参数实现扁平化。最后通过自己self.dense定义的全连接层进行最后的分类
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x
 
 
# 在编写完搭建卷积神经网络模型的代码后，我们可以对模型进行训练和参数进行优化了
# 首先 定义在训练之前使用哪种损失函数和优化函数
# 下面定义了计算损失值的损失函数使用的是交叉熵
# 优化函数使用的额是Adam自适应优化算法
model = Model()
# 将所有的模型参数移动到GPU上
if torch.cuda.is_available():
    model.cuda()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# print(model)
 
# 卷积神经网络模型进行模型训练和参数优化的代码
n_epochs = 5
 
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch  {}/{}".format(epoch, n_epochs))
    print("-"*10)
    for data in data_loader_train:
        X_train , y_train = data
        # 有GPU加下面这行，没有不用加
        # X_train, y_train = X_train.cuda(), y_train.cuda()
        X_train , y_train = Variable(X_train),Variable(y_train)
        # print(y_train)
        outputs = model(X_train)
        # print(outputs)
        _,pred = torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss = cost(outputs,y_train)
 
        loss.backward()
        optimizer.step()
        # running_loss += loss.data[0]
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        # print("ok")
        # print("**************%s"%running_corrrect)
 
    print("train ok ")
    testing_correct = 0
    for data in data_loader_test:
        X_test,y_test = data
        # 有GPU加下面这行，没有不用加
        # X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test,y_test = Variable(X_test),Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs,1)
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)
 
    print( "Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(
                 running_loss / len(data_train),100 * running_correct / len(data_train),
                 100 * testing_correct / len(data_test)))
 
 
stop_time = time.time()
print("time is %s" %(stop_time-start_time))
