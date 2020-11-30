#_*_coding:utf-8_*_
import torch
from torch.autograd import Variable
 
# 批量输入的数据量
batch_n = 100
# 通过隐藏层后输出的特征数
hidden_layer = 100
# 输入数据的特征个数
input_data = 1000
# 最后输出的分类结果数
output_data = 10
 
x = Variable(torch.randn(batch_n , input_data) , requires_grad = False)
y = Variable(torch.randn(batch_n , output_data) , requires_grad = False)
 
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad = True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad = True)
 
# 训练次数设置为20
epoch_n = 20
# 将学习效率设置为0.000001
learning_rate = 1e-6
 
for epoch in range(epoch_n):
 
    # y_pred = x.mm(w1).clamp(min= 0 ).mm(w2)
    y_pred = model(x, w1, w2)
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{} , Loss:{:.4f}".format(epoch, loss.data[0]))
 
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
 
    w1.grad.data.zero_()
    w2.grad.data.zero_()
