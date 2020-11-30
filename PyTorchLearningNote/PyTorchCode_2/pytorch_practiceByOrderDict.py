#_*_coding:utf-8_*_
import torch
from torch.autograd import Variable
from collections import OrderedDict
 
# 批量输入的数据量
batch_n = 100
# 通过隐藏层后输出的特征数
hidden_layer = 100
# 输入数据的特征个数
input_data = 1000
# 最后输出的分类结果数
output_data = 10
 
models = torch.nn.Sequential(OrderedDict([
    ("Linel",torch.nn.Linear(input_data,hidden_layer)),
    ("ReLU1",torch.nn.ReLU()),
    ("Line2",torch.nn.Linear(hidden_layer,output_data))
])
)
print(models)
