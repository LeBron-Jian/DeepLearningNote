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
 
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x
