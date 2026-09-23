import torch
from torch import nn
from d2l import torch as d2l

# 第一隐藏层丢弃20%的激活值。
dropout1 = 0.2

# 第二隐藏层丢弃50%的激活值。
dropout2 = 0.5

# Sequential按照传入顺序执行每一层。
# Flatten把图片展开为784维向量。
# 第一组Linear、ReLU和Dropout构成第一个隐藏层。
# 第二组Linear、ReLU和Dropout构成第二个隐藏层。
# 最后的Linear输出10个类别的logit。
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2), nn.Linear(256, 10))

def init_weights(m):
    # 只有全连接层包含需要在这里初始化的权重。
    if type(m) == nn.Linear:
        # 将权重初始化为均值为0、标准差为0.01的正态随机数。
        nn.init.normal_(m.weight, std=0.01)

# 对网络里的每一个模块调用init_weights。
net.apply(init_weights)

# 每批读取256张图片。
batch_size = 256

# 一共训练10轮。
num_epochs = 10

# SGD学习率设置为0.5。
lr = 0.5

# 加载Fashion-MNIST训练集和测试集。
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 交叉熵内部会处理logits，不需要在模型末尾添加Softmax。
# reduction='none'表示先保留每个样本各自的损失。
loss = nn.CrossEntropyLoss(reduction='none')

# 创建SGD优化器，负责更新模型的全部可训练参数。
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练模型。
# 训练时nn.Dropout自动开启，测试时nn.Dropout自动关闭。
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)