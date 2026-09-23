import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    # 检查丢弃概率是否合法。
    assert 0 <= dropout <= 1

    # 丢弃概率为1时，返回形状相同的全零张量。
    if dropout == 1:
        return torch.zeros_like(X)

    # 丢弃概率为0时，保留原张量。
    if dropout == 0:
        return X

    # 为X中的每个元素生成一个0或1的随机遮罩。
    # 随机数大于dropout的位置保留，因此保留概率为1-dropout。
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()

    # 丢弃部分元素，并缩放保留的元素以维持期望不变。
    return mask * X / (1.0 - dropout)

# 测试自定义Dropout函数。
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print('原始X：')
print(X)
print('dropout=0：')
print(dropout_layer(X, 0.0))
print('dropout=0.5：')
print(dropout_layer(X, 0.5))
print('dropout=1：')
print(dropout_layer(X, 1.0))

# 定义模型维度。
num_inputs, num_outputs = 784, 10
num_hiddens1, num_hiddens2 = 256, 256

# 定义两个隐藏层的丢弃概率。
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        # 初始化父类nn.Module。
        super().__init__()

        # 保存输入维度。
        self.num_inputs = num_inputs

        # 创建三个全连接层。
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)

        # 创建ReLU激活函数。
        self.relu = nn.ReLU()

    def forward(self, X):
        # 展开图片，经过第一全连接层和ReLU。
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))

        # 只在训练模式下对第一隐藏层使用Dropout。
        if self.training:
            H1 = dropout_layer(H1, dropout1)

        # 经过第二全连接层和ReLU。
        H2 = self.relu(self.lin2(H1))

        # 只在训练模式下对第二隐藏层使用Dropout。
        if self.training:
            H2 = dropout_layer(H2, dropout2)

        # 输出10个类别的logits。
        return self.lin3(H2)

# 创建模型。
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 设置训练参数。
num_epochs, lr, batch_size = 10, 0.5, 256

# 创建交叉熵损失。
loss = nn.CrossEntropyLoss(reduction='none')

# 加载Fashion-MNIST。
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 创建SGD优化器。
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练并评估模型。
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)