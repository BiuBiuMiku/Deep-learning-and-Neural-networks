import torch
from d2l import torch as d2l

# 训练样本数、测试样本数、输入特征数和批量大小。
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5

# 真实权重共有200个，并且全部为0.01。
true_w = torch.ones((num_inputs, 1)) * 0.01

# 真实偏置为0.05。
true_b = 0.05

# 生成训练数据，并包装成批量迭代器。
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)

# 生成测试数据，并包装成批量迭代器。
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    # 随机初始化200个权重，并开启梯度记录。
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)

    # 将偏置初始化为0，并开启梯度记录。
    b = torch.zeros(1, requires_grad=True)

    return [w, b]

def l2_penalty(w):
    # 计算1/2乘以所有权重的平方和。
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    # 为当前实验重新初始化参数。
    w, b = init_params()

    # 定义线性回归模型Xw+b。
    net = lambda X: d2l.linreg(X, w, b)

    # 使用平方损失。
    loss = d2l.squared_loss

    # 设置训练轮数和学习率。
    num_epochs, lr = 100, 0.003

    # 创建损失曲线绘制工具。
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 总目标等于预测损失加上L2权重惩罚。
            l = loss(net(X), y) + lambd * l2_penalty(w)

            # 根据总目标计算梯度。
            l.sum().backward()

            # 使用总梯度更新权重和偏置。
            d2l.sgd([w, b], lr, batch_size)

        if (epoch + 1) % 5 == 0:
            # 评估时只比较预测损失，不把惩罚项当成预测错误。
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))

    # 输出训练后权重的整体大小。
    print('w的L2范数是：', torch.norm(w).item())

# lambd=0表示不使用权重衰减，可以观察严重过拟合。
train(lambd=0)

# lambd=3表示使用权重衰减，可以观察测试损失下降。
# 分别运行两个实验时，会显示两张训练曲线。
train(lambd=3)