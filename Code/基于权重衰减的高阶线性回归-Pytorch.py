def train_concise(wd):
    # 创建一个只有线性层的模型。
    # 输入有200个特征，输出只有1个预测值。
    net = nn.Sequential(nn.Linear(num_inputs, 1))

    # 使用标准正态分布初始化模型的权重和偏置。
    for param in net.parameters():
        param.data.normal_()

    # 返回每个样本各自的平方误差，之后再手动求平均。
    loss = nn.MSELoss(reduction='none')

    # 训练100轮，学习率为0.003。
    num_epochs, lr = 100, 0.003

    # 把权重和偏置分成两个参数组。
    # 第一组是weight，使用wd指定的权重衰减。
    # 第二组是bias，没有设置weight_decay，所以偏置不会衰减。
    # 两组参数共同使用lr指定的学习率。
    trainer = torch.optim.SGD([{'params': net[0].weight, 'weight_decay': wd}, {'params': net[0].bias}], lr=lr)

    # 创建训练损失和测试损失的动画曲线。
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])

    # 开始训练。
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 清空上一批保存在模型参数中的梯度。
            trainer.zero_grad()

            # 前向传播并计算每个样本的平方损失。
            l = loss(net(X), y)

            # 对当前批次的平均损失执行反向传播。
            l.mean().backward()

            # 更新参数。
            # 更新weight时会自动加入权重衰减。
            # 更新bias时不会使用权重衰减。
            trainer.step()

        # 每5轮记录一次训练损失和测试损失。
        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))

    # 输出权重的L2范数，用于检查权重是否被压小。
    print('w的L2范数：', net[0].weight.norm().item())