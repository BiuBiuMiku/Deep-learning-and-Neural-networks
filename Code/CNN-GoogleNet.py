import torch
from torch import nn
from torch.nn import functional as F


class Inception(nn.Module):
    """
    in_channels：输入通道数。
    c1：路径1的最终输出通道数。
    c2：(路径2先降到多少通道, 路径2最终输出通道数)。
    c3：(路径3先降到多少通道, 路径3最终输出通道数)。
    c4：路径4的最终输出通道数。
    """

    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()

        # 路径1：1×1卷积，处理当前位置的通道。
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 路径2：先用1×1卷积降通道，再用3×3卷积提取空间特征。
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 路径3：先用1×1卷积降通道，再用5×5卷积观察更大区域。
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 路径4：先最大汇聚，再用1×1卷积调整通道数。
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # 四条路径接收同一份输入x，并分别计算。
        p1 = F.relu(self.p1_1(x))

        p2 = F.relu(self.p2_1(x))
        p2 = F.relu(self.p2_2(p2))

        p3 = F.relu(self.p3_1(x))
        p3 = F.relu(self.p3_2(p3))

        p4 = self.p4_1(x)
        p4 = F.relu(self.p4_2(p4))

        # dim=1是通道维度；拼接后通道数是四条路径输出通道数之和。
        return torch.cat((p1, p2, p3, p4), dim=1)


# b1：从1通道灰度图中提取64通道特征，并缩小空间尺寸。
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# b2：先用1×1卷积处理通道，再用3×3卷积增加到192通道。
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# b3：两个Inception块。第一个输出256通道，第二个输出480通道。
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),     # 64+128+32+32=256
    Inception(256, 128, (128, 192), (32, 96), 64),   # 128+192+96+64=480
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# b4：五个Inception块。每行末尾的注释是该块最终输出通道数。
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),    # 512
    Inception(512, 160, (112, 224), (24, 64), 64),   # 512
    Inception(512, 128, (128, 256), (24, 64), 64),   # 512
    Inception(512, 112, (144, 288), (32, 64), 64),   # 528
    Inception(528, 256, (160, 320), (32, 128), 128), # 832
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# b5：再接两个Inception块，然后将每个通道平均为一个数。
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128), # 832
    Inception(832, 384, (192, 384), (48, 128), 128), # 1024
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten()
)

# 最后一层把1024个高级特征映射为10个类别logits。
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 用一张96×96的灰度图检查前向传播和形状。
X = torch.randn((1, 1, 96, 96))
print("输入：", X.shape)

net.eval()
with torch.inference_mode():
    for name, block in zip(("b1", "b2", "b3", "b4", "b5", "输出层"), net):
        X = block(X)
        print(f"{name}：{X.shape}")

print("最终输出是10个logits：", X)