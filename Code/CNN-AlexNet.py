import torch
from torch import nn


layers = [
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(),

    nn.Linear(256 * 5 * 5, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10)
]

net = nn.Sequential(*layers)


# 检查每一层的输出形状。
X = torch.randn((1, 1, 224, 224))

for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:12s} -> {tuple(X.shape)}")