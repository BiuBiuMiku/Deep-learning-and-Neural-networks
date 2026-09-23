import torch
from torch import nn


# ============================================================
# 1. 构造一个VGG块
# ============================================================

def vgg_block(num_convs, in_channels, out_channels):
    """
    构造一个VGG块。

    参数：
    num_convs：
        当前VGG块中包含多少个卷积层。

    in_channels：
        当前VGG块收到的输入通道数。

    out_channels：
        当前VGG块产生的输出通道数。

    VGG块结构：
        重复num_convs次：
            3×3卷积
            ReLU

        最后执行：
            2×2最大汇聚，步幅为2

    形状变化：
        输入：(批量大小, in_channels, 高度, 宽度)
        输出：(批量大小, out_channels, 高度/2, 宽度/2)
    """

    # 用列表暂时保存这个VGG块中的所有层。
    layers = []

    # 添加num_convs个卷积层。
    for _ in range(num_convs):

        # 3×3卷积、填充1、步幅1。
        # 输出高度 = 输入高度 + 2×1 - 3 + 1 = 输入高度。
        # 因此卷积不会改变特征图的高度和宽度。
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        # ReLU给网络加入非线性。
        # ReLU不会改变张量形状。
        layers.append(nn.ReLU())

        # 第一个卷积层执行以后，张量已经有out_channels个通道。
        # 所以后续卷积层的输入通道数必须改为out_channels。
        in_channels = out_channels

    # 每个VGG块最后进行一次空间降采样。
    # 2×2窗口、步幅2，会让高度和宽度分别减半。
    # 最大汇聚不会改变通道数。
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    # 星号把layers列表中的层逐个传入Sequential。
    # Sequential会按照列表顺序依次执行这些层。
    return nn.Sequential(*layers)


# ============================================================
# 2. 根据配置表构造完整VGG网络
# ============================================================

def vgg(conv_arch, input_channels=1, num_classes=10):
    """
    根据配置表构造VGG网络。

    参数：
    conv_arch：
        VGG卷积部分的配置。
        每个元素的形式是：
            (当前块中的卷积层数量, 当前块的输出通道数)

    input_channels：
        输入图片的通道数。
        Fashion-MNIST是灰度图，所以为1。
        RGB图片应该设置为3。

    num_classes：
        最终需要分类的类别数量。
        Fashion-MNIST有10个类别，所以为10。

    注意：
        这份网络按照书中的224×224输入设计。
        五次最大汇聚以后，空间尺寸为7×7。
    """

    # 保存所有VGG卷积块。
    conv_blocks = []

    # 第一个VGG块接收原始图片，所以输入通道数等于input_channels。
    in_channels = input_channels

    # 依次读取配置表，创建每一个VGG块。
    for num_convs, out_channels in conv_arch:

        # 当前块接收in_channels个通道，输出out_channels个通道。
        block = vgg_block(num_convs, in_channels, out_channels)

        # 把当前块添加到网络中。
        conv_blocks.append(block)

        # 当前块的输出通道数，就是下一个块的输入通道数。
        in_channels = out_channels

    # 循环结束后，in_channels就是最后一个VGG块的输出通道数。
    final_channels = in_channels

    # 输入为224×224，连续经过五次步幅为2的汇聚：
    # 224 → 112 → 56 → 28 → 14 → 7
    #
    # 所以最后一个卷积块输出：
    # (批量大小, final_channels, 7, 7)
    #
    # Flatten后，每个样本包含：
    # final_channels × 7 × 7个特征。
    flatten_features = final_channels * 7 * 7

    # 构造全连接分类器。
    classifier_layers = [
        # 把三维特征图展平成一维特征向量。
        nn.Flatten(),

        # 第一层全连接层。
        nn.Linear(flatten_features, 4096),
        nn.ReLU(),

        # 训练时随机屏蔽大约一半激活值，减轻过拟合。
        nn.Dropout(p=0.5),

        # 第二层全连接层。
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        # 输出层产生num_classes个logit。
        # 不添加Softmax，因为CrossEntropyLoss内部会处理。
        nn.Linear(4096, num_classes)
    ]

    # 先执行所有VGG块，再执行全连接分类器。
    return nn.Sequential(*conv_blocks, *classifier_layers)


# ============================================================
# 3. VGG-11配置
# ============================================================

# 每一个二元组表示：
# (当前VGG块包含多少个卷积层, 当前VGG块输出多少个通道)
#
# 卷积层总数：
# 1 + 1 + 2 + 2 + 2 = 8
#
# 再加上3个全连接层：
# 8 + 3 = 11
#
# 因此这个网络叫作VGG-11。
VGG11_ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# ============================================================
# 4. 创建适合Fashion-MNIST的缩小版VGG-11
# ============================================================

# 完整VGG-11参数非常多，训练Fashion-MNIST没有必要。
# 书中把所有卷积块的输出通道数缩小到原来的四分之一。
channel_ratio = 4

# 卷积层数量保持不变，只把输出通道数除以4。
SMALL_VGG11_ARCH = tuple((num_convs, out_channels // channel_ratio) for num_convs, out_channels in VGG11_ARCH)

# 缩小后的配置为：
# ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
net = vgg(SMALL_VGG11_ARCH, input_channels=1, num_classes=10)


# ============================================================
# 5. 检查每个VGG块和全连接层的输出形状
# ============================================================

def print_output_shapes(model, input_shape, number_of_blocks):
    """
    给模型输入一个随机张量，并打印每个顶层模块的输出形状。

    input_shape：
        完整输入形状，例如：
        (批量大小, 通道数, 高度, 宽度)

    number_of_blocks：
        网络中包含多少个VGG块。
    """

    # 根据指定形状创建随机输入。
    X = torch.randn(input_shape)

    # 进入评估模式，使Dropout停止随机屏蔽神经元。
    model.eval()

    # 这里只检查前向传播，不需要计算梯度。
    with torch.no_grad():

        # Sequential会按照顺序返回其中的顶层模块。
        for index, layer in enumerate(model):

            # 执行当前模块。
            X = layer(X)

            # 前number_of_blocks个顶层模块都是VGG块。
            if index < number_of_blocks:
                layer_name = f"VGG块{index + 1}"
            else:
                layer_name = layer.__class__.__name__

            print(f"{layer_name:12s} -> {tuple(X.shape)}")


# ============================================================
# 6. 统计模型参数
# ============================================================

def count_parameters(model):
    """统计模型中所有需要训练的参数数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ============================================================
# 7. 运行检查
# ============================================================

if __name__ == "__main__":

    print("原始VGG-11配置：")
    print(VGG11_ARCH)

    print("\n缩小版VGG-11配置：")
    print(SMALL_VGG11_ARCH)

    print("\n模型结构：")
    print(net)

    print("\n逐层输出形状：")

    # 输入：
    # 1个样本
    # 1个灰度通道
    # 高度224
    # 宽度224
    print_output_shapes(net, input_shape=(1, 1, 224, 224), number_of_blocks=len(SMALL_VGG11_ARCH))

    print("\n需要训练的参数总数：")
    print(count_parameters(net))