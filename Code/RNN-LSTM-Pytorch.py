import torch
from torch import nn
from torch.nn import functional as F


class LSTMLanguageModel(nn.Module):
    """使用nn.LSTM实现的字符级语言模型。"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        num_layers=1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

        # 封装输入门、遗忘门、输出门、记忆元和时间循环
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            batch_first=False
        )

        # 把隐状态转换成词表logits
        self.linear = nn.Linear(
            num_hiddens,
            vocab_size
        )

    def begin_state(self, batch_size, device):
        """
        LSTM初始状态是(H, C)。

        二者形状都是：
            (num_layers, batch_size, num_hiddens)
        """

        H = torch.zeros(
            (
                self.num_layers,
                batch_size,
                self.num_hiddens
            ),
            device=device
        )

        C = torch.zeros(
            (
                self.num_layers,
                batch_size,
                self.num_hiddens
            ),
            device=device
        )

        return H, C

    def forward(self, inputs, state):
        """
        inputs：
            (batch_size, num_steps)

        state：
            (H, C)
        """

        # 转成：
        # (num_steps, batch_size, vocab_size)
        X = F.one_hot(
            inputs.T.long(),
            num_classes=self.vocab_size
        ).float()

        # Y：
        # (num_steps, batch_size, num_hiddens)
        #
        # state：
        # (H, C)
        Y, state = self.lstm(X, state)

        # 合并时间步和批量维
        Y = Y.reshape(
            -1,
            Y.shape[-1]
        )

        # 输出：
        # (num_steps * batch_size, vocab_size)
        logits = self.linear(Y)

        return logits, state


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_size = 28
    num_hiddens = 256
    num_layers = 1
    batch_size = 32
    num_steps = 35

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        num_hiddens=num_hiddens,
        num_layers=num_layers
    ).to(device)

    # 随机生成字符索引，模拟输入和标签
    X = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, num_steps),
        device=device
    )

    Y = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, num_steps),
        device=device
    )

    # 创建初始(H, C)
    state = model.begin_state(
        batch_size=batch_size,
        device=device
    )

    # 前向传播
    logits, new_state = model(X, state)

    H_new, C_new = new_state

    print(f"输入X：{tuple(X.shape)}")
    print(f"logits：{tuple(logits.shape)}")
    print(f"最终H：{tuple(H_new.shape)}")
    print(f"最终C：{tuple(C_new.shape)}")

    # 标签按照时间优先顺序展平
    labels = Y.T.reshape(-1)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.0
    )

    optimizer.zero_grad()

    loss = loss_function(
        logits,
        labels.long()
    )

    loss.backward()

    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    print(f"损失：{loss.item():.4f}")
    print("一次完整的前向传播、反向传播和参数更新已完成。")


if __name__ == "__main__":
    main()