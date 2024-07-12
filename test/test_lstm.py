import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.LSTM(1, 50, num_layers=2, batch_first=True)
        self.ff = nn.Linear(50, 1)

    def forward(self, x):
        out, states = self.enc(x)
        print(out.size())
        logits = self.ff(out)
        print(logits)


model = MyModel()
inp = torch.rand(2, 10, 1)
model(inp)
