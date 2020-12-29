import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.activ = nn.ReLU()
        self.bnorm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(self.activ(self.bnorm(x)))        
        out = torch.cat([x, out], 1)

        return out


class BNDense(nn.Module):
    def __init__(self, input_size, output_size, bn_size):
        super(BNDense, self).__init__()
        self.activ = nn.ReLU()
	self.bnorm_1 = nn.BatchNorm1d(input_size)
	self.bnorm_2 = nn.BatchNorm1d(output_size * bn_size)
        self.lin_1 = nn.Linear(input_size, output_size * bn_size)
        self.lin_2 = nn.Linear(output_size * bn_size, output_size)

    def forward(self, x):
        out = self.lin_1(self.activ(self.bnorm_1(x))
	out = self.lin_2(self.activ(self.bnorm_2(x))
        out = self.lin2(self.activ(out))
        out = torch.cat([x, out], 1)

        return out
