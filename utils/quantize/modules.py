import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quantizer import WeightQuantizer
from .observer import MinMaxObserver

class QLinear(nn.Module):
    def __init__(self, weight, bias):
        super(QLinear, self).__init__()
        self.weight = weight
        self.bias = bias
        self.weight_quantizer = WeightQuantizer(MinMaxObserver(symmetric=True, ch_axis=-1))

    def forward(self, x):
        return F.linear(x, self.weight_quantizer(self.weight), self.bias)


class QRMSNorm(nn.Module):
    def __init__(self, weight, eps, w_sym=True):
        super(QRMSNorm, self).__init__()
        self.eps = eps
        self.weight = weight
        self.weight_quantizer = WeightQuantizer(MinMaxObserver(symmetric=w_sym, ch_axis=-1))

    def forward(self, x):
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x.half() * self.weight_quantizer(self.weight)
        return x
