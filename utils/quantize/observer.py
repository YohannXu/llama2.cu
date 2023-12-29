import torch
import torch.nn as nn


class Observer(nn.Module):
    def __init__(self, symmetric=True, ch_axis=-1):
        super(Observer, self).__init__()
        if symmetric:
            self.quant_min, self.quant_max = -127, 127
        else:
            self.quant_min, self.quant_max = 0, 255
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.min_val = None
        self.max_val = None

    def calc_qparams(self):
        max_val = torch.max(self.max_val, torch.zeros_like(self.max_val))
        min_val = torch.min(self.min_val, torch.zeros_like(self.min_val))

        scale = (max_val - min_val) / (self.quant_max - self.quant_min)

        if self.symmetric:
            zero_point = torch.zeros_like(scale, dtype=torch.int)
        else:
            zero_point = torch.clamp(torch.round(self.quant_min - self.min_val / scale).int(), self.quant_min, self.quant_max)

        return scale, zero_point

    def calc_min_max_value(self, x):
        if self.ch_axis == -1:
            if self.symmetric:
                x_ = x.detach()
                max_val = x_.abs().max()
                del x_
                min_val = -max_val
            else:
                x_ = x.detach()
                max_val = x_.max()
                min_val = x_.min()
                del x_
        else:
            index = list(range(x.dim()))
            index[0], index[self.ch_axis] = index[self.ch_axis], index[0]
            x = x.permute(index).flatten(start_dim=1)
            if self.symmetric:
                max_val = x.abs().max(dim=1)[0]
                min_val = -max_val
            else:
                max_val = x.max(dim=1)[0]
                min_val = x.min(dim=1)[0]

        return min_val, max_val


class MinMaxObserver(Observer):
    def forward(self, x):
        min_val, max_val = self.calc_min_max_value(x)

        if self.min_val is None:
            self.min_val = min_val
        else:
            self.min_val = torch.min(self.min_val, min_val)

        if self.max_val is None:
            self.max_val = max_val
        else:
            self.max_val = torch.max(self.max_val, max_val)

        scale, zero_point = self.calc_qparams()
        return scale, zero_point


class EmaMinMaxObserver(Observer):
    def __init__(self, symmetric, ch_axis, ema_ratio=0.9):
        super(EmaMinMaxObserver, self).__init__(symmetric, ch_axis)
        self.ema_ratio = ema_ratio

    def forward(self, x):
        min_val, max_val = self.calc_min_max_value(x)

        if self.min_val is None:
            self.min_val = min_val
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val * (1 - self.ema_ratio)

        if self.max_val is None:
            self.max_val = max_val
        else:
            self.max_val = self.max_val * self.ema_ratio + max_val * (1 - self.ema_ratio)

        scale, zero_point = self.calc_qparams()
        return scale, zero_point
