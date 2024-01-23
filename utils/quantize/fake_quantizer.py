import torch
import torch.nn as nn


class FakeQuantizer(nn.Module):
    def __init__(self, observer, mode='calibration'):
        super(FakeQuantizer, self).__init__()
        self.observer = observer
        self.ch_axis = observer.ch_axis
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        assert mode in ['calibration', 'quantization']
        self.mode = mode

    def forward(self, x):
        if self.mode == 'calibration':
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                self.scale, self.zero_point = self.observer(x)

        if self.mode == 'quantization':
            if self.ch_axis == -1:
                x = torch.clamp(torch.round(x / self.scale + self.zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = self.scale * (x - self.zero_point)
            else:
                shape = torch.ones(x.dim(), dtype=torch.int64).tolist()
                shape[self.ch_axis] = -1
                scale = self.scale.view(shape)
                zero_point = self.zero_point.view(shape)
                x = torch.clamp(torch.round(x / scale + zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = scale * (x - zero_point)
        return x


class FixedQuantizer(FakeQuantizer):
    def __init__(self, scale, zero_point, observer):
        super(FixedQuantizer, self).__init__(observer)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        if self.mode == 'quantization':
            if self.ch_axis == -1:
                x = torch.clamp(torch.round(x / self.scale + self.zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = self.scale * (x - self.zero_point)
            else:
                shape = torch.ones(x.dim(), dtype=torch.int64).tolist()
                shape[self.ch_axis] = -1
                scale = self.scale.view(shape)
                zero_point = self.zero_point.view(shape)
                x = torch.clamp(torch.round(x / scale + zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = scale * (x - zero_point)
        return x


class WeightQuantizer(FakeQuantizer):
    def __init__(self, observer):
        super(WeightQuantizer, self).__init__(observer)
        self.pass_flag = False

    def forward(self, x):
        if self.mode == 'calibration':
            if not self.pass_flag:
                self.pass_flag = True
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    self.scale, self.zero_point = self.observer(x)

        if self.mode == 'quantization':
            if self.ch_axis == -1:
                x = torch.clamp(torch.round(x / self.scale + self.zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = self.scale * (x - self.zero_point)
            else:
                shape = torch.ones(x.dim(), dtype=torch.int64).tolist()
                shape[self.ch_axis] = -1
                scale = self.scale.view(shape)
                zero_point = self.zero_point.view(shape)
                x = torch.clamp(torch.round(x / scale + zero_point).int(), self.observer.quant_min, self.observer.quant_max)
                x = scale * (x - zero_point)

        return x
