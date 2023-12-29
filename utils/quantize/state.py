from .fake_quantizer import FakeQuantizer


def enable_calibration(model):
    for module in model.modules():
        if isinstance(module, FakeQuantizer):
            module.mode = 'calibration'


def enable_quantization(model):
    for module in model.modules():
        if isinstance(module, FakeQuantizer):
            module.mode = 'quantization'