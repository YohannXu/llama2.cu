import torch.nn as nn
from typing import List
from torch.fx import Tracer


class CustomTracer(Tracer):
    def __init__(self, leaf_modules: List = []):
        super(CustomTracer, self).__init__()
        assert isinstance(leaf_modules, list), 'leaf_modules must be a list'
        for leaf_module in leaf_modules:
            assert issubclass(leaf_module, nn.Module), 'leaf module must be subclass of nn.Module'
        self.leaf_modules = leaf_modules

    def is_leaf_module(self, module, qualified_name):
        if self.leaf_modules and type(module) in self.leaf_modules:
            return True
        return super().is_leaf_module(module, qualified_name)