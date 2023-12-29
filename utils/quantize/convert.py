import operator
import torch
import torch.nn as nn
from models.llama2 import RMSNorm
from .fake_quantizer import FakeQuantizer, FixedQuantizer
from .observer import EmaMinMaxObserver


def set_convert_module(model, name, mod):
    if '.' in name:
        splits = name.split('.')
        module = getattr(model, splits[0])
        set_convert_module(module, '.'.join(splits[1:]), mod)
    else:
        setattr(model, name, mod)


def get_convert_module(model, name):
    if '.' in name:
        splits = name.split('.')
        module = getattr(model, splits[0])
        return get_convert_module(module, '.'.join(splits[1:]))
    else:
        return getattr(model, name)


def convert(model, mappings, quant_functions, quant_modules, input_scale, freq_scale):
    converted_modules = {}
    for name, module in model.named_modules():
        if type(module) in mappings:
            if type(module) == nn.Linear:
                converted_modules[name] = mappings[type(module)](module.weight, module.bias)
            elif type(module) == RMSNorm:
                converted_modules[name] = mappings[type(module)](module.weight, module.eps)
    for name, module in converted_modules.items():
        set_convert_module(model, name, module)

    nodes = list(model.graph.nodes)
    insert_candidate_nodes = []
    for node in nodes:
        if (node.op == 'call_module' and type(get_convert_module(model, node.target)) in quant_modules) or (node.op == 'call_function' and node.target in quant_functions):
            if node.target == torch.matmul:
                insert_candidate_nodes.append(node.args[0])
            else:
                for input_node in node.args:
                    if isinstance(input_node, torch.fx.node.Node):
                        insert_candidate_nodes.append(input_node)
                    elif isinstance(input_node, list):
                        for item in input_node:
                            if isinstance(item, torch.fx.node.Node):
                                insert_candidate_nodes.append(item)

    insert_nodes = []
    for node in insert_candidate_nodes:
        if node.op == 'placeholder' and node.name not in ['x', 'freqs_cos', 'freqs_sin']:
            continue
        if node.target == operator.getitem:
            continue
        if node in insert_nodes:
            continue
        insert_nodes.append(node)

    for insert_node in insert_nodes:
        if insert_node.name == 'x':
            fake_quantizer = FixedQuantizer(input_scale, torch.zeros_like(input_scale), EmaMinMaxObserver(symmetric=True, ch_axis=-1))
        elif insert_node.name in ['freqs_cos', 'freqs_sin']:
            fake_quantizer = FixedQuantizer(freq_scale, torch.zeros_like(freq_scale), EmaMinMaxObserver(symmetric=True, ch_axis=-1))
        elif 'attention_score' in insert_node.name:
            fake_quantizer = FixedQuantizer(torch.tensor(1. / 255, dtype=torch.half), torch.tensor([0.], dtype=torch.half), EmaMinMaxObserver(symmetric=False, ch_axis=-1))
        elif insert_node.name == 'layers_1_feed_forward_silu':
            continue
        elif insert_node.name == 'layers_1_feed_forward_w3':
            continue
        elif insert_node.name == 'layers_1_feed_forward_w2':
            continue
        elif insert_node.name == 'mul_1':
            continue
        elif insert_node.name in [f'add_{i}' for i in range(3, 64)]:
            continue
        else:
            fake_quantizer = FakeQuantizer(EmaMinMaxObserver(symmetric=True, ch_axis=-1, ema_ratio=0.9))
        quantizer_name = insert_node.name + '_quantizer'
        setattr(model, quantizer_name, fake_quantizer)
        with model.graph.inserting_after(insert_node):
            inserted_node = model.graph.create_node('call_module', quantizer_name, (insert_node,), {})
            for node in nodes:
                if len(node.args) and isinstance(node.args[0], list):
                    args = list(node.args[0])
                    for index, item in enumerate(args):
                        if item == insert_node:
                            args[index] = inserted_node
                    node.args = (args,)
                else:
                    args = list(node.args)
                    for index, input_node in enumerate(args):
                        if input_node == insert_node:
                            args[index] = inserted_node
                    node.args = tuple(args)

    model.recompile()
    model.graph.lint()
    return model
