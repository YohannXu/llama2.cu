import onnx
from onnx import helper, numpy_helper
import torch
import operator


def flatten_args(args):
    res = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            res.extend(flatten_args(arg))
        else:
            res.append(arg)
    return res


def export_onnx(model, save_name):
    model.cpu()
    graph = helper.make_graph([], 'llama2', [], [])
    modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            input_node = helper.make_tensor_value_info(node.name, onnx.TensorProto.FLOAT, [])
            graph.input.append(input_node)
        elif node.op == 'output':
            for arg in flatten_args(node.args):
                output_node = helper.make_tensor_value_info(arg.name, onnx.TensorProto.FLOAT, [])
                graph.output.append(output_node)
        elif node.op == 'call_module':
            class_name = modules[node.target].__class__.__name__
            if class_name in ['FakeQuantizer', 'FixedQuantizer']:
                if len(node.users) == 0:
                    continue
                scale = modules[node.target].scale.detach().numpy()
                scale_init = numpy_helper.from_array(scale, node.target + '.scale')
                graph.initializer.append(scale_init)
                zp = modules[node.target].zero_point.detach().numpy()
                zp_init = numpy_helper.from_array(zp, node.target + '.zero_point')
                graph.initializer.append(zp_init)
                quantizer_node = helper.make_node('QuantizeLinear', [node.args[0].name, scale_init.name, zp_init.name], [node.name], node.name)
                graph.node.append(quantizer_node)
            elif class_name == 'QLinear':
                scale = modules[node.target].weight_quantizer.scale.detach().numpy()
                scale_init = numpy_helper.from_array(scale, node.target + '.weight_quantizer.scale')
                graph.initializer.append(scale_init)
                zp = modules[node.target].weight_quantizer.zero_point.detach().numpy()
                zp_init = numpy_helper.from_array(zp, node.target + '.weight_quantizer.zero_point')
                graph.initializer.append(zp_init)
                if modules[node.target].bias:
                    bias = modules[node.target].bias.detach().numpy()
                    bias_init = numpy_helper.from_array(bias, node.target + '_bias')
                    graph.initializer.append(bias_init)
                    linear_node = helper.make_node('QLinear', [node.args[0].name, scale_init.name, zp_init.name, bias_init.name], [node.name], node.name)
                else:
                    linear_node = helper.make_node('QLinear', [node.args[0].name, scale_init.name, zp_init.name], [node.name], node.name)
                graph.node.append(linear_node)
            elif class_name == 'SiLU':
                silu_node = helper.make_node('SiLU', [node.args[0].name], [node.name], node.name)
                graph.node.append(silu_node)
            elif class_name == 'QRMSNorm':
                weight_scale = modules[node.target].weight_quantizer.scale.detach().numpy()
                weight_scale_init = helper.make_tensor(node.target + '.weight_quantizer.scale', onnx.TensorProto.FLOAT, [1], [weight_scale.tolist()])
                graph.initializer.append(weight_scale_init)
                weight_zp = modules[node.target].weight_quantizer.zero_point.detach().numpy()
                weight_zp_init = helper.make_tensor(node.target + '.weight_quantizer.zero_point', onnx.TensorProto.FLOAT, [1], [weight_zp.tolist()])
                graph.initializer.append(weight_zp_init)
                rmsnorm_node = helper.make_node('QRMSNorm', [node.args[0].name, weight_scale_init.name, weight_zp_init.name], [node.name], node.name)
                graph.node.append(rmsnorm_node)
            elif class_name == 'RotaryEmb':
                rotary_emb_node = helper.make_node('RotaryEmb', [arg.name for arg in node.args], [node.name], node.name)
                graph.node.append(rotary_emb_node)
            elif class_name == 'Score':
                score_node = helper.make_node('Score', [arg.name for arg in node.args[:2]], [node.name], node.name, factor=node.args[2])
                graph.node.append(score_node)
            else:
                print('Unknown class name:', class_name)
        elif node.op == 'call_function':
            if node.target == torch.cat:
                cat_node = helper.make_node('Concat', [arg.name for arg in node.args[0]], [node.name], node.name)
                graph.node.append(cat_node)
            elif node.target == torch.matmul:
                matmul_node = helper.make_node('MatMul', [arg.name for arg in node.args], [node.name], node.name)
                graph.node.append(matmul_node)
            elif node.target == operator.add:
                add_node = helper.make_node('Add', [arg.name for arg in node.args], [node.name], node.name)
                graph.node.append(add_node)
            elif node.target == operator.mul:
                mul_node = helper.make_node('Mul', [arg.name for arg in node.args], [node.name], node.name)
                graph.node.append(mul_node)
            elif node.target == operator.getitem:
                getitem_node = helper.make_node('Slice', [node.args[0].name], [node.name], node.name, index=node.args[1])
                graph.node.append(getitem_node)
            elif node.target == getattr:
                if len(node.users) == 0:
                    continue
                getattr_node = helper.make_node('Shape', [node.args[0].name], [node.name], node.name)
                graph.node.append(getattr_node)
            else:
                print('Unknown function:', node.target)
        elif node.op == 'call_method':
            if node.target == 'reshape':
                inputs = []
                index = 0
                for arg in node.args:
                    if isinstance(arg, torch.fx.node.Node):
                        inputs.append(arg.name)
                    else:
                        constant_init = helper.make_tensor(node.target + f'_constant_{index}', onnx.TensorProto.INT64, [1], [arg])
                        index += 1
                        graph.initializer.append(constant_init)
                        inputs.append(constant_init.name)
                reshape_node = helper.make_node('Reshape', inputs, [node.name], node.name)
                graph.node.append(reshape_node)
            elif node.target == 'transpose':
                inputs = []
                index = 0
                for arg in node.args:
                    if isinstance(arg, torch.fx.node.Node):
                        inputs.append(arg.name)
                    else:
                        constant_init = helper.make_tensor(node.target + f'_constant_{index}', onnx.TensorProto.INT64, [1], [arg])
                        index += 1
                        graph.initializer.append(constant_init)
                        inputs.append(constant_init.name)
                transpose_node = helper.make_node('Transpose', inputs, [node.name], node.name)
                graph.node.append(transpose_node)
            else:
                print('Unknown method:', node.target)
        else:
            print('Unknown op:', node.op)
    model = helper.make_model(graph)
    onnx.save(model, save_name)