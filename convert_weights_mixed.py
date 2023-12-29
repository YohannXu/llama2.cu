import torch
import onnx
import argparse
import numpy as np


def find_node_by_input(model, name):
    for node in model.graph.node:
        if node.input[0] == name:
            return node


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    model = onnx.load(args.ckpt.replace('.pth', '.onnx'))

    with open(args.ckpt.replace('.pth', '.bin'), 'wb') as f:
        w_embed = ckpt['tok_embeddings.weight']
        x_scale = ckpt['x_quantizer.scale']
        w_embed_int8 = torch.clamp(torch.round(w_embed / x_scale), -127, 127).cpu().numpy().astype(np.int8)
        f.write(w_embed_int8)

        w_norms = []
        wqs = []
        wks = []
        wvs = []
        wos = []
        w_ff_w1s = []
        w_ff_w2s = []
        w_ff_w3s = []

        norm_w_scales = []
        wq_scales = []
        wk_scales = []
        wv_scales = []
        wo_scales = []
        ff_w1_scales = []
        ff_w2_scales = []
        ff_w3_scales = []

        for i in range(32):
            w_attn_norm_name = f'layers.{i}.attention_norm.weight'
            w_ffn_norm_name = f'layers.{i}.ffn_norm.weight'
            w_attn_norm_scale_name = f'layers.{i}.attention_norm.weight_quantizer.scale'
            w_ffn_norm_scale_name = f'layers.{i}.ffn_norm.weight_quantizer.scale'
            w_attn_norm = ckpt[w_attn_norm_name]
            w_attn_norm_scale = ckpt[w_attn_norm_scale_name]
            w_attn_norm_int8 = torch.clamp(torch.round(w_attn_norm / w_attn_norm_scale), -127, 127).cpu().numpy().astype(np.int8)
            w_norms.append(w_attn_norm_int8)
            norm_w_scales.append(w_attn_norm_scale.cpu().numpy())
            w_ffn_norm = ckpt[w_ffn_norm_name]
            w_ffn_norm_scale = ckpt[w_ffn_norm_scale_name]
            w_ffn_norm_int8 = torch.clamp(torch.round(w_ffn_norm / w_ffn_norm_scale), -127, 127).cpu().numpy().astype(np.int8)
            w_norms.append(w_ffn_norm_int8)
            norm_w_scales.append(w_ffn_norm_scale.cpu().numpy())

        w_norm = ckpt['norm.weight']
        norm_scale = ckpt['norm.weight_quantizer.scale']
        w_norm_int8 = torch.clamp(torch.round(w_norm / norm_scale), -127, 127).cpu().numpy().astype(np.int8)
        w_norms.append(w_norm_int8)
        norm_w_scales.append(norm_scale.cpu().numpy())
        f.write(np.concatenate(w_norms))
        del w_norms

        for i in range(32):
            wq_name = f'layers.{i}.attention.wq.weight'
            wq_scale_name = f'layers.{i}.attention.wq.weight_quantizer.scale'

            wq = ckpt[wq_name].T.contiguous()
            wq_scale = ckpt[wq_scale_name]
            wq_int8 = torch.clip(torch.round(wq / wq_scale), -127, 127).cpu().numpy().astype(np.int8)
            wqs.append(wq_int8)
            wq_scales.append(wq_scale.cpu().numpy())
        f.write(np.concatenate(wqs))
        del wqs

        for i in range(32):
            wk_name = f'layers.{i}.attention.wk.weight'
            wk_scale_name = f'layers.{i}.attention.wk.weight_quantizer.scale'

            wk = ckpt[wk_name].T.contiguous()
            wk_scale = ckpt[wk_scale_name]
            wk_int8 = torch.clamp(torch.round(wk / wk_scale), -127, 127).cpu().numpy().astype(np.int8)
            wks.append(wk_int8)
            wk_scales.append(wk_scale.cpu().numpy())
        f.write(np.concatenate(wks))
        del wks

        for i in range(32):
            wv_name = f'layers.{i}.attention.wv.weight'
            wv_scale_name = f'layers.{i}.attention.wv.weight_quantizer.scale'

            wv = ckpt[wv_name].T.contiguous()
            wv_scale = ckpt[wv_scale_name]
            wv_int8 = torch.clamp(torch.round(wv / wv_scale), -127, 127).cpu().numpy().astype(np.int8)
            wvs.append(wv_int8)
            wv_scales.append(wv_scale.cpu().numpy())
        f.write(np.concatenate(wvs))
        del wvs

        for i in range(32):
            wo_name = f'layers.{i}.attention.wo.weight'
            wo_scale_name = f'layers.{i}.attention.wo.weight_quantizer.scale'

            wo = ckpt[wo_name].T.contiguous()
            wo_scale = ckpt[wo_scale_name]
            wo_int8 = torch.clip(torch.round(wo / wo_scale), -127, 127).cpu().numpy().astype(np.int8)
            wos.append(wo_int8)
            wo_scales.append(wo_scale.cpu().numpy())
        f.write(np.concatenate(wos))
        del wos

        for i in range(32):
            w_ff_w1_name = f'layers.{i}.feed_forward.w1.weight'
            w_ff_w1_scale_name = f'layers.{i}.feed_forward.w1.weight_quantizer.scale'
            w_ff_w1 = ckpt[w_ff_w1_name].T.contiguous()
            w_ff_w1_scale = ckpt[w_ff_w1_scale_name]
            w_ff_w1_int8 = torch.clip(torch.round(w_ff_w1 / w_ff_w1_scale), -127, 127).cpu().numpy().astype(np.int8)
            w_ff_w1s.append(w_ff_w1_int8)
            ff_w1_scales.append(w_ff_w1_scale.cpu().numpy())
        f.write(np.concatenate(w_ff_w1s))
        del w_ff_w1s

        for i in range(32):
            w_ff_w2_name = f'layers.{i}.feed_forward.w2.weight'
            w_ff_w2_scale_name = f'layers.{i}.feed_forward.w2.weight_quantizer.scale'
            w_ff_w2 = ckpt[w_ff_w2_name].T.contiguous()
            w_ff_w2_scale = ckpt[w_ff_w2_scale_name]
            w_ff_w2_int8 = torch.clip(torch.round(w_ff_w2 / w_ff_w2_scale), -127, 127).cpu().numpy().astype(np.int8)
            w_ff_w2s.append(w_ff_w2_int8)
            ff_w2_scales.append(w_ff_w2_scale.cpu().numpy())
        f.write(np.concatenate(w_ff_w2s))
        del w_ff_w2s

        for i in range(32):
            w_ff_w3_name = f'layers.{i}.feed_forward.w3.weight'
            w_ff_w3_scale_name = f'layers.{i}.feed_forward.w3.weight_quantizer.scale'
            w_ff_w3 = ckpt[w_ff_w3_name].T.contiguous()
            w_ff_w3_scale = ckpt[w_ff_w3_scale_name]
            w_ff_w3_int8 = torch.clamp(torch.round(w_ff_w3 / w_ff_w3_scale), -127, 127).cpu().numpy().astype(np.int8)
            w_ff_w3s.append(w_ff_w3_int8)
            ff_w3_scales.append(w_ff_w3_scale.cpu().numpy())
        f.write(np.concatenate(w_ff_w3s))
        del w_ff_w3s

        w_output = ckpt['output.weight'].T.contiguous()
        w_output_scale = ckpt['output.weight_quantizer.scale']
        w_output_int8 = torch.clamp(torch.round(w_output / w_output_scale), -127, 127).cpu().numpy().astype(np.int8)
        f.write(w_output_int8)

        norm_x_scales = []
        norm_out_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'QRMSNorm':
                if index < 4:
                    x_scale = ckpt[f'{node.input[0]}.scale'].cpu().numpy()
                    norm_x_scales.append(x_scale)
                out_scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                norm_out_scales.append(out_scale)
                index += 1
        f.write(np.array(norm_x_scales, dtype=np.half))
        f.write(np.array(norm_w_scales, dtype=np.half))
        f.write(np.array(norm_out_scales, dtype=np.half))
        f.write(np.array(wq_scales, dtype=np.half))
        f.write(np.array(wk_scales, dtype=np.half))
        f.write(np.array(wv_scales, dtype=np.half))

        xq_scales = []
        xk_scales = []
        xv_scales = []
        xo_scales = []
        ff_x1_scales = []
        ff_x2_scales = []
        ff_x3_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'QLinear':
                if 'wq' in node.name:
                    reshape_node = find_node_by_input(model, node.output[0])
                    scale = ckpt[f'{reshape_node.output[0]}_quantizer.scale'].cpu().numpy()
                    xq_scales.append(scale)
                elif 'wk' in node.name:
                    reshape_node = find_node_by_input(model, node.output[0])
                    scale = ckpt[f'{reshape_node.output[0]}_quantizer.scale'].cpu().numpy()
                    xk_scales.append(scale)
                elif 'wv' in node.name:
                    reshape_node = find_node_by_input(model, node.output[0])
                    scale = ckpt[f'{reshape_node.output[0]}_quantizer.scale'].cpu().numpy()
                    xv_scales.append(scale)
                elif 'wo' in node.name:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    xo_scales.append(scale)
                elif 'w1' in node.name:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    ff_x1_scales.append(scale)
                elif 'w2' in node.name:
                    if index == 1:
                        ff_x2_scales.append(np.zeros(()).astype(np.half))
                    else:
                        scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                        ff_x2_scales.append(scale)
                    index += 1
                elif 'w3' in node.name:
                    if index == 1:
                        ff_x3_scales.append(np.zeros(()).astype(np.half))
                    else:
                        scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                        ff_x3_scales.append(scale)

        f.write(np.array(xq_scales, dtype=np.half))
        f.write(np.array(xk_scales, dtype=np.half))
        f.write(np.array(xv_scales, dtype=np.half))

        rotary_q_scales = []
        rotary_k_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'RotaryEmb':
                if index % 2 == 0:
                    transpose_node = find_node_by_input(model, node.output[0])
                    scale = ckpt[f'{transpose_node.output[0]}_quantizer.scale'].cpu().numpy()
                    rotary_q_scales.append(scale)
                else:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    rotary_k_scales.append(scale)
                index += 1
        f.write(np.array(rotary_q_scales, dtype=np.half))
        f.write(np.array(rotary_k_scales, dtype=np.half))

        matmul_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'MatMul':
                if index % 2 == 0:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    matmul_scales.append(scale)
                else:
                    transpose_node = find_node_by_input(model, node.output[0])
                    reshape_node = find_node_by_input(model, transpose_node.output[0])
                    scale = ckpt[f'{reshape_node.output[0]}_quantizer.scale'].cpu().numpy()
                    matmul_scales.append(scale)
                index += 1
        f.write(np.array(matmul_scales, dtype=np.half))
        f.write(np.array(wo_scales, dtype=np.half))
        f.write(np.array(xo_scales, dtype=np.half))
        f.write(np.array(ff_w1_scales, dtype=np.half))
        f.write(np.array(ff_x1_scales, dtype=np.half))
        f.write(np.array(ff_w3_scales, dtype=np.half))
        f.write(np.array(ff_x3_scales, dtype=np.half))

        silu_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'SiLU':
                if index == 1:
                    silu_scales.append(np.zeros(()).astype(np.half))
                else:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    silu_scales.append(scale)
                index += 1
        f.write(np.array(silu_scales, dtype=np.half))

        mul_scales = []
        index = 0
        for node in model.graph.node:
            if node.op_type == 'Mul':
                if index == 1:
                    mul_scales.append(np.zeros(()).astype(np.half))
                else:
                    scale = ckpt[f'{node.output[0]}_quantizer.scale'].cpu().numpy()
                    mul_scales.append(scale)
                index += 1
        f.write(np.array(mul_scales, dtype=np.half))
        f.write(np.array(ff_w2_scales, dtype=np.half))
        f.write(np.array(ff_x2_scales, dtype=np.half))
        f.write(w_output_scale.cpu().numpy())