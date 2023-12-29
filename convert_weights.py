import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='weights/llama-7b-chat/consolidated.00.pth')
    parser.add_argument('--out', type=str, default='weights/llama-7b-chat/weight.bin')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')

    with open(args.out, 'wb') as f:
        w_embed = ckpt['tok_embeddings.weight'].half().numpy()
        f.write(w_embed)

        w_norm = ckpt['norm.weight'].half().numpy().flatten()
        f.write(w_norm)

        w_output = ckpt['output.weight'].T.contiguous().half().numpy().flatten()
        f.write(w_output)

        for i in range(32):
            wq_name = f'layers.{i}.attention.wq.weight'
            wk_name = f'layers.{i}.attention.wk.weight'
            wv_name = f'layers.{i}.attention.wv.weight'
            wo_name = f'layers.{i}.attention.wo.weight'
            w_ff_w1_name = f'layers.{i}.feed_forward.w1.weight'
            w_ff_w2_name = f'layers.{i}.feed_forward.w2.weight'
            w_ff_w3_name = f'layers.{i}.feed_forward.w3.weight'
            w_attn_norm_name = f'layers.{i}.attention_norm.weight'
            w_ffn_norm_name = f'layers.{i}.ffn_norm.weight'

            wq = ckpt[wq_name].T.contiguous().half().numpy().flatten()
            f.write(wq)
            wk = ckpt[wk_name].T.contiguous().half().numpy().flatten()
            f.write(wk)
            wv = ckpt[wv_name].T.contiguous().half().numpy().flatten()
            f.write(wv)
            wo = ckpt[wo_name].T.contiguous().half().numpy().flatten()
            f.write(wo)

            w_ff_w1 = ckpt[w_ff_w1_name].T.contiguous().half().numpy().flatten()
            f.write(w_ff_w1)
            w_ff_w2 = ckpt[w_ff_w2_name].T.contiguous().half().numpy().flatten()
            f.write(w_ff_w2)
            w_ff_w3 = ckpt[w_ff_w3_name].T.contiguous().half().numpy().flatten()
            f.write(w_ff_w3)

            w_attn_norm = ckpt[w_attn_norm_name].half().numpy()
            f.write(w_attn_norm)
            w_ffn_norm = ckpt[w_ffn_norm_name].half().numpy()
            f.write(w_ffn_norm)
