import torch


def precompute_rotary_emb(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos.unsqueeze(1).unsqueeze(0).half(), freqs_sin.unsqueeze(1).unsqueeze(0).half()