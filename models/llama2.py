import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryEmb(nn.Module):
    def forward(self, x, freqs_cos, freqs_sin):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)

        x = torch.stack((x[..., 0] * freqs_cos - x[..., 1] * freqs_sin, x[..., 0] * freqs_sin + x[..., 1] * freqs_cos), dim=-1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x


class Score(nn.Module):
    def forward(self, score, mask, factor):
        score *= factor
        score += mask
        score = F.softmax(score, dim=-1).type_as(score)
        return score


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x.half() * self.weight
        return x


class Attention(nn.Module):
    def __init__(self, n_heads, dim):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmb()
        self.score = Score()

        self.dim_sqrt_inv = 1 / math.sqrt(self.head_dim)

    def forward(self, x, k_cache, v_cache, freqs_cos, freqs_sin, mask):
        bs, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bs, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bs, seq_len, self.n_heads, self.head_dim)
        xv = xv.reshape(bs, seq_len, self.n_heads, self.head_dim)

        xq = self.rotary_emb(xq, freqs_cos, freqs_sin)
        xk = self.rotary_emb(xk, freqs_cos, freqs_sin)
        k_cache = torch.cat([k_cache, xk], dim=1)
        v_cache = torch.cat([v_cache, xv], dim=1)

        xq = xq.transpose(1, 2)
        keys = k_cache.transpose(1, 2)
        values = v_cache.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3))
        scores = self.score(scores, mask, self.dim_sqrt_inv)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).reshape(bs, seq_len, self.dim)
        output = self.wo(output)
        return k_cache, v_cache, output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super(FeedForward, self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, dim, multiple_of, norm_eps):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(n_heads, dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = FeedForward(dim, 4 * dim, multiple_of)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, k_cache, v_cache, freqs_cos, freqs_sin, mask):
        shortcut = x
        x = self.attention_norm(x)
        k_cache, v_cache, x = self.attention(x, k_cache, v_cache, freqs_cos, freqs_sin, mask)
        x = x + shortcut
        x = x + self.feed_forward(self.ffn_norm(x))
        return k_cache, v_cache, x


class Llama(nn.Module):
    def __init__(self, dim, n_heads, n_layers, multiple_of, norm_eps, vocab_size):
        super(Llama, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    n_heads,
                    dim,
                    multiple_of,
                    norm_eps
                )
            )

        self.norm = RMSNorm(dim, norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, k_caches, v_caches, freqs_cos, freqs_sin, mask):
        new_k_caches = []
        new_v_caches = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache, x = layer(x, k_caches[i], v_caches[i], freqs_cos, freqs_sin, mask)
            new_k_caches.append(k_cache)
            new_v_caches.append(v_cache)
        x = self.norm(x)
        x = self.output(x)
        return x, new_k_caches, new_v_caches