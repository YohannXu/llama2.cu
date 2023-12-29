import torch

def get_mask(seq_len):
    if seq_len > 1:
        mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
    else:
        mask = torch.zeros((1, 1, 1, 1))
    return mask