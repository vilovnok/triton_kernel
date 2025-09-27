import torch

def get_batch(split, model, train_data, val_data, seq_length, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    return x.to(model.token_embedding.weight.device), y.to(model.token_embedding.weight.device)

def estimate_mfu(model, dt, batch_size):
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = sum(p.numel() for p in model.parameters())
    L, H, Q, T = model.num_layers, model.num_heads, model.dim // model.num_heads, model.seq_length
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T * batch_size
    flops_achieved = flops_per_fwdbwd * (1.0/dt)
    flops_promised = 312e12
    mfu = flops_achieved / flops_promised
    return mfu