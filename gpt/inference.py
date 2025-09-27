import torch

from .core import *


mdoel

model.load_state_dict(torch.load('checkp/nanoGPT_cpkt.pth', weights_only=True))

model.eval()
start_text = "Once upon"

def inference(text: str):
    input_ids = encode(start_text).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(240):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_logits = torch.clamp(next_token_logits, -100, 100)
            probs = F.softmax(next_token_logits, dim=-1) + 1e-8
            probs = probs / probs.sum()
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode(input_ids[0].cpu())