import torch
import torch.nn as nn

from .triton_kernel import *



class TritonSoftmax(nn.Module):
    def forward(self, x):
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])
        x = x.clamp(-100, 100)
        B, N = x.shape
        y = torch.empty_like(x)
        grid = lambda meta: (B,)
        softmax_kernel[grid](
            y, x,
            x.stride(0), y.stride(0), N,
            BLOCK_SIZE=triton.next_power_of_2(N)
        )
        y = y + 1e-8
        y = y / y.sum(dim=-1, keepdim=True)
        return y.view(original_shape)
    
def triton_cross_entropy_loss(logits, targets):
    return TritonCrossEntropyLoss.apply(logits, targets)

class TritonCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        n_elements, n_classes = logits.shape
        loss = torch.empty(n_elements, device=logits.device, dtype=logits.dtype)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        cross_entropy_loss_kernel[grid](
            logits, targets, loss,
            n_classes, n_elements,
            BLOCK_SIZE=1024
        )
        
        ctx.save_for_backward(logits, targets)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        batch_size, n_classes = logits.shape

        logits_exp = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        softmax_output = logits_exp / logits_exp.sum(dim=-1, keepdim=True)

        grad_input = softmax_output.clone()
        grad_input.scatter_add_(1, targets.unsqueeze(1), -torch.ones_like(grad_input))
        grad_input *= grad_output.view(-1, 1) / batch_size

        return grad_input, None


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = eps

    def forward(self, x):
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, "Input shape does not match normalized_shape."
        y = torch.empty_like(x)
        x_ = x.reshape(-1, self.normalized_shape[-1])
        y_ = y.reshape(-1, self.normalized_shape[-1])
        M, N = x_.shape
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
        layer_norm_kernel[grid](
            x_, self.weight, self.bias, y_,
            N, eps=self.eps,
            BLOCK_SIZE=128
        )
        return y

class TritonGELU(nn.Module):
    def forward(self, x):
        n_elements = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](
            x, y, n_elements,
            BLOCK_SIZE=1024
        )
        return y