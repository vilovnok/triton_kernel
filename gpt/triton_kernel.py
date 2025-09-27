import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
    output_row_ptr = output_ptr + row_idx * output_row_stride + col_offsets

    logits = tl.load(input_row_ptr, mask=mask, other=float('-inf'))
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0) + 1e-6

    softmax_output = exp_logits / sum_exp_logits
    tl.store(output_row_ptr, softmax_output, mask=mask)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    N, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_offset = x_ptr + row_idx * N + cols
    x = tl.load(x_offset, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    y = (x_centered * rstd) * w + b
    tl.store(y_ptr + row_idx * N + cols, y, mask=mask)

@triton.jit
def cross_entropy_loss_kernel(
    logits_ptr, targets_ptr, loss_ptr, 
    n_classes, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    targets = tl.load(targets_ptr + offsets, mask=mask, other=-1)

    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, logit)

    loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf'))
        exp_logit = tl.exp(logit - row_max)
        row_sum += exp_logit
        loss = tl.where(targets == i, loss - logit + row_max, loss)

    loss += tl.log(row_sum)

    tl.store(loss_ptr + offsets, loss, mask=mask)

@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    sqrt_2_over_pi = 0.7978845608028654 # sqrt(2 / pi)
    coeff = sqrt_2_over_pi * (1 + 0.044715 * x * x)
    y = 0.5 * x * (1 + (x * coeff) / (1 + tl.abs(x * coeff)))

    tl.store(y_ptr + offsets, y, mask=mask)