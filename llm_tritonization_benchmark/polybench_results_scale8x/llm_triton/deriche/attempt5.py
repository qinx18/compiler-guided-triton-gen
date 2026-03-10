import triton
import triton.language as tl
import torch

@triton.jit
def deriche_kernel_rows(
    imgIn_ptr, yy1_ptr, y2_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr
):
    row = tl.program_id(0)
    
    if row >= W:
        return
    
    # Calculate coefficients
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a1 = k
    a2 = k * exp_neg_alpha * (alpha - 1.0)
    a3 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = -k * exp_neg_2alpha
    b1 = tl.exp(-alpha * tl.log(2.0))
    b2 = -exp_neg_2alpha
    
    # Forward pass for this row
    ym1 = 0.0
    ym2 = 0.0
    xm1 = 0.0
    for j in range(H):
        idx = row * H + j
        imgIn_val = tl.load(imgIn_ptr + idx)
        yy1_val = a1 * imgIn_val + a2 * xm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1_ptr + idx, yy1_val)
        xm1 = imgIn_val
        ym2 = ym1
        ym1 = yy1_val
    
    # Backward pass for this row
    yp1 = 0.0
    yp2 = 0.0
    xp1 = 0.0
    xp2 = 0.0
    for j in range(H):
        j_rev = H - 1 - j
        idx = row * H + j_rev
        imgIn_val = tl.load(imgIn_ptr + idx)
        y2_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
        tl.store(y2_ptr + idx, y2_val)
        xp2 = xp1
        xp1 = imgIn_val
        yp2 = yp1
        yp1 = y2_val

@triton.jit
def deriche_kernel_combine(
    yy1_ptr, y2_ptr, imgOut_ptr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < W * H
    
    yy1_vals = tl.load(yy1_ptr + offsets, mask=mask)
    y2_vals = tl.load(y2_ptr + offsets, mask=mask)
    imgOut_vals = yy1_vals + y2_vals
    tl.store(imgOut_ptr + offsets, imgOut_vals, mask=mask)

@triton.jit
def deriche_kernel_cols(
    imgOut_ptr, yy1_ptr, y2_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr
):
    col = tl.program_id(0)
    
    if col >= H:
        return
    
    # Calculate coefficients
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    exp_2alpha = tl.exp(2.0 * alpha)
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - exp_2alpha)
    a5 = k
    a6 = k * exp_neg_alpha * (alpha - 1.0)
    a7 = k * exp_neg_alpha * (alpha + 1.0)
    a8 = -k * exp_neg_2alpha
    b1 = tl.exp(-alpha * tl.log(2.0))
    b2 = -exp_neg_2alpha
    
    # Forward pass for this column
    tm1 = 0.0
    ym1 = 0.0
    ym2 = 0.0
    for i in range(W):
        idx = i * H + col
        imgOut_val = tl.load(imgOut_ptr + idx)
        yy1_val = a5 * imgOut_val + a6 * tm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1_ptr + idx, yy1_val)
        tm1 = imgOut_val
        ym2 = ym1
        ym1 = yy1_val
    
    # Backward pass for this column
    tp1 = 0.0
    tp2 = 0.0
    yp1 = 0.0
    yp2 = 0.0
    for i in range(W):
        i_rev = W - 1 - i
        idx = i_rev * H + col
        imgOut_val = tl.load(imgOut_ptr + idx)
        y2_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
        tl.store(y2_ptr + idx, y2_val)
        tp2 = tp1
        tp1 = imgOut_val
        yp2 = yp1
        yp1 = y2_val

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 256
    
    # Pass 1: Process rows (forward and backward)
    deriche_kernel_rows[(W,)](
        imgIn, yy1, y2,
        alpha,
        H, W
    )
    
    # Pass 2: Combine row results
    grid_size = triton.cdiv(W * H, BLOCK_SIZE)
    deriche_kernel_combine[(grid_size,)](
        yy1, y2, imgOut,
        H, W,
        BLOCK_SIZE
    )
    
    # Pass 3: Process columns (forward and backward)
    deriche_kernel_cols[(H,)](
        imgOut, yy1, y2,
        alpha,
        H, W
    )
    
    # Pass 4: Final combine
    deriche_kernel_combine[(grid_size,)](
        yy1, y2, imgOut,
        H, W,
        BLOCK_SIZE
    )