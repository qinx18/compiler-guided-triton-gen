import triton
import triton.language as tl
import torch

@triton.jit
def deriche_pass1_kernel(
    imgIn_ptr, yy1_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr,
    a1, a2, b1, b2,
):
    i = tl.program_id(0)
    if i >= W:
        return
    
    ym1 = 0.0
    ym2 = 0.0
    xm1 = 0.0
    
    for j in range(H):
        idx = i * H + j
        imgIn_val = tl.load(imgIn_ptr + idx)
        
        yy1_val = a1 * imgIn_val + a2 * xm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1_ptr + idx, yy1_val)
        
        xm1 = imgIn_val
        ym2 = ym1
        ym1 = yy1_val

@triton.jit
def deriche_pass2_kernel(
    imgIn_ptr, y2_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr,
    a3, a4, b1, b2,
):
    i = tl.program_id(0)
    if i >= W:
        return
    
    yp1 = 0.0
    yp2 = 0.0
    xp1 = 0.0
    xp2 = 0.0
    
    for j in range(H - 1, -1, -1):
        idx = i * H + j
        imgIn_val = tl.load(imgIn_ptr + idx)
        
        y2_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
        tl.store(y2_ptr + idx, y2_val)
        
        xp2 = xp1
        xp1 = imgIn_val
        yp2 = yp1
        yp1 = y2_val

@triton.jit
def deriche_combine_kernel(
    yy1_ptr, y2_ptr, imgOut_ptr,
    H: tl.constexpr, W: tl.constexpr,
    c1,
):
    idx = tl.program_id(0) * tl.program_id(1) + tl.program_id(1)
    total_elements = H * W
    if idx >= total_elements:
        return
    
    yy1_val = tl.load(yy1_ptr + idx)
    y2_val = tl.load(y2_ptr + idx)
    imgOut_val = c1 * (yy1_val + y2_val)
    tl.store(imgOut_ptr + idx, imgOut_val)

@triton.jit
def deriche_pass4_kernel(
    imgOut_ptr, yy1_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr,
    a5, a6, b1, b2,
):
    j = tl.program_id(0)
    if j >= H:
        return
    
    tm1 = 0.0
    ym1 = 0.0
    ym2 = 0.0
    
    for i in range(W):
        idx = i * H + j
        imgOut_val = tl.load(imgOut_ptr + idx)
        
        yy1_val = a5 * imgOut_val + a6 * tm1 + b1 * ym1 + b2 * ym2
        tl.store(yy1_ptr + idx, yy1_val)
        
        tm1 = imgOut_val
        ym2 = ym1
        ym1 = yy1_val

@triton.jit
def deriche_pass5_kernel(
    imgOut_ptr, y2_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr,
    a7, a8, b1, b2,
):
    j = tl.program_id(0)
    if j >= H:
        return
    
    tp1 = 0.0
    tp2 = 0.0
    yp1 = 0.0
    yp2 = 0.0
    
    for i in range(W - 1, -1, -1):
        idx = i * H + j
        imgOut_val = tl.load(imgOut_ptr + idx)
        
        y2_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
        tl.store(y2_ptr + idx, y2_val)
        
        tp2 = tp1
        tp1 = imgOut_val
        yp2 = yp1
        yp1 = y2_val

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    # Calculate coefficients
    exp_neg_alpha = torch.exp(torch.tensor(-alpha))
    exp_neg_2alpha = torch.exp(torch.tensor(-2.0 * alpha))
    
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - torch.exp(torch.tensor(2.0 * alpha)))
    a1 = k
    a2 = k * exp_neg_alpha * (alpha - 1.0)
    a3 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = -k * exp_neg_2alpha
    a5 = k
    a6 = k * exp_neg_alpha * (alpha - 1.0)
    a7 = k * exp_neg_alpha * (alpha + 1.0)
    a8 = -k * exp_neg_2alpha
    b1 = torch.pow(torch.tensor(2.0), -alpha)
    b2 = -exp_neg_2alpha
    c1 = 1.0
    c2 = 1.0
    
    # Pass 1: Forward in j direction for each i
    grid1 = (W,)
    deriche_pass1_kernel[grid1](
        imgIn, yy1, alpha, H, W, a1, a2, b1, b2
    )
    
    # Pass 2: Backward in j direction for each i
    grid2 = (W,)
    deriche_pass2_kernel[grid2](
        imgIn, y2, alpha, H, W, a3, a4, b1, b2
    )
    
    # Pass 3: Combine results
    grid3 = (triton.cdiv(H * W, 256),)
    deriche_combine_kernel[grid3](
        yy1, y2, imgOut, H, W, c1
    )
    
    # Pass 4: Forward in i direction for each j
    grid4 = (H,)
    deriche_pass4_kernel[grid4](
        imgOut, yy1, alpha, H, W, a5, a6, b1, b2
    )
    
    # Pass 5: Backward in i direction for each j
    grid5 = (H,)
    deriche_pass5_kernel[grid5](
        imgOut, y2, alpha, H, W, a7, a8, b1, b2
    )
    
    # Final pass: Combine final results
    grid6 = (triton.cdiv(H * W, 256),)
    deriche_combine_kernel[grid6](
        yy1, y2, imgOut, H, W, c2
    )