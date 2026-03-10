import triton
import triton.language as tl
import torch

@triton.jit
def deriche_pass1_kernel(imgIn_ptr, yy1_ptr, alpha, H: tl.constexpr, W: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= W:
        return
    
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - tl.exp(2.0 * alpha))
    a1 = k
    a2 = k * exp_neg_alpha * (alpha - 1.0)
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
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
def deriche_pass2_kernel(imgIn_ptr, y2_ptr, alpha, H: tl.constexpr, W: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= W:
        return
    
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - tl.exp(2.0 * alpha))
    a3 = k * exp_neg_alpha * (alpha + 1.0)
    a4 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
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
def deriche_combine_kernel(yy1_ptr, y2_ptr, imgOut_ptr, c_val, H: tl.constexpr, W: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= W:
        return
    
    for j in range(H):
        idx = i * H + j
        yy1_val = tl.load(yy1_ptr + idx)
        y2_val = tl.load(y2_ptr + idx)
        imgOut_val = c_val * (yy1_val + y2_val)
        tl.store(imgOut_ptr + idx, imgOut_val)

@triton.jit
def deriche_pass3_kernel(imgOut_ptr, yy1_ptr, alpha, H: tl.constexpr, W: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= H:
        return
    
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - tl.exp(2.0 * alpha))
    a5 = k
    a6 = k * exp_neg_alpha * (alpha - 1.0)
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
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
def deriche_pass4_kernel(imgOut_ptr, y2_ptr, alpha, H: tl.constexpr, W: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= H:
        return
    
    exp_neg_alpha = tl.exp(-alpha)
    exp_neg_2alpha = tl.exp(-2.0 * alpha)
    k = (1.0 - exp_neg_alpha) * (1.0 - exp_neg_alpha) / (1.0 + 2.0 * alpha * exp_neg_alpha - tl.exp(2.0 * alpha))
    a7 = k * exp_neg_alpha * (alpha + 1.0)
    a8 = -k * exp_neg_2alpha
    b1 = tl.exp2(-alpha)
    b2 = -exp_neg_2alpha
    
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
    grid_w = (W,)
    grid_h = (H,)
    
    # Pass 1: Forward pass in j direction for each i
    deriche_pass1_kernel[grid_w](imgIn, yy1, alpha, H, W)
    
    # Pass 2: Backward pass in j direction for each i  
    deriche_pass2_kernel[grid_w](imgIn, y2, alpha, H, W)
    
    # Pass 3: Combine first two passes
    deriche_combine_kernel[grid_w](yy1, y2, imgOut, 1.0, H, W)
    
    # Pass 4: Forward pass in i direction for each j
    deriche_pass3_kernel[grid_h](imgOut, yy1, alpha, H, W)
    
    # Pass 5: Backward pass in i direction for each j
    deriche_pass4_kernel[grid_h](imgOut, y2, alpha, H, W)
    
    # Pass 6: Final combine
    deriche_combine_kernel[grid_w](yy1, y2, imgOut, 1.0, H, W)