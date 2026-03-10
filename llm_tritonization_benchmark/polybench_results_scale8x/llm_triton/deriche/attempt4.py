import triton
import triton.language as tl
import torch

@triton.jit
def deriche_kernel(
    imgIn_ptr, imgOut_ptr, y2_ptr, yy1_ptr,
    alpha,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
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
    c1 = 1.0
    
    a5 = a1
    a6 = a2
    a7 = a3
    a8 = a4
    c2 = 1.0
    
    # First pass: forward scan for each row
    for i in range(W):
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
    
    # Second pass: backward scan for each row
    for i in range(W):
        yp1 = 0.0
        yp2 = 0.0
        xp1 = 0.0
        xp2 = 0.0
        for j in range(H):
            j_rev = H - 1 - j
            idx = i * H + j_rev
            imgIn_val = tl.load(imgIn_ptr + idx)
            y2_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + idx, y2_val)
            xp2 = xp1
            xp1 = imgIn_val
            yp2 = yp1
            yp1 = y2_val
    
    # Combine results
    for i in range(W):
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = c1 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + idx, imgOut_val)
    
    # Third pass: forward scan for each column
    for j in range(H):
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
    
    # Fourth pass: backward scan for each column
    for j in range(H):
        tp1 = 0.0
        tp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        for i in range(W):
            i_rev = W - 1 - i
            idx = i_rev * H + j
            imgOut_val = tl.load(imgOut_ptr + idx)
            y2_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tl.store(y2_ptr + idx, y2_val)
            tp2 = tp1
            tp1 = imgOut_val
            yp2 = yp1
            yp1 = y2_val
    
    # Final combination
    for i in range(W):
        for j in range(H):
            idx = i * H + j
            yy1_val = tl.load(yy1_ptr + idx)
            y2_val = tl.load(y2_ptr + idx)
            imgOut_val = c2 * (yy1_val + y2_val)
            tl.store(imgOut_ptr + idx, imgOut_val)

def deriche_triton(imgIn, imgOut, y2, yy1, alpha, H, W):
    BLOCK_SIZE = 64
    
    deriche_kernel[(1,)](
        imgIn, imgOut, y2, yy1,
        alpha,
        H, W,
        BLOCK_SIZE
    )