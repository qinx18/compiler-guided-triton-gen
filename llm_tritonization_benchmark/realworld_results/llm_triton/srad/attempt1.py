import triton
import triton.language as tl
import torch

@triton.jit
def srad_phase0_kernel(
    J_ptr,
    sum_ptr,
    sum2_ptr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    roi_size = (R2 - R1 + 1) * (C2 - C1 + 1)
    mask = indices < roi_size
    
    i_coords = indices // (C2 - C1 + 1) + R1
    j_coords = indices % (C2 - C1 + 1) + C1
    linear_idx = i_coords * COLS + j_coords
    
    vals = tl.load(J_ptr + linear_idx, mask=mask, other=0.0)
    vals2 = vals * vals
    
    tl.store(sum_ptr + indices, vals, mask=mask)
    tl.store(sum2_ptr + indices, vals2, mask=mask)

@triton.jit
def srad_phase1_kernel(
    J_ptr,
    c_ptr,
    dN_ptr,
    dS_ptr,
    dW_ptr,
    dE_ptr,
    iN_ptr,
    iS_ptr,
    jW_ptr,
    jE_ptr,
    q0sqr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    k = block_start + offsets
    
    mask = k < (ROWS * COLS)
    
    i = k // COLS
    j = k % COLS
    
    Jc = tl.load(J_ptr + k, mask=mask, other=0.0)
    
    iN_vals = tl.load(iN_ptr + i, mask=mask, other=0)
    iS_vals = tl.load(iS_ptr + i, mask=mask, other=0)
    jW_vals = tl.load(jW_ptr + j, mask=mask, other=0)
    jE_vals = tl.load(jE_ptr + j, mask=mask, other=0)
    
    north_idx = iN_vals * COLS + j
    south_idx = iS_vals * COLS + j
    west_idx = i * COLS + jW_vals
    east_idx = i * COLS + jE_vals
    
    J_north = tl.load(J_ptr + north_idx, mask=mask, other=0.0)
    J_south = tl.load(J_ptr + south_idx, mask=mask, other=0.0)
    J_west = tl.load(J_ptr + west_idx, mask=mask, other=0.0)
    J_east = tl.load(J_ptr + east_idx, mask=mask, other=0.0)
    
    dN_val = J_north - Jc
    dS_val = J_south - Jc
    dW_val = J_west - Jc
    dE_val = J_east - Jc
    
    tl.store(dN_ptr + k, dN_val, mask=mask)
    tl.store(dS_ptr + k, dS_val, mask=mask)
    tl.store(dW_ptr + k, dW_val, mask=mask)
    tl.store(dE_ptr + k, dE_val, mask=mask)
    
    G2 = (dN_val * dN_val + dS_val * dS_val + dW_val * dW_val + dE_val * dE_val) / (Jc * Jc)
    L = (dN_val + dS_val + dW_val + dE_val) / Jc
    
    num = (0.5 * G2) - ((1.0 / 16.0) * (L * L))
    den = 1.0 + (0.25 * L)
    qsqr = num / (den * den)
    
    den = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr))
    c_val = 1.0 / (1.0 + den)
    
    c_val = tl.where(c_val < 0.0, 0.0, c_val)
    c_val = tl.where(c_val > 1.0, 1.0, c_val)
    
    tl.store(c_ptr + k, c_val, mask=mask)

@triton.jit
def srad_phase2_kernel(
    J_ptr,
    c_ptr,
    dN_ptr,
    dS_ptr,
    dW_ptr,
    dE_ptr,
    iS_ptr,
    jE_ptr,
    lambda_val,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    k = block_start + offsets
    
    mask = k < (ROWS * COLS)
    
    i = k // COLS
    j = k % COLS
    
    iS_vals = tl.load(iS_ptr + i, mask=mask, other=0)
    jE_vals = tl.load(jE_ptr + j, mask=mask, other=0)
    
    cN = tl.load(c_ptr + k, mask=mask, other=0.0)
    cS = tl.load(c_ptr + iS_vals * COLS + j, mask=mask, other=0.0)
    cW = tl.load(c_ptr + k, mask=mask, other=0.0)
    cE = tl.load(c_ptr + i * COLS + jE_vals, mask=mask, other=0.0)
    
    dN_val = tl.load(dN_ptr + k, mask=mask, other=0.0)
    dS_val = tl.load(dS_ptr + k, mask=mask, other=0.0)
    dW_val = tl.load(dW_ptr + k, mask=mask, other=0.0)
    dE_val = tl.load(dE_ptr + k, mask=mask, other=0.0)
    
    D = cN * dN_val + cS * dS_val + cW * dW_val + cE * dE_val
    
    J_old = tl.load(J_ptr + k, mask=mask, other=0.0)
    J_new = J_old + 0.25 * lambda_val * D
    
    tl.store(J_ptr + k, J_new, mask=mask)

def srad_triton(J, c, dE, dN, dS, dW, iN, iS, jE, jW, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS):
    BLOCK_SIZE = 256
    SIZE_R = (R2 - R1 + 1) * (C2 - C1 + 1)
    SIZE_I = ROWS * COLS
    
    sum_tensor = torch.zeros(SIZE_R, dtype=torch.float32, device=J.device)
    sum2_tensor = torch.zeros(SIZE_R, dtype=torch.float32, device=J.device)
    
    for iter in range(NITER):
        # Phase 0: Compute ROI statistics
        grid0 = (triton.cdiv(SIZE_R, BLOCK_SIZE),)
        srad_phase0_kernel[grid0](
            J, sum_tensor, sum2_tensor,
            R1, R2, C1, C2, COLS, BLOCK_SIZE
        )
        
        sum_val = sum_tensor.sum().item()
        sum2_val = sum2_tensor.sum().item()
        meanROI = sum_val / SIZE_R
        varROI = (sum2_val / SIZE_R) - meanROI * meanROI
        q0sqr = varROI / (meanROI * meanROI)
        
        # Phase 1: Directional derivatives + diffusion coefficient
        grid1 = (triton.cdiv(SIZE_I, BLOCK_SIZE),)
        srad_phase1_kernel[grid1](
            J, c, dN, dS, dW, dE, iN, iS, jW, jE,
            q0sqr, ROWS, COLS, BLOCK_SIZE
        )
        
        # Phase 2: Divergence + image update
        grid2 = (triton.cdiv(SIZE_I, BLOCK_SIZE),)
        srad_phase2_kernel[grid2](
            J, c, dN, dS, dW, dE, iS, jE,
            lambda_val, ROWS, COLS, BLOCK_SIZE
        )