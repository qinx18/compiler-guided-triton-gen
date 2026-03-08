import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output element this program handles
    i = pid // NL
    j = pid % NL
    
    # Early exit if out of bounds
    if i >= NI or j >= NL:
        return
    
    # First compute tmp[i][k] = alpha * sum(A[i][m] * B[m][k]) for all k in [0, NJ)
    # We need tmp[i][k] for k in [0, NJ) to compute D[i][j]
    
    # Compute tmp values we need
    for k in range(NJ):
        tmp_val = 0.0
        for m in range(NK):
            a_idx = i * NK + m
            b_idx = m * NJ + k
            a_val = tl.load(A_ptr + a_idx)
            b_val = tl.load(B_ptr + b_idx)
            tmp_val += alpha * a_val * b_val
        
        # Store tmp[i][k]
        tmp_idx = i * NJ + k
        tl.store(tmp_ptr + tmp_idx, tmp_val)
    
    # Now compute D[i][j] = beta * D[i][j] + sum(tmp[i][k] * C[k][j])
    d_idx = i * NL + j
    d_val = tl.load(D_ptr + d_idx)
    d_val *= beta
    
    for k in range(NJ):
        tmp_idx = i * NJ + k
        c_idx = k * NL + j
        tmp_val = tl.load(tmp_ptr + tmp_idx)
        c_val = tl.load(C_ptr + c_idx)
        d_val += tmp_val * c_val
    
    tl.store(D_ptr + d_idx, d_val)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    # Grid size: one thread per output element in D
    grid_size = NI * NL
    BLOCK_SIZE = 1
    
    k2mm_kernel[(grid_size,)](
        A, B, C, D, tmp, alpha, beta,
        NI=NI, NJ=NJ, NK=NK, NL=NL,
        BLOCK_SIZE=BLOCK_SIZE
    )