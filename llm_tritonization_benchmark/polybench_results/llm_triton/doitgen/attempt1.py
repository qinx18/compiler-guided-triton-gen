import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK_P: tl.constexpr):
    # Get program ID for r and q dimensions
    program_id = tl.program_id(0)
    r = program_id // NQ
    q = program_id % NQ
    
    # Create offsets for p and s dimensions
    p_offsets = tl.arange(0, BLOCK_P)
    s_offsets = tl.arange(0, BLOCK_P)
    
    # Initialize sum array for this (r, q) pair
    sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
    
    # Compute matrix multiplication: sum[p] = sum_s(A[r][q][s] * C4[s][p])
    for s in range(NP):
        # Load A[r][q][s] (scalar broadcasted)
        a_idx = r * (NQ * NP) + q * NP + s
        a_val = tl.load(A_ptr + a_idx)
        
        # Load C4[s][p_offsets]
        c4_idx = s * NP + p_offsets
        p_mask = p_offsets < NP
        c4_vals = tl.load(C4_ptr + c4_idx, mask=p_mask, other=0.0)
        
        # Accumulate sum[p_offsets] += A[r][q][s] * C4[s][p_offsets]
        sum_vals += a_val * c4_vals
    
    # Store sum values to temporary array
    sum_base_idx = (r * NQ + q) * NP
    sum_idx = sum_base_idx + p_offsets
    p_mask = p_offsets < NP
    tl.store(sum_ptr + sum_idx, sum_vals, mask=p_mask)
    
    # Copy sum back to A[r][q][p]
    a_base_idx = r * (NQ * NP) + q * NP
    a_idx = a_base_idx + p_offsets
    tl.store(A_ptr + a_idx, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Launch parameters
    BLOCK_P = triton.next_power_of_2(NP)
    grid = (NR * NQ,)
    
    # Launch kernel
    doitgen_kernel[grid](
        A, C4, sum,
        NP=NP, NQ=NQ, NR=NR,
        BLOCK_P=BLOCK_P
    )