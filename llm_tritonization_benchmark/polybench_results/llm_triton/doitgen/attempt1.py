import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_P: tl.constexpr):
    # Get program ID for r and q dimensions
    pid = tl.program_id(0)
    r = pid // NQ
    q = pid % NQ
    
    # Check bounds
    if r >= NR or q >= NQ:
        return
    
    # Pre-compute offsets for p dimension
    p_offsets = tl.arange(0, BLOCK_P)
    
    # Process all p values for this (r, q) pair
    for p_block_start in range(0, NP, BLOCK_P):
        p_current = p_block_start + p_offsets
        p_mask = p_current < NP
        
        # Initialize sum for this block of p values
        sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
        
        # Compute sum[p] = sum over s of A[r][q][s] * C4[s][p]
        for s in range(NP):
            # Load A[r][q][s] (scalar)
            a_idx = r * (NQ * NP) + q * NP + s
            a_val = tl.load(A + a_idx)
            
            # Load C4[s][p_current] (vector)
            c4_idx = s * NP + p_current
            c4_vals = tl.load(C4 + c4_idx, mask=p_mask, other=0.0)
            
            # Accumulate
            sum_vals += a_val * c4_vals
        
        # Store sum values
        sum_idx = p_current
        tl.store(sum + sum_idx, sum_vals, mask=p_mask)
        
        # Store back to A[r][q][p]
        a_store_idx = r * (NQ * NP) + q * NP + p_current
        tl.store(A + a_store_idx, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Block size for p dimension
    BLOCK_P = 32
    
    # Grid size: one thread block per (r, q) pair
    grid = (NR * NQ,)
    
    doitgen_kernel[grid](
        A, C4, sum, NP, NQ, NR, BLOCK_P
    )