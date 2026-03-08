import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr):
    # Get program ID for the r-q pair
    pid = tl.program_id(0)
    r = pid // NQ
    q = pid % NQ
    
    # Early exit if out of bounds
    if r >= NR:
        return
    if q >= NQ:
        return
    
    # Precompute offsets for vectorized operations
    p_offsets = tl.arange(0, NP)
    s_offsets = tl.arange(0, NP)
    
    # Initialize sum array for this (r, q) pair
    sum_vals = tl.zeros([NP], dtype=tl.float32)
    
    # Compute matrix-vector multiplication: sum[p] = A[r][q][:] @ C4[:][p]
    for p in range(NP):
        # Load A[r][q][s] for all s
        a_base = r * (NQ * NP) + q * NP
        a_offsets = a_base + s_offsets
        a_vals = tl.load(A_ptr + a_offsets)
        
        # Load C4[s][p] for all s
        c4_base = p
        c4_offsets = s_offsets * NP + c4_base
        c4_vals = tl.load(C4_ptr + c4_offsets)
        
        # Compute dot product
        prod = a_vals * c4_vals
        sum_val = tl.sum(prod)
        sum_vals = tl.where(p_offsets == p, sum_val, sum_vals)
    
    # Store results back to A[r][q][p]
    a_base = r * (NQ * NP) + q * NP
    a_offsets = a_base + p_offsets
    tl.store(A_ptr + a_offsets, sum_vals)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Grid configuration: one thread block per (r, q) pair
    grid = (NR * NQ,)
    
    # Launch kernel
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR
    )