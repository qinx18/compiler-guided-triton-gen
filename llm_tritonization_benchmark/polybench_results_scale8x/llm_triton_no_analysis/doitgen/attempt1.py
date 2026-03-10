import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr):
    # Get program ID for parallelizing over r and q dimensions
    pid = tl.program_id(0)
    
    # Calculate r and q from pid
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
    
    # Compute matrix multiplication: sum[p] = sum_s(A[r][q][s] * C4[s][p])
    for p in range(NP):
        sum_p = 0.0
        
        # Load A[r][q][:] once for this p
        a_base = r * (NQ * NP) + q * NP
        a_vals = tl.load(A_ptr + a_base + s_offsets)
        
        # Load C4[:][p] once for this p
        c4_base = p
        c4_vals = tl.load(C4_ptr + s_offsets * NP + c4_base)
        
        # Compute dot product
        products = a_vals * c4_vals
        sum_p = tl.sum(products)
        
        # Store to sum array
        tl.store(sum_ptr + p, sum_p)
        
        # Store back to A[r][q][p]
        a_store_offset = r * (NQ * NP) + q * NP + p
        tl.store(A_ptr + a_store_offset, sum_p)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Launch kernel with one thread block per (r, q) pair
    grid = (NR * NQ,)
    
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR
    )