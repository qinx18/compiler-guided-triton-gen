import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the r-q pair
    pid = tl.program_id(0)
    r = pid // NQ
    q = pid % NQ
    
    # Early exit if out of bounds
    if r >= NR:
        return
    if q >= NQ:
        return
    
    # Process NP elements in blocks
    for block_start in range(0, NP, BLOCK_SIZE):
        p_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        p_mask = p_offsets < NP
        
        # Initialize sum array for this block
        sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute matrix-vector multiplication
        for s in range(NP):
            # Load A[r][q][s] (broadcast to block)
            a_offset = r * (NQ * NP) + q * NP + s
            a_val = tl.load(A_ptr + a_offset)
            
            # Load C4[s][p] for current block
            c4_offsets = s * NP + p_offsets
            c4_vals = tl.load(C4_ptr + c4_offsets, mask=p_mask, other=0.0)
            
            # Accumulate: sum[p] += A[r][q][s] * C4[s][p]
            sum_vals += a_val * c4_vals
        
        # Store results back to A[r][q][p]
        a_offsets = r * (NQ * NP) + q * NP + p_offsets
        tl.store(A_ptr + a_offsets, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_SIZE = 32
    
    # Grid configuration: one thread block per (r, q) pair
    grid = (NR * NQ,)
    
    # Launch kernel
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR, BLOCK_SIZE
    )