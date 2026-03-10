import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, r, q, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for parallelizing over p dimension
    pid = tl.program_id(0)
    p_start = pid * BLOCK_SIZE
    
    # Create offsets for vectorized operations
    p_offsets = tl.arange(0, BLOCK_SIZE)
    p_indices = p_start + p_offsets
    p_mask = p_indices < NP
    
    # Initialize sum values
    sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Inner loop over s dimension
    for s in range(NP):
        # Load A[r][q][s]
        a_offset = r * (NQ * NP) + q * NP + s
        a_val = tl.load(A_ptr + a_offset)
        
        # Load C4[s][p_indices]
        c4_offsets = s * NP + p_indices
        c4_vals = tl.load(C4_ptr + c4_offsets, mask=p_mask, other=0.0)
        
        # Accumulate sum[p] += A[r][q][s] * C4[s][p]
        sum_vals += a_val * c4_vals
    
    # Store sum values
    tl.store(sum_ptr + p_indices, sum_vals, mask=p_mask)
    
    # Store back to A[r][q][p]
    a_store_offsets = r * (NQ * NP) + q * NP + p_indices
    tl.store(A_ptr + a_store_offsets, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_SIZE = 64
    
    # Launch kernel for each (r, q) pair
    for r in range(NR):
        for q in range(NQ):
            grid = (triton.cdiv(NP, BLOCK_SIZE),)
            doitgen_kernel[grid](
                A, C4, sum, r, q,
                NP, NQ, NR, BLOCK_SIZE
            )