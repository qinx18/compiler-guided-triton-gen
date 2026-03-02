import torch
import triton
import triton.language as tl

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr,
                   BLOCK_SIZE_P: tl.constexpr):
    # Get program IDs for r and q dimensions
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    # Check bounds
    if pid_r >= NR or pid_q >= NQ:
        return
    
    # Pre-compute offsets for p dimension
    p_offsets = tl.arange(0, BLOCK_SIZE_P)
    
    # Initialize sum array for current (r, q) pair
    for p_block_start in range(0, NP, BLOCK_SIZE_P):
        current_p_offsets = p_block_start + p_offsets
        p_mask = current_p_offsets < NP
        
        # Initialize sum[p] = 0.0
        sum_vals = tl.zeros([BLOCK_SIZE_P], dtype=tl.float32)
        
        # Inner loop: sum[p] += A[r][q][s] * C4[s][p] for all s
        for s in range(NP):
            # Load A[r][q][s] (scalar)
            a_idx = pid_r * (NQ * NP) + pid_q * NP + s
            a_val = tl.load(A_ptr + a_idx)
            
            # Load C4[s][p] (vector)
            c4_idx_base = s * NP
            c4_indices = c4_idx_base + current_p_offsets
            c4_vals = tl.load(C4_ptr + c4_indices, mask=p_mask, other=0.0)
            
            # Accumulate: sum[p] += A[r][q][s] * C4[s][p]
            sum_vals += a_val * c4_vals
        
        # Store sum values to temporary array
        sum_indices = current_p_offsets
        tl.store(sum_ptr + sum_indices, sum_vals, mask=p_mask)
    
    # Second loop: A[r][q][p] = sum[p]
    for p_block_start in range(0, NP, BLOCK_SIZE_P):
        current_p_offsets = p_block_start + p_offsets
        p_mask = current_p_offsets < NP
        
        # Load sum values
        sum_vals = tl.load(sum_ptr + current_p_offsets, mask=p_mask, other=0.0)
        
        # Store to A[r][q][p]
        a_idx_base = pid_r * (NQ * NP) + pid_q * NP
        a_indices = a_idx_base + current_p_offsets
        tl.store(A_ptr + a_indices, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Define block size
    BLOCK_SIZE_P = 32
    
    # Launch grid: one program per (r, q) pair
    grid = (NR, NQ)
    
    # Launch kernel
    doitgen_kernel[grid](
        A, C4, sum, 
        NP, NQ, NR,
        BLOCK_SIZE_P
    )