import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
    p_offsets = tl.arange(0, BLOCK_SIZE)
    s_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process p dimension in blocks
    for p_start in range(0, NP, BLOCK_SIZE):
        p_current = p_start + p_offsets
        p_mask = p_current < NP
        
        # Initialize sum values for this block
        sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute matrix multiplication for this block of p values
        for s_start in range(0, NP, BLOCK_SIZE):
            s_current = s_start + s_offsets
            s_mask = s_current < NP
            
            # Load A[r][q][s_current] values
            a_base = r * (NQ * NP) + q * NP
            a_vals = tl.load(A_ptr + a_base + s_current, mask=s_mask, other=0.0)
            
            # For each p in current block, accumulate sum
            for p_idx in range(BLOCK_SIZE):
                p_actual = p_start + p_idx
                if p_actual >= NP:
                    continue
                    
                # Load C4[s_current][p_actual] values  
                c4_offsets = s_current * NP + p_actual
                c4_vals = tl.load(C4_ptr + c4_offsets, mask=s_mask, other=0.0)
                
                # Compute partial sum
                products = a_vals * c4_vals
                partial_sum = tl.sum(products)
                sum_vals = tl.where(p_offsets == p_idx, sum_vals + partial_sum, sum_vals)
        
        # Store sum values and update A
        sum_base = 0
        tl.store(sum_ptr + sum_base + p_current, sum_vals, mask=p_mask)
        
        # Store back to A[r][q][p_current]
        a_store_base = r * (NQ * NP) + q * NP
        tl.store(A_ptr + a_store_base + p_current, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Launch kernel with one thread block per (r, q) pair
    grid = (NR * NQ,)
    BLOCK_SIZE = 32
    
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR, BLOCK_SIZE
    )