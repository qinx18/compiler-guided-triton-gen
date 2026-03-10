import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum_ptr, r, q, NP: tl.constexpr, BLOCK_P: tl.constexpr):
    # Get program ID for p dimension
    pid_p = tl.program_id(0)
    
    # Calculate offsets for p dimension
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offsets < NP
    
    # Initialize sum values to 0
    sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
    
    # Compute sum[p] = sum over s of A[r][q][s] * C4[s][p]
    for s in range(NP):
        # Load A[r][q][s] - this is a scalar for all p values
        a_idx = r * (NQ * NP) + q * NP + s
        a_val = tl.load(A + a_idx)
        
        # Load C4[s][p] for current p block
        c4_offsets = s * NP + p_offsets
        c4_vals = tl.load(C4 + c4_offsets, mask=p_mask, other=0.0)
        
        # Accumulate sum
        sum_vals += a_val * c4_vals
    
    # Store sum values
    sum_offsets = p_offsets
    tl.store(sum_ptr + sum_offsets, sum_vals, mask=p_mask)
    
    # Store back to A[r][q][p]
    a_offsets = r * (NQ * NP) + q * NP + p_offsets
    tl.store(A + a_offsets, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_P = 32
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    C4 = C4.contiguous()
    sum = sum.contiguous()
    
    # Launch kernel for each (r, q) pair
    for r in range(NR):
        for q in range(NQ):
            grid = (triton.cdiv(NP, BLOCK_P),)
            doitgen_kernel[grid](
                A, C4, sum,
                r, q,
                NP=NP,
                BLOCK_P=BLOCK_P
            )