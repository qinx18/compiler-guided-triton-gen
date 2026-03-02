import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Get program ID
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block offsets
    i_offset = pid_i * BLOCK_SIZE_N
    j_offset = pid_j * BLOCK_SIZE_M
    
    # Create offset vectors
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE_N)
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE_M)
    
    # Create masks
    i_mask = i_offsets < N
    j_mask = j_offsets < M
    
    # Initialize s to 0 for this block
    if pid_i == 0:  # Only first i-block initializes s
        s_vals = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        tl.store(s_ptr + j_offsets, s_vals, mask=j_mask)
    
    # Initialize q to 0 for this block  
    if pid_j == 0:  # Only first j-block initializes q
        q_vals = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        tl.store(q_ptr + i_offsets, q_vals, mask=i_mask)
    
    # Create 2D masks for matrix access
    A_mask = i_mask[:, None] & j_mask[None, :]
    
    # Load A block
    A_offsets = i_offsets[:, None] * M + j_offsets[None, :]
    A_vals = tl.load(A_ptr + A_offsets, mask=A_mask)
    
    # Load r and p values
    r_vals = tl.load(r_ptr + i_offsets, mask=i_mask)
    p_vals = tl.load(p_ptr + j_offsets, mask=j_mask)
    
    # Compute updates
    # s[j] += r[i] * A[i][j] - need to sum over i dimension
    r_A = r_vals[:, None] * A_vals
    s_update = tl.sum(r_A, axis=0)
    
    # q[i] += A[i][j] * p[j] - need to sum over j dimension  
    A_p = A_vals * p_vals[None, :]
    q_update = tl.sum(A_p, axis=1)
    
    # Atomically update s and q
    tl.atomic_add(s_ptr + j_offsets, s_update, mask=j_mask)
    tl.atomic_add(q_ptr + i_offsets, q_update, mask=i_mask)

def bicg_triton(A, p, q, r, s, M, N):
    # Block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    # Grid dimensions
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    # Launch kernel
    bicg_kernel[(grid_n, grid_m)](
        A, p, q, r, s, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )