import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block starts and offsets
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    indices_i = block_start_i + offsets_i
    indices_j = block_start_j + offsets_j
    mask_i = indices_i < M
    mask_j = indices_j < N
    
    # Initialize y to 0 (only once)
    if pid_i == 0:
        y_ptrs = y + indices_j
        y_vals = tl.zeros((BLOCK_SIZE_J,), dtype=tl.float32)
        tl.store(y_ptrs, y_vals, mask=mask_j)
    
    # Process each row i in this block
    for i_idx in range(BLOCK_SIZE_I):
        i = block_start_i + i_idx
        if i >= M:
            break
            
        # Initialize tmp[i] to 0
        if pid_j == 0:
            tl.store(tmp + i, 0.0)
        
        # Compute tmp[i] = sum(A[i][j] * x[j]) for current block
        a_ptrs = A + i * N + indices_j
        a_vals = tl.load(a_ptrs, mask=mask_j, other=0.0)
        x_vals = tl.load(x + indices_j, mask=mask_j, other=0.0)
        
        # Compute partial sum for tmp[i]
        partial_tmp = tl.sum(a_vals * x_vals)
        
        # Atomically add to tmp[i]
        tl.atomic_add(tmp + i, partial_tmp)
        
        # Synchronize to ensure tmp[i] is complete
        tl.debug_barrier()
        
        # Load the complete tmp[i] value
        tmp_val = tl.load(tmp + i)
        
        # Update y values: y[j] += A[i][j] * tmp[i]
        y_update = a_vals * tmp_val
        y_ptrs = y + indices_j
        tl.atomic_add(y_ptrs, y_update, mask=mask_j)

def atax_triton(A, tmp, x, y, M, N):
    # Choose block sizes
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 64
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_I), triton.cdiv(N, BLOCK_SIZE_J))
    
    # Launch kernel
    atax_kernel[grid](
        A, tmp, x, y, M, N,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )