import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Initialize y to 0
    y_ptrs = y + indices
    y_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(y_ptrs, y_vals, mask=mask)
    
    # Process each row i sequentially
    for i in range(M):
        # Compute tmp[i] = sum(A[i][j] * x[j]) for current block
        a_ptrs = A + i * N + indices
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        x_vals = tl.load(x + indices, mask=mask, other=0.0)
        
        # Compute partial sum for tmp[i]
        partial_tmp = tl.sum(a_vals * x_vals)
        
        # Atomically add to tmp[i]
        tl.atomic_add(tmp + i, partial_tmp)
        
        # Wait for all blocks to finish computing tmp[i]
        tl.debug_barrier()
        
        # Load the complete tmp[i] value
        tmp_val = tl.load(tmp + i)
        
        # Update y values: y[j] += A[i][j] * tmp[i]
        y_update = a_vals * tmp_val
        tl.atomic_add(y_ptrs, y_update, mask=mask)

def atax_triton(A, tmp, x, y, M, N):
    # Choose block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Zero out tmp array
    tmp.zero_()
    
    # Launch kernel
    atax_kernel[grid](
        A, tmp, x, y, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )