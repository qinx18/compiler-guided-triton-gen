import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offset arrays
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate actual indices
    m_indices = block_start_m + offsets_m
    n_indices = block_start_n + offsets_n
    
    # Create masks
    mask_m = m_indices < M
    mask_n = n_indices < N
    
    # Initialize y to 0 (only for the first M block)
    if pid_m == 0:
        y_vals = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        tl.store(y + n_indices, y_vals, mask=mask_n)
    
    # Wait for all blocks to finish initialization
    tl.debug_barrier()
    
    # Process each row i sequentially (cannot parallelize due to tmp dependency)
    for i in range(M):
        # Compute tmp[i] = sum(A[i][j] * x[j]) for current block
        if pid_m == 0:  # Only one block computes tmp
            # Load A[i, :] for current block
            a_ptrs = A + i * N + n_indices
            a_vals = tl.load(a_ptrs, mask=mask_n, other=0.0)
            
            # Load x values
            x_vals = tl.load(x + n_indices, mask=mask_n, other=0.0)
            
            # Compute partial sum
            partial_tmp = tl.sum(a_vals * x_vals)
            
            # Atomic add to tmp[i] (in case of multiple blocks processing same row)
            tl.atomic_add(tmp + i, partial_tmp)
    
    # Wait for tmp computation to complete
    tl.debug_barrier()
    
    # Second phase: update y
    for i in range(M):
        # Load tmp[i]
        tmp_val = tl.load(tmp + i)
        
        # Load A[i, :] for current block
        a_ptrs = A + i * N + n_indices
        a_vals = tl.load(a_ptrs, mask=mask_n, other=0.0)
        
        # Update y values
        y_update = a_vals * tmp_val
        
        # Atomic add to y
        y_ptrs = y + n_indices
        tl.atomic_add(y_ptrs, y_update, mask=mask_n)

def atax_triton(A, tmp, x, y, M, N):
    # Choose block sizes
    BLOCK_SIZE_M = min(32, M)
    BLOCK_SIZE_N = min(64, N)
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    # Zero out tmp and y arrays
    tmp.zero_()
    y.zero_()
    
    # Launch kernel
    atax_kernel[(grid_m, grid_n)](
        A, tmp, x, y, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )