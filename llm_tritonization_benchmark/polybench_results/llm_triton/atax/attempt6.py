import triton
import triton.language as tl

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # Get program IDs for 2D grid
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    # Calculate block boundaries
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    
    # Create offset arrays once
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    
    # Calculate actual indices
    indices_i = block_start_i + offsets_i
    indices_j = block_start_j + offsets_j
    
    # Create masks
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
            
        # Initialize tmp[i] = 0.0
        if pid_j == 0:
            tl.store(tmp + i, 0.0)
        
        # Compute tmp[i] += A[i][j] * x[j] for all j in this block
        A_ptrs = A + i * N + indices_j
        A_vals = tl.load(A_ptrs, mask=mask_j, other=0.0)
        x_ptrs = x + indices_j
        x_vals = tl.load(x_ptrs, mask=mask_j, other=0.0)
        
        partial_sum = tl.sum(A_vals * x_vals)
        if pid_j == 0:
            tl.atomic_add(tmp + i, partial_sum)
        else:
            tl.atomic_add(tmp + i, partial_sum)
    
    # Wait for tmp computation to complete
    tl.debug_barrier()
    
    # Process each row i in this block for y update
    for i_idx in range(BLOCK_SIZE_I):
        i = block_start_i + i_idx
        if i >= M:
            break
            
        # Load tmp[i]
        tmp_val = tl.load(tmp + i)
        
        # Compute y[j] += A[i][j] * tmp[i] for all j in this block
        A_ptrs = A + i * N + indices_j
        A_vals = tl.load(A_ptrs, mask=mask_j, other=0.0)
        
        y_updates = A_vals * tmp_val
        y_ptrs = y + indices_j
        
        # Atomically add to y
        for j_idx in range(BLOCK_SIZE_J):
            j = block_start_j + j_idx
            if j < N:
                tl.atomic_add(y + j, y_updates[j_idx])

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_I), triton.cdiv(N, BLOCK_SIZE_J))
    
    atax_kernel[grid](A, tmp, x, y, M, N, BLOCK_SIZE_I, BLOCK_SIZE_J)