import triton
import triton.language as tl

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Initialize y to 0 (only first program)
    if pid == 0:
        block_start = 0
        offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            y_ptrs = y + current_offsets
            tl.store(y_ptrs, 0.0, mask=mask)
    
    # Wait for y initialization
    tl.debug_barrier()
    
    # Process rows sequentially (as in original C code)
    for i in range(M):
        # Initialize tmp[i] = 0.0 (only first program)
        if pid == 0:
            tl.store(tmp + i, 0.0)
        
        # Wait for tmp[i] initialization
        tl.debug_barrier()
        
        # Compute tmp[i] += A[i][j] * x[j] across all programs
        block_start = pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        if block_start < N:
            # Load A[i][j] and x[j]
            A_ptrs = A + i * N + current_offsets
            A_vals = tl.load(A_ptrs, mask=mask, other=0.0)
            x_ptrs = x + current_offsets
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Compute partial sum and add to tmp[i]
            partial_sum = tl.sum(A_vals * x_vals)
            tl.atomic_add(tmp + i, partial_sum)
        
        # Wait for tmp[i] computation to complete
        tl.debug_barrier()
        
        # Compute y[j] += A[i][j] * tmp[i] across all programs
        if block_start < N:
            # Load tmp[i]
            tmp_val = tl.load(tmp + i)
            
            # Load A[i][j] (same as before)
            A_ptrs = A + i * N + current_offsets
            A_vals = tl.load(A_ptrs, mask=mask, other=0.0)
            
            # Compute updates to y[j]
            y_updates = A_vals * tmp_val
            
            # Update y[j]
            y_ptrs = y + current_offsets
            tl.atomic_add(y_ptrs, y_updates, mask=mask)
        
        # Wait for y updates to complete before next iteration
        tl.debug_barrier()

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    atax_kernel[grid](A, tmp, x, y, M, N, BLOCK_SIZE)