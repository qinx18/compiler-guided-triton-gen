import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize y[i] = 0 for all i < N
        y_offsets = tl.arange(0, BLOCK_SIZE)
        for start_j in range(0, N, BLOCK_SIZE):
            current_y_offsets = start_j + y_offsets
            mask = current_y_offsets < N
            tl.store(y_ptr + current_y_offsets, 0.0, mask=mask)
    
    # Main computation loop over i (rows of A)
    for i in range(M):
        if pid == 0:
            # Initialize tmp[i] = 0.0
            tl.store(tmp_ptr + i, 0.0)
        
        tl.debug_barrier()
        
        # First inner loop: tmp[i] += A[i][j] * x[j] for all j
        if pid == 0:
            tmp_val = 0.0
            j_offsets = tl.arange(0, BLOCK_SIZE)
            
            for start_j in range(0, N, BLOCK_SIZE):
                current_j_offsets = start_j + j_offsets
                mask = current_j_offsets < N
                
                # Load A[i][j] values
                A_offsets = i * N + current_j_offsets
                A_vals = tl.load(A_ptr + A_offsets, mask=mask, other=0.0)
                
                # Load x[j] values
                x_vals = tl.load(x_ptr + current_j_offsets, mask=mask, other=0.0)
                
                # Accumulate tmp[i]
                products = A_vals * x_vals
                tmp_val += tl.sum(products)
            
            # Store tmp[i]
            tl.store(tmp_ptr + i, tmp_val)
        
        tl.debug_barrier()
        
        # Second inner loop: y[j] += A[i][j] * tmp[i] for all j
        if pid == 0:
            # Load tmp[i]
            tmp_i = tl.load(tmp_ptr + i)
            j_offsets = tl.arange(0, BLOCK_SIZE)
            
            for start_j in range(0, N, BLOCK_SIZE):
                current_j_offsets = start_j + j_offsets
                mask = current_j_offsets < N
                
                # Load A[i][j] values
                A_offsets = i * N + current_j_offsets
                A_vals = tl.load(A_ptr + A_offsets, mask=mask, other=0.0)
                
                # Load current y[j] values
                y_vals = tl.load(y_ptr + current_j_offsets, mask=mask, other=0.0)
                
                # Update y[j] = y[j] + A[i][j] * tmp[i]
                y_updates = y_vals + A_vals * tmp_i
                tl.store(y_ptr + current_j_offsets, y_updates, mask=mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    
    # Launch with single program
    grid = (1,)
    
    atax_kernel[grid](
        A, tmp, x, y, M, N, BLOCK_SIZE
    )