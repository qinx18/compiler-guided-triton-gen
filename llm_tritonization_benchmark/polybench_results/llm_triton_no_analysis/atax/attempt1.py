import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Initialize y[i] = 0 for all i < N
        block_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            y_offsets = block_start + block_offsets
            y_mask = y_offsets < N
            tl.store(y_ptr + y_offsets, 0.0, mask=y_mask)
        
        # Main computation loops
        for i in range(M):
            # tmp[i] = 0.0
            tl.store(tmp_ptr + i, 0.0)
            
            # Compute tmp[i] = sum(A[i][j] * x[j]) for j in range(N)
            tmp_val = 0.0
            for block_start in range(0, N, BLOCK_SIZE):
                j_offsets = block_start + block_offsets
                j_mask = j_offsets < N
                
                A_offsets = i * N + j_offsets
                A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
                x_vals = tl.load(x_ptr + j_offsets, mask=j_mask, other=0.0)
                
                tmp_val += tl.sum(A_vals * x_vals)
            
            tl.store(tmp_ptr + i, tmp_val)
            
            # Update y[j] = y[j] + A[i][j] * tmp[i] for j in range(N)
            for block_start in range(0, N, BLOCK_SIZE):
                j_offsets = block_start + block_offsets
                j_mask = j_offsets < N
                
                A_offsets = i * N + j_offsets
                A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
                y_vals = tl.load(y_ptr + j_offsets, mask=j_mask, other=0.0)
                
                new_y_vals = y_vals + A_vals * tmp_val
                tl.store(y_ptr + j_offsets, new_y_vals, mask=j_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    grid = (1,)
    
    atax_kernel[grid](
        A, tmp, x, y,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )