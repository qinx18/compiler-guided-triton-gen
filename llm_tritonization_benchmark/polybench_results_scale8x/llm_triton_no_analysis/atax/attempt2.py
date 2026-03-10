import triton
import triton.language as tl

@triton.jit
def atax_kernel(A_ptr, tmp_ptr, x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Initialize y array to zero (single thread)
    if pid == 0:
        y_offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_y_offsets = block_start + y_offsets
            mask = current_y_offsets < N
            tl.store(y_ptr + current_y_offsets, 0.0, mask=mask)
    
    # Process rows of A sequentially
    if pid == 0:
        x_offsets = tl.arange(0, BLOCK_SIZE)
        y_offsets = tl.arange(0, BLOCK_SIZE)
        
        for i in range(M):
            # Initialize tmp[i] = 0.0
            tmp_val = 0.0
            
            # Compute tmp[i] = sum(A[i][j] * x[j]) for j in 0..N-1
            for block_start in range(0, N, BLOCK_SIZE):
                current_x_offsets = block_start + x_offsets
                mask = current_x_offsets < N
                
                # Load x values
                x_vals = tl.load(x_ptr + current_x_offsets, mask=mask, other=0.0)
                
                # Load A values for current row
                a_offsets = i * N + current_x_offsets
                a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
                
                # Accumulate tmp[i]
                tmp_val += tl.sum(a_vals * x_vals)
            
            # Store tmp[i]
            tl.store(tmp_ptr + i, tmp_val)
            
            # Update y[j] += A[i][j] * tmp[i] for j in 0..N-1
            for block_start in range(0, N, BLOCK_SIZE):
                current_y_offsets = block_start + y_offsets
                mask = current_y_offsets < N
                
                # Load A values for current row
                a_offsets = i * N + current_y_offsets
                a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
                
                # Load current y values
                y_vals = tl.load(y_ptr + current_y_offsets, mask=mask, other=0.0)
                
                # Update y values
                new_y_vals = y_vals + a_vals * tmp_val
                tl.store(y_ptr + current_y_offsets, new_y_vals, mask=mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 128
    grid = (1,)
    
    atax_kernel[grid](
        A, tmp, x, y,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )