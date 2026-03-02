import triton
import triton.language as tl

@triton.jit
def seidel_2d_kernel(A_ptr, N, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    i_base = pid_i * BLOCK_SIZE_I + 1
    j_base = pid_j * BLOCK_SIZE_J + 1
    
    i_indices = i_base + i_offsets
    j_indices = j_base + j_offsets
    
    i_mask = i_indices <= (N - 2)
    j_mask = j_indices <= (N - 2)
    
    for i_idx in range(BLOCK_SIZE_I):
        if i_base + i_idx > N - 2:
            continue
        i = i_base + i_idx
        
        for j_idx in range(BLOCK_SIZE_J):
            if j_base + j_idx > N - 2:
                continue
            j = j_base + j_idx
            
            # Load 9-point stencil values
            val_tl = tl.load(A_ptr + (i-1) * N + (j-1))  # top-left
            val_tc = tl.load(A_ptr + (i-1) * N + j)      # top-center
            val_tr = tl.load(A_ptr + (i-1) * N + (j+1))  # top-right
            val_ml = tl.load(A_ptr + i * N + (j-1))      # middle-left
            val_mc = tl.load(A_ptr + i * N + j)          # middle-center
            val_mr = tl.load(A_ptr + i * N + (j+1))      # middle-right
            val_bl = tl.load(A_ptr + (i+1) * N + (j-1))  # bottom-left
            val_bc = tl.load(A_ptr + (i+1) * N + j)      # bottom-center
            val_br = tl.load(A_ptr + (i+1) * N + (j+1))  # bottom-right
            
            # Compute average
            result = (val_tl + val_tc + val_tr + val_ml + val_mc + 
                     val_mr + val_bl + val_bc + val_br) / 9.0
            
            # Store result
            tl.store(A_ptr + i * N + j, result)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    for t in range(TSTEPS):
        grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
        grid_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
        grid = (grid_i, grid_j)
        
        seidel_2d_kernel[grid](A, N, BLOCK_SIZE_I, BLOCK_SIZE_J)