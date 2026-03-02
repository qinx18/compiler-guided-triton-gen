import triton
import triton.language as tl

@triton.jit
def seidel_2d_kernel(A_ptr, N, t, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    
    # Calculate base i index
    i_base = pid_i * BLOCK_SIZE + 1
    
    # Create offsets for vectorized processing
    offsets_i = tl.arange(0, BLOCK_SIZE)
    i_indices = i_base + offsets_i
    
    # Mask for valid i indices
    i_mask = (i_indices >= 1) & (i_indices <= N - 2)
    
    # Process all j values for valid i indices
    for j in range(1, N - 1):
        # Calculate addresses for all 9 neighbors
        addr_im1_jm1 = (i_indices - 1) * N + (j - 1)
        addr_im1_j = (i_indices - 1) * N + j
        addr_im1_jp1 = (i_indices - 1) * N + (j + 1)
        addr_i_jm1 = i_indices * N + (j - 1)
        addr_i_j = i_indices * N + j
        addr_i_jp1 = i_indices * N + (j + 1)
        addr_ip1_jm1 = (i_indices + 1) * N + (j - 1)
        addr_ip1_j = (i_indices + 1) * N + j
        addr_ip1_jp1 = (i_indices + 1) * N + (j + 1)
        
        # Load all 9 values with masking
        val_im1_jm1 = tl.load(A_ptr + addr_im1_jm1, mask=i_mask, other=0.0)
        val_im1_j = tl.load(A_ptr + addr_im1_j, mask=i_mask, other=0.0)
        val_im1_jp1 = tl.load(A_ptr + addr_im1_jp1, mask=i_mask, other=0.0)
        val_i_jm1 = tl.load(A_ptr + addr_i_jm1, mask=i_mask, other=0.0)
        val_i_j = tl.load(A_ptr + addr_i_j, mask=i_mask, other=0.0)
        val_i_jp1 = tl.load(A_ptr + addr_i_jp1, mask=i_mask, other=0.0)
        val_ip1_jm1 = tl.load(A_ptr + addr_ip1_jm1, mask=i_mask, other=0.0)
        val_ip1_j = tl.load(A_ptr + addr_ip1_j, mask=i_mask, other=0.0)
        val_ip1_jp1 = tl.load(A_ptr + addr_ip1_jp1, mask=i_mask, other=0.0)
        
        # Compute the average
        result = (val_im1_jm1 + val_im1_j + val_im1_jp1 + 
                 val_i_jm1 + val_i_j + val_i_jp1 +
                 val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
        
        # Store the result with masking
        tl.store(A_ptr + addr_i_j, result, mask=i_mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 32
    
    for t in range(TSTEPS):
        # Calculate grid size for i dimension
        grid_i = triton.cdiv(N - 2, BLOCK_SIZE)
        grid = (grid_i,)
        
        seidel_2d_kernel[grid](A, N, t, BLOCK_SIZE)