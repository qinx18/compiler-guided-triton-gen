import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 3D blocking
    pid = tl.program_id(0)
    
    # Calculate total number of inner elements
    inner_size = (N - 2) * (N - 2) * (N - 2)
    
    # Calculate block start position
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = current_offsets < inner_size
    
    for t in range(TSTEPS):
        # First phase: A -> B
        for block_idx in range(0, inner_size, BLOCK_SIZE):
            block_offsets = block_idx + offsets
            block_mask = block_offsets < inner_size
            
            # Convert linear indices to 3D coordinates (i, j, k)
            # where i, j, k are in range [1, N-2]
            linear_idx = block_offsets
            k_inner = linear_idx % (N - 2)
            j_inner = (linear_idx // (N - 2)) % (N - 2)
            i_inner = linear_idx // ((N - 2) * (N - 2))
            
            # Convert to actual coordinates
            i = i_inner + 1
            j = j_inner + 1
            k = k_inner + 1
            
            # Calculate 3D array indices
            center_idx = i * N * N + j * N + k
            
            # Load center and neighbor values
            center_val = tl.load(A_ptr + center_idx, mask=block_mask)
            
            # i-direction neighbors
            i_plus = tl.load(A_ptr + center_idx + N * N, mask=block_mask)
            i_minus = tl.load(A_ptr + center_idx - N * N, mask=block_mask)
            
            # j-direction neighbors
            j_plus = tl.load(A_ptr + center_idx + N, mask=block_mask)
            j_minus = tl.load(A_ptr + center_idx - N, mask=block_mask)
            
            # k-direction neighbors
            k_plus = tl.load(A_ptr + center_idx + 1, mask=block_mask)
            k_minus = tl.load(A_ptr + center_idx - 1, mask=block_mask)
            
            # Calculate stencil result
            result = (0.125 * (i_plus - 2.0 * center_val + i_minus) +
                     0.125 * (j_plus - 2.0 * center_val + j_minus) +
                     0.125 * (k_plus - 2.0 * center_val + k_minus) +
                     center_val)
            
            # Store to B
            tl.store(B_ptr + center_idx, result, mask=block_mask)
        
        # Second phase: B -> A
        for block_idx in range(0, inner_size, BLOCK_SIZE):
            block_offsets = block_idx + offsets
            block_mask = block_offsets < inner_size
            
            # Convert linear indices to 3D coordinates (i, j, k)
            linear_idx = block_offsets
            k_inner = linear_idx % (N - 2)
            j_inner = (linear_idx // (N - 2)) % (N - 2)
            i_inner = linear_idx // ((N - 2) * (N - 2))
            
            # Convert to actual coordinates
            i = i_inner + 1
            j = j_inner + 1
            k = k_inner + 1
            
            # Calculate 3D array indices
            center_idx = i * N * N + j * N + k
            
            # Load center and neighbor values from B
            center_val = tl.load(B_ptr + center_idx, mask=block_mask)
            
            # i-direction neighbors
            i_plus = tl.load(B_ptr + center_idx + N * N, mask=block_mask)
            i_minus = tl.load(B_ptr + center_idx - N * N, mask=block_mask)
            
            # j-direction neighbors
            j_plus = tl.load(B_ptr + center_idx + N, mask=block_mask)
            j_minus = tl.load(B_ptr + center_idx - N, mask=block_mask)
            
            # k-direction neighbors
            k_plus = tl.load(B_ptr + center_idx + 1, mask=block_mask)
            k_minus = tl.load(B_ptr + center_idx - 1, mask=block_mask)
            
            # Calculate stencil result
            result = (0.125 * (i_plus - 2.0 * center_val + i_minus) +
                     0.125 * (j_plus - 2.0 * center_val + j_minus) +
                     0.125 * (k_plus - 2.0 * center_val + k_minus) +
                     center_val)
            
            # Store back to A
            tl.store(A_ptr + center_idx, result, mask=block_mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    inner_size = (N - 2) * (N - 2) * (N - 2)
    
    if inner_size == 0:
        return
    
    grid = (triton.cdiv(inner_size, BLOCK_SIZE),)
    
    heat_3d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )