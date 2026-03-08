import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate block bounds for this program
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Total number of inner points
    inner_size = (N - 2) * (N - 2) * (N - 2)
    mask = indices < inner_size
    
    for t in range(TSTEPS):
        # Phase 1: A -> B
        # Convert linear index to 3D coordinates
        k_coord = indices % (N - 2) + 1
        j_coord = (indices // (N - 2)) % (N - 2) + 1
        i_coord = indices // ((N - 2) * (N - 2)) + 1
        
        # Calculate linear array index
        center_idx = i_coord * N * N + j_coord * N + k_coord
        
        # Load values for stencil computation
        center = tl.load(A_ptr + center_idx, mask=mask)
        i_plus = tl.load(A_ptr + center_idx + N * N, mask=mask)
        i_minus = tl.load(A_ptr + center_idx - N * N, mask=mask)
        j_plus = tl.load(A_ptr + center_idx + N, mask=mask)
        j_minus = tl.load(A_ptr + center_idx - N, mask=mask)
        k_plus = tl.load(A_ptr + center_idx + 1, mask=mask)
        k_minus = tl.load(A_ptr + center_idx - 1, mask=mask)
        
        # Compute stencil
        result = (0.125 * (i_plus - 2.0 * center + i_minus) +
                 0.125 * (j_plus - 2.0 * center + j_minus) +
                 0.125 * (k_plus - 2.0 * center + k_minus) +
                 center)
        
        # Store to B
        tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Synchronization barrier between phases
        tl.debug_barrier()
        
        # Phase 2: B -> A
        # Load values from B for stencil computation
        center = tl.load(B_ptr + center_idx, mask=mask)
        i_plus = tl.load(B_ptr + center_idx + N * N, mask=mask)
        i_minus = tl.load(B_ptr + center_idx - N * N, mask=mask)
        j_plus = tl.load(B_ptr + center_idx + N, mask=mask)
        j_minus = tl.load(B_ptr + center_idx - N, mask=mask)
        k_plus = tl.load(B_ptr + center_idx + 1, mask=mask)
        k_minus = tl.load(B_ptr + center_idx - 1, mask=mask)
        
        # Compute stencil
        result = (0.125 * (i_plus - 2.0 * center + i_minus) +
                 0.125 * (j_plus - 2.0 * center + j_minus) +
                 0.125 * (k_plus - 2.0 * center + k_minus) +
                 center)
        
        # Store back to A
        tl.store(A_ptr + center_idx, result, mask=mask)
        
        # Synchronization barrier between timesteps
        tl.debug_barrier()

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    inner_size = (N - 2) * (N - 2) * (N - 2)
    
    if inner_size <= 0:
        return
        
    grid = (triton.cdiv(inner_size, BLOCK_SIZE),)
    
    heat_3d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )