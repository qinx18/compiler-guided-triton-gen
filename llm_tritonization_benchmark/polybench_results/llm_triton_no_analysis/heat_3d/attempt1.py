import triton
import triton.language as tl

@triton.jit
def heat_3d_kernel(A, B, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block indices
    pid = tl.program_id(0)
    
    # Calculate total number of interior points
    interior_size = (N - 2) * (N - 2) * (N - 2)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < interior_size
    
    # Convert linear indices to 3D coordinates (i, j, k) in interior region
    # Interior coordinates are from 1 to N-2
    temp = indices
    k_interior = temp % (N - 2)
    temp = temp // (N - 2)
    j_interior = temp % (N - 2)
    i_interior = temp // (N - 2)
    
    # Convert to actual array coordinates
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1
    
    # Compute linear indices for 3D array access
    center_idx = i * N * N + j * N + k
    
    # Load center values
    A_center = tl.load(A + center_idx, mask=mask)
    
    # Load neighboring values for stencil computation
    A_ip1 = tl.load(A + (i + 1) * N * N + j * N + k, mask=mask)
    A_im1 = tl.load(A + (i - 1) * N * N + j * N + k, mask=mask)
    A_jp1 = tl.load(A + i * N * N + (j + 1) * N + k, mask=mask)
    A_jm1 = tl.load(A + i * N * N + (j - 1) * N + k, mask=mask)
    A_kp1 = tl.load(A + i * N * N + j * N + (k + 1), mask=mask)
    A_km1 = tl.load(A + i * N * N + j * N + (k - 1), mask=mask)
    
    # Compute B values
    B_val = (0.125 * (A_ip1 - 2.0 * A_center + A_im1) +
             0.125 * (A_jp1 - 2.0 * A_center + A_jm1) +
             0.125 * (A_kp1 - 2.0 * A_center + A_km1) +
             A_center)
    
    # Store B values
    tl.store(B + center_idx, B_val, mask=mask)

@triton.jit
def heat_3d_kernel_ba(A, B, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block indices
    pid = tl.program_id(0)
    
    # Calculate total number of interior points
    interior_size = (N - 2) * (N - 2) * (N - 2)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < interior_size
    
    # Convert linear indices to 3D coordinates (i, j, k) in interior region
    # Interior coordinates are from 1 to N-2
    temp = indices
    k_interior = temp % (N - 2)
    temp = temp // (N - 2)
    j_interior = temp % (N - 2)
    i_interior = temp // (N - 2)
    
    # Convert to actual array coordinates
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1
    
    # Compute linear indices for 3D array access
    center_idx = i * N * N + j * N + k
    
    # Load center values
    B_center = tl.load(B + center_idx, mask=mask)
    
    # Load neighboring values for stencil computation
    B_ip1 = tl.load(B + (i + 1) * N * N + j * N + k, mask=mask)
    B_im1 = tl.load(B + (i - 1) * N * N + j * N + k, mask=mask)
    B_jp1 = tl.load(B + i * N * N + (j + 1) * N + k, mask=mask)
    B_jm1 = tl.load(B + i * N * N + (j - 1) * N + k, mask=mask)
    B_kp1 = tl.load(B + i * N * N + j * N + (k + 1), mask=mask)
    B_km1 = tl.load(B + i * N * N + j * N + (k - 1), mask=mask)
    
    # Compute A values
    A_val = (0.125 * (B_ip1 - 2.0 * B_center + B_im1) +
             0.125 * (B_jp1 - 2.0 * B_center + B_jm1) +
             0.125 * (B_kp1 - 2.0 * B_center + B_km1) +
             B_center)
    
    # Store A values
    tl.store(A + center_idx, A_val, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    # Calculate total number of interior points
    interior_size = (N - 2) * (N - 2) * (N - 2)
    
    # Block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Run time steps
    for t in range(1, TSTEPS + 1):
        # A -> B step
        heat_3d_kernel[(grid_size,)](A, B, N, BLOCK_SIZE)
        
        # B -> A step
        heat_3d_kernel_ba[(grid_size,)](A, B, N, BLOCK_SIZE)