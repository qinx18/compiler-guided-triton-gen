import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 3D parallelization
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate base indices
    i_base = pid_i * BLOCK_SIZE
    j_base = pid_j * BLOCK_SIZE
    k_base = pid_k * BLOCK_SIZE
    
    # Create offset arrays
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    i_indices = i_base + i_offsets
    j_indices = j_base + j_offsets
    k_indices = k_base + k_offsets
    
    # Create masks for boundary conditions (1 <= i,j,k < N-1)
    i_mask = (i_indices >= 1) & (i_indices < N - 1)
    j_mask = (j_indices >= 1) & (j_indices < N - 1)
    k_mask = (k_indices >= 1) & (k_indices < N - 1)
    
    for i_idx in range(BLOCK_SIZE):
        if i_base + i_idx >= 1 and i_base + i_idx < N - 1:
            i = i_base + i_idx
            for j_idx in range(BLOCK_SIZE):
                if j_base + j_idx >= 1 and j_base + j_idx < N - 1:
                    j = j_base + j_idx
                    for k_idx in range(BLOCK_SIZE):
                        if k_base + k_idx >= 1 and k_base + k_idx < N - 1:
                            k = k_base + k_idx
                            
                            # Calculate 3D linear indices
                            center_idx = i * (N * N) + j * N + k
                            i_plus_idx = (i + 1) * (N * N) + j * N + k
                            i_minus_idx = (i - 1) * (N * N) + j * N + k
                            j_plus_idx = i * (N * N) + (j + 1) * N + k
                            j_minus_idx = i * (N * N) + (j - 1) * N + k
                            k_plus_idx = i * (N * N) + j * N + (k + 1)
                            k_minus_idx = i * (N * N) + j * N + (k - 1)
                            
                            # Load values from A
                            a_center = tl.load(A_ptr + center_idx)
                            a_i_plus = tl.load(A_ptr + i_plus_idx)
                            a_i_minus = tl.load(A_ptr + i_minus_idx)
                            a_j_plus = tl.load(A_ptr + j_plus_idx)
                            a_j_minus = tl.load(A_ptr + j_minus_idx)
                            a_k_plus = tl.load(A_ptr + k_plus_idx)
                            a_k_minus = tl.load(A_ptr + k_minus_idx)
                            
                            # Compute heat equation
                            b_val = (0.125 * (a_i_plus - 2.0 * a_center + a_i_minus) +
                                   0.125 * (a_j_plus - 2.0 * a_center + a_j_minus) +
                                   0.125 * (a_k_plus - 2.0 * a_center + a_k_minus) +
                                   a_center)
                            
                            # Store to B
                            tl.store(B_ptr + center_idx, b_val)

@triton.jit
def heat_3d_kernel_b_to_a(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 3D parallelization
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate base indices
    i_base = pid_i * BLOCK_SIZE
    j_base = pid_j * BLOCK_SIZE
    k_base = pid_k * BLOCK_SIZE
    
    for i_idx in range(BLOCK_SIZE):
        if i_base + i_idx >= 1 and i_base + i_idx < N - 1:
            i = i_base + i_idx
            for j_idx in range(BLOCK_SIZE):
                if j_base + j_idx >= 1 and j_base + j_idx < N - 1:
                    j = j_base + j_idx
                    for k_idx in range(BLOCK_SIZE):
                        if k_base + k_idx >= 1 and k_base + k_idx < N - 1:
                            k = k_base + k_idx
                            
                            # Calculate 3D linear indices
                            center_idx = i * (N * N) + j * N + k
                            i_plus_idx = (i + 1) * (N * N) + j * N + k
                            i_minus_idx = (i - 1) * (N * N) + j * N + k
                            j_plus_idx = i * (N * N) + (j + 1) * N + k
                            j_minus_idx = i * (N * N) + (j - 1) * N + k
                            k_plus_idx = i * (N * N) + j * N + (k + 1)
                            k_minus_idx = i * (N * N) + j * N + (k - 1)
                            
                            # Load values from B
                            b_center = tl.load(B_ptr + center_idx)
                            b_i_plus = tl.load(B_ptr + i_plus_idx)
                            b_i_minus = tl.load(B_ptr + i_minus_idx)
                            b_j_plus = tl.load(B_ptr + j_plus_idx)
                            b_j_minus = tl.load(B_ptr + j_minus_idx)
                            b_k_plus = tl.load(B_ptr + k_plus_idx)
                            b_k_minus = tl.load(B_ptr + k_minus_idx)
                            
                            # Compute heat equation
                            a_val = (0.125 * (b_i_plus - 2.0 * b_center + b_i_minus) +
                                   0.125 * (b_j_plus - 2.0 * b_center + b_j_minus) +
                                   0.125 * (b_k_plus - 2.0 * b_center + b_k_minus) +
                                   b_center)
                            
                            # Store to A
                            tl.store(A_ptr + center_idx, a_val)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 8
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)
    grid_k = triton.cdiv(N - 2, BLOCK_SIZE)
    
    for t in range(1, TSTEPS + 1):
        # A to B step
        heat_3d_kernel[(grid_i, grid_j, grid_k)](
            A, B, N, BLOCK_SIZE
        )
        
        # B to A step
        heat_3d_kernel_b_to_a[(grid_i, grid_j, grid_k)](
            A, B, N, BLOCK_SIZE
        )