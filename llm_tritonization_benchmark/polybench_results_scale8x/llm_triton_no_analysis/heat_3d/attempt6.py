import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for t in range(1, TSTEPS + 1):
        # First phase: A -> B
        for block_start in range(0, (N-2) * (N-2) * (N-2), BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            idx = block_start + offsets
            mask = idx < (N-2) * (N-2) * (N-2)
            
            # Convert linear index to 3D coordinates (1 to N-2)
            k_idx = idx % (N-2) + 1
            temp = idx // (N-2)
            j_idx = temp % (N-2) + 1
            i_idx = temp // (N-2) + 1
            
            # Calculate 3D array indices
            center_idx = i_idx * N * N + j_idx * N + k_idx
            
            # Load values for stencil computation
            A_center = tl.load(A_ptr + center_idx, mask=mask)
            A_ip1 = tl.load(A_ptr + center_idx + N * N, mask=mask)
            A_im1 = tl.load(A_ptr + center_idx - N * N, mask=mask)
            A_jp1 = tl.load(A_ptr + center_idx + N, mask=mask)
            A_jm1 = tl.load(A_ptr + center_idx - N, mask=mask)
            A_kp1 = tl.load(A_ptr + center_idx + 1, mask=mask)
            A_km1 = tl.load(A_ptr + center_idx - 1, mask=mask)
            
            # Compute B values
            B_val = (0.125 * (A_ip1 - 2.0 * A_center + A_im1) +
                    0.125 * (A_jp1 - 2.0 * A_center + A_jm1) +
                    0.125 * (A_kp1 - 2.0 * A_center + A_km1) +
                    A_center)
            
            tl.store(B_ptr + center_idx, B_val, mask=mask)
        
        # Second phase: B -> A
        for block_start in range(0, (N-2) * (N-2) * (N-2), BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            idx = block_start + offsets
            mask = idx < (N-2) * (N-2) * (N-2)
            
            # Convert linear index to 3D coordinates (1 to N-2)
            k_idx = idx % (N-2) + 1
            temp = idx // (N-2)
            j_idx = temp % (N-2) + 1
            i_idx = temp // (N-2) + 1
            
            # Calculate 3D array indices
            center_idx = i_idx * N * N + j_idx * N + k_idx
            
            # Load values for stencil computation
            B_center = tl.load(B_ptr + center_idx, mask=mask)
            B_ip1 = tl.load(B_ptr + center_idx + N * N, mask=mask)
            B_im1 = tl.load(B_ptr + center_idx - N * N, mask=mask)
            B_jp1 = tl.load(B_ptr + center_idx + N, mask=mask)
            B_jm1 = tl.load(B_ptr + center_idx - N, mask=mask)
            B_kp1 = tl.load(B_ptr + center_idx + 1, mask=mask)
            B_km1 = tl.load(B_ptr + center_idx - 1, mask=mask)
            
            # Compute A values
            A_val = (0.125 * (B_ip1 - 2.0 * B_center + B_im1) +
                    0.125 * (B_jp1 - 2.0 * B_center + B_jm1) +
                    0.125 * (B_kp1 - 2.0 * B_center + B_km1) +
                    B_center)
            
            tl.store(A_ptr + center_idx, A_val, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid = (1,)
    heat_3d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)