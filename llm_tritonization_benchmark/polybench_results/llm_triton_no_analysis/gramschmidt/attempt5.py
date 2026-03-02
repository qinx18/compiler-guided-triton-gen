import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(
    A_ptr, Q_ptr, R_ptr,
    M, N,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each block handles one column j for the current k
    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Calculate which j column this block is processing (for j > k operations)
    j = pid_j
    
    # Calculate row indices for this block
    i_start = pid_i * BLOCK_SIZE_M
    i_offsets = tl.arange(0, BLOCK_SIZE_M)
    i_indices = i_start + i_offsets
    i_mask = i_indices < M
    
    # Process each k sequentially (outer k loop must be sequential)
    for k in range(N):
        # Step 1: Compute norm for column k (all blocks participate)
        if j == 0:  # Only first block group computes norm
            a_k_ptrs = A_ptr + i_indices * stride_A_m + k * stride_A_n
            a_k_vals = tl.load(a_k_ptrs, mask=i_mask, other=0.0)
            squared_vals = a_k_vals * a_k_vals
            partial_sum = tl.sum(squared_vals)
            
            # Atomic add to R[k,k] location
            tl.atomic_add(R_ptr + k * stride_R_m + k * stride_R_n, partial_sum)
        
        # Synchronization point - wait for norm computation
        tl.debug_barrier()
        
        # Step 2: Finalize norm and compute Q column (all blocks participate)
        if (j == 0) & (pid_i == 0) & (i_start == 0):
            # Only one thread finalizes the square root
            r_kk_ptr = R_ptr + k * stride_R_m + k * stride_R_n
            nrm_squared = tl.load(r_kk_ptr)
            nrm = tl.sqrt(nrm_squared)
            tl.store(r_kk_ptr, nrm)
        
        tl.debug_barrier()
        
        # All blocks compute their portion of Q[:, k]
        if j == 0:
            r_kk = tl.load(R_ptr + k * stride_R_m + k * stride_R_n)
            a_k_ptrs = A_ptr + i_indices * stride_A_m + k * stride_A_n
            q_k_ptrs = Q_ptr + i_indices * stride_Q_m + k * stride_Q_n
            
            a_k_vals = tl.load(a_k_ptrs, mask=i_mask)
            q_k_vals = a_k_vals / r_kk
            tl.store(q_k_ptrs, q_k_vals, mask=i_mask)
        
        tl.debug_barrier()
        
        # Step 3: Process remaining columns j > k
        actual_j = k + 1 + pid_j
        if actual_j < N:
            # Initialize R[k, j] = 0 (only one thread does this)
            if (pid_i == 0) & (i_start == 0):
                tl.store(R_ptr + k * stride_R_m + actual_j * stride_R_n, 0.0)
            
            tl.debug_barrier()
            
            # Compute R[k, j] += Q[i, k] * A[i, j] for this block's rows
            q_k_ptrs = Q_ptr + i_indices * stride_Q_m + k * stride_Q_n
            a_j_ptrs = A_ptr + i_indices * stride_A_m + actual_j * stride_A_n
            
            q_k_vals = tl.load(q_k_ptrs, mask=i_mask, other=0.0)
            a_j_vals = tl.load(a_j_ptrs, mask=i_mask, other=0.0)
            
            products = q_k_vals * a_j_vals
            partial_dot = tl.sum(products)
            
            # Atomic add to R[k, j]
            tl.atomic_add(R_ptr + k * stride_R_m + actual_j * stride_R_n, partial_dot)
            
            tl.debug_barrier()
            
            # Update A[i, j] -= Q[i, k] * R[k, j]
            r_kj = tl.load(R_ptr + k * stride_R_m + actual_j * stride_R_n)
            new_a_j_vals = a_j_vals - q_k_vals * r_kj
            tl.store(a_j_ptrs, new_a_j_vals, mask=i_mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE_M = 32
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(M, BLOCK_SIZE_M)
    max_j_blocks = max(1, N - 1)  # Maximum j columns to process simultaneously
    grid_j = min(max_j_blocks, 4)  # Limit to avoid too many blocks
    
    # Zero out R matrix
    R.zero_()
    
    # Launch kernel with 2D grid
    gramschmidt_kernel[(grid_j, grid_i)](
        A, Q, R,
        M, N,
        A.stride(0), A.stride(1),
        Q.stride(0), Q.stride(1),
        R.stride(0), R.stride(1),
        BLOCK_SIZE_M,
    )