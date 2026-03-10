import torch
import triton
import triton.language as tl

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr):
    # This algorithm has inherent sequential dependencies that make it unsuitable for GPU parallelization
    # Each table[i][j] depends on previously computed values in a complex pattern
    # We implement it sequentially on a single thread
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Sequential implementation matching C code exactly
    for i_offset in range(N):
        i = N - 1 - i_offset  # i from N-1 down to 0
        
        for j in range(i + 1, N):
            # Load current value
            table_idx = i * N + j
            current_val = tl.load(table_ptr + table_idx)
            
            # if (j-1>=0)
            if j - 1 >= 0:
                left_idx = i * N + (j - 1)
                left_val = tl.load(table_ptr + left_idx)
                current_val = tl.maximum(current_val, left_val)
            
            # if (i+1<N)  
            if i + 1 < N:
                top_idx = (i + 1) * N + j
                top_val = tl.load(table_ptr + top_idx)
                current_val = tl.maximum(current_val, top_val)
            
            # if (j-1>=0 && i+1<N)
            if j - 1 >= 0 and i + 1 < N:
                diag_idx = (i + 1) * N + (j - 1)
                diag_val = tl.load(table_ptr + diag_idx)
                
                if i < j - 1:
                    # Load seq values for match calculation
                    seq_i = tl.load(seq_ptr + i)
                    seq_j = tl.load(seq_ptr + j)
                    match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                    current_val = tl.maximum(current_val, diag_val + match_val)
                else:
                    current_val = tl.maximum(current_val, diag_val)
            
            # Inner k loop: for (k=i+1; k<j; k++)
            for k in range(i + 1, j):
                left_k_idx = i * N + k
                right_k_idx = (k + 1) * N + j
                left_k_val = tl.load(table_ptr + left_k_idx)
                right_k_val = tl.load(table_ptr + right_k_idx)
                current_val = tl.maximum(current_val, left_k_val + right_k_val)
            
            # Store updated value
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Launch with single thread since algorithm is inherently sequential
    grid = (1,)
    nussinov_kernel[grid](seq, table, N)