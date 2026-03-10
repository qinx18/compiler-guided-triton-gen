import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr):
    # Sequential execution within single CTA
    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            table_idx = i * N + j
            
            # Load current value
            current_val = tl.load(table_ptr + table_idx)
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            if j - 1 >= 0:
                left_idx = i * N + (j - 1)
                left_val = tl.load(table_ptr + left_idx)
                current_val = tl.maximum(current_val, left_val)
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            if i + 1 < N:
                down_idx = (i + 1) * N + j
                down_val = tl.load(table_ptr + down_idx)
                current_val = tl.maximum(current_val, down_val)
            
            # if (j-1>=0 && i+1<N)
            if (j - 1 >= 0) & (i + 1 < N):
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
            
            # for (k=i+1; k<j; k++)
            for k in range(i + 1, j):
                left_split_idx = i * N + k
                right_split_idx = (k + 1) * N + j
                left_split_val = tl.load(table_ptr + left_split_idx)
                right_split_val = tl.load(table_ptr + right_split_idx)
                current_val = tl.maximum(current_val, left_split_val + right_split_val)
            
            # Store final value
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Launch single thread block for sequential execution
    grid = (1,)
    nussinov_kernel[grid](seq, table, N)