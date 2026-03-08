import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr):
    # Sequential execution - process all (i,j) pairs in the correct order
    for i in range(N-1, -1, -1):
        for j in range(i+1, N):
            # Calculate linear index for table[i][j]
            idx_ij = i * N + j
            
            # Load current value
            current_val = tl.load(table_ptr + idx_ij)
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            if j-1 >= 0:
                idx_left = i * N + (j-1)
                left_val = tl.load(table_ptr + idx_left)
                current_val = tl.maximum(current_val, left_val)
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            if i+1 < N:
                idx_down = (i+1) * N + j
                down_val = tl.load(table_ptr + idx_down)
                current_val = tl.maximum(current_val, down_val)
            
            # if (j-1>=0 && i+1<N)
            if j-1 >= 0 and i+1 < N:
                idx_diag = (i+1) * N + (j-1)
                diag_val = tl.load(table_ptr + idx_diag)
                
                # don't allow adjacent elements to bond
                if i < j-1:
                    seq_i = tl.load(seq_ptr + i)
                    seq_j = tl.load(seq_ptr + j)
                    match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                    current_val = tl.maximum(current_val, diag_val + match_val)
                else:
                    current_val = tl.maximum(current_val, diag_val)
            
            # for (k=i+1; k<j; k++)
            for k in range(i+1, j):
                idx_left_split = i * N + k
                idx_right_split = (k+1) * N + j
                left_split_val = tl.load(table_ptr + idx_left_split)
                right_split_val = tl.load(table_ptr + idx_right_split)
                current_val = tl.maximum(current_val, left_split_val + right_split_val)
            
            # Store the final result
            tl.store(table_ptr + idx_ij, current_val)

def nussinov_triton(seq, table, N):
    grid = (1,)
    nussinov_kernel[grid](seq, table, N)