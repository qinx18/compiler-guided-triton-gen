import triton
import triton.language as tl
import torch

def match(b1, b2):
    return (b1 + b2) == 3

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, i: tl.constexpr):
    for j in range(i + 1, N):
        # Current position
        idx = i * N + j
        
        # Get current value
        current = tl.load(table_ptr + idx)
        
        # Update from left: table[i][j-1]
        if j - 1 >= 0:
            left_idx = i * N + (j - 1)
            left_val = tl.load(table_ptr + left_idx)
            current = tl.maximum(current, left_val)
        
        # Update from below: table[i+1][j]
        if i + 1 < N:
            below_idx = (i + 1) * N + j
            below_val = tl.load(table_ptr + below_idx)
            current = tl.maximum(current, below_val)
        
        # Update from diagonal: table[i+1][j-1]
        if j - 1 >= 0 and i + 1 < N:
            diag_idx = (i + 1) * N + (j - 1)
            diag_val = tl.load(table_ptr + diag_idx)
            
            if i < j - 1:
                # Load sequence values
                seq_i = tl.load(seq_ptr + i)
                seq_j = tl.load(seq_ptr + j)
                match_val = tl.where((seq_i + seq_j) == 3, 1, 0)
                current = tl.maximum(current, diag_val + match_val)
            else:
                current = tl.maximum(current, diag_val)
        
        # Update from split points
        for k in range(i + 1, j):
            left_split_idx = i * N + k
            right_split_idx = (k + 1) * N + j
            left_split_val = tl.load(table_ptr + left_split_idx)
            right_split_val = tl.load(table_ptr + right_split_idx)
            current = tl.maximum(current, left_split_val + right_split_val)
        
        # Store result
        tl.store(table_ptr + idx, current)

def nussinov_triton(seq, table, N):
    for i in range(N-1, -1, -1):
        nussinov_kernel[(1,)](seq, table, N, i)