import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N: tl.constexpr, TSTEPS: tl.constexpr, 
                     BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr):
    row_offsets = tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_COL)
    
    for t in range(TSTEPS):
        # Phase 1: Update B from A
        for row_start in range(1, N - 1, BLOCK_ROW):
            for col_start in range(1, N - 1, BLOCK_COL):
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                row_mask = rows < N - 1
                col_mask = cols < N - 1
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Load stencil values
                center_idx = rows[:, None] * N + cols[None, :]
                left_idx = rows[:, None] * N + (cols[None, :] - 1)
                right_idx = rows[:, None] * N + (cols[None, :] + 1)
                up_idx = (rows[:, None] - 1) * N + cols[None, :]
                down_idx = (rows[:, None] + 1) * N + cols[None, :]
                
                a_center = tl.load(A + center_idx, mask=mask)
                a_left = tl.load(A + left_idx, mask=mask)
                a_right = tl.load(A + right_idx, mask=mask)
                a_up = tl.load(A + up_idx, mask=mask)
                a_down = tl.load(A + down_idx, mask=mask)
                
                result = 0.2 * (a_center + a_left + a_right + a_down + a_up)
                tl.store(B + center_idx, result, mask=mask)
        
        # Phase 2: Update A from B
        for row_start in range(1, N - 1, BLOCK_ROW):
            for col_start in range(1, N - 1, BLOCK_COL):
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                row_mask = rows < N - 1
                col_mask = cols < N - 1
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Load stencil values
                center_idx = rows[:, None] * N + cols[None, :]
                left_idx = rows[:, None] * N + (cols[None, :] - 1)
                right_idx = rows[:, None] * N + (cols[None, :] + 1)
                up_idx = (rows[:, None] - 1) * N + cols[None, :]
                down_idx = (rows[:, None] + 1) * N + cols[None, :]
                
                b_center = tl.load(B + center_idx, mask=mask)
                b_left = tl.load(B + left_idx, mask=mask)
                b_right = tl.load(B + right_idx, mask=mask)
                b_up = tl.load(B + up_idx, mask=mask)
                b_down = tl.load(B + down_idx, mask=mask)
                
                result = 0.2 * (b_center + b_left + b_right + b_down + b_up)
                tl.store(A + center_idx, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_ROW = 16
    BLOCK_COL = 16
    
    jacobi_2d_kernel[(1,)](
        A, B, N, TSTEPS, BLOCK_ROW, BLOCK_COL
    )