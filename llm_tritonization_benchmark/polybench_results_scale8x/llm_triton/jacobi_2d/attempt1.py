import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, 
                     BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr):
    row_offsets = tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_COL)
    
    for t in range(TSTEPS):
        # Phase 1: A -> B
        for row_start in range(1, N - 1, BLOCK_ROW):
            for col_start in range(1, N - 1, BLOCK_COL):
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                row_mask = rows < N - 1
                col_mask = cols < N - 1
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Load center and neighbors from A
                center_idx = rows[:, None] * N + cols[None, :]
                left_idx = rows[:, None] * N + (cols[None, :] - 1)
                right_idx = rows[:, None] * N + (cols[None, :] + 1)
                up_idx = (rows[:, None] - 1) * N + cols[None, :]
                down_idx = (rows[:, None] + 1) * N + cols[None, :]
                
                center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
                left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
                right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
                up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
                down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
                
                result = 0.2 * (center + left + right + up + down)
                tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Phase 2: B -> A
        for row_start in range(1, N - 1, BLOCK_ROW):
            for col_start in range(1, N - 1, BLOCK_COL):
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                row_mask = rows < N - 1
                col_mask = cols < N - 1
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Load center and neighbors from B
                center_idx = rows[:, None] * N + cols[None, :]
                left_idx = rows[:, None] * N + (cols[None, :] - 1)
                right_idx = rows[:, None] * N + (cols[None, :] + 1)
                up_idx = (rows[:, None] - 1) * N + cols[None, :]
                down_idx = (rows[:, None] + 1) * N + cols[None, :]
                
                center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
                left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
                right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
                up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
                down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
                
                result = 0.2 * (center + left + right + up + down)
                tl.store(A_ptr + center_idx, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_ROW = 16
    BLOCK_COL = 16
    
    kernel = jacobi_2d_kernel[(1,)](
        A, B, N, TSTEPS, BLOCK_ROW, BLOCK_COL
    )