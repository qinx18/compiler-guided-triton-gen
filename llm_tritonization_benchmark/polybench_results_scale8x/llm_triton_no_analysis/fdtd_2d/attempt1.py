import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t] for j = 0 to NY-1
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_start in range(0, NY, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < NY
            
            ey_indices = 0 * NY + j_indices
            fict_val = tl.load(_fict_ptr + t)
            tl.store(ey_ptr + ey_indices, fict_val, mask=j_mask)
        
        # Step 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]) for i = 1 to NX-1, j = 0 to NY-1
        ij_offsets = tl.arange(0, BLOCK_SIZE)
        for i in range(1, NX):
            for j_start in range(0, NY, BLOCK_SIZE):
                j_indices = j_start + ij_offsets
                j_mask = j_indices < NY
                
                ey_indices = i * NY + j_indices
                hz_indices = i * NY + j_indices
                hz_prev_indices = (i - 1) * NY + j_indices
                
                ey_val = tl.load(ey_ptr + ey_indices, mask=j_mask)
                hz_val = tl.load(hz_ptr + hz_indices, mask=j_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_indices, mask=j_mask)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_indices, new_ey, mask=j_mask)
        
        # Step 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]) for i = 0 to NX-1, j = 1 to NY-1
        for i in range(NX):
            for j_start in range(1, NY, BLOCK_SIZE):
                j_indices = j_start + ij_offsets
                j_mask = j_indices < NY
                
                ex_indices = i * NY + j_indices
                hz_indices = i * NY + j_indices
                hz_prev_indices = i * NY + (j_indices - 1)
                
                ex_val = tl.load(ex_ptr + ex_indices, mask=j_mask)
                hz_val = tl.load(hz_ptr + hz_indices, mask=j_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_indices, mask=j_mask)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ex_ptr + ex_indices, new_ex, mask=j_mask)
        
        # Step 4: hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]) 
        # for i = 0 to NX-2, j = 0 to NY-2
        for i in range(NX - 1):
            for j_start in range(0, NY - 1, BLOCK_SIZE):
                j_indices = j_start + ij_offsets
                j_mask = j_indices < (NY - 1)
                
                hz_indices = i * NY + j_indices
                ex_indices = i * NY + j_indices
                ex_next_indices = i * NY + (j_indices + 1)
                ey_indices = i * NY + j_indices
                ey_next_indices = (i + 1) * NY + j_indices
                
                hz_val = tl.load(hz_ptr + hz_indices, mask=j_mask)
                ex_val = tl.load(ex_ptr + ex_indices, mask=j_mask)
                ex_next_val = tl.load(ex_ptr + ex_next_indices, mask=j_mask)
                ey_val = tl.load(ey_ptr + ey_indices, mask=j_mask)
                ey_next_val = tl.load(ey_ptr + ey_next_indices, mask=j_mask)
                
                new_hz = hz_val - 0.7 * ((ex_next_val - ex_val) + (ey_next_val - ey_val))
                tl.store(hz_ptr + hz_indices, new_hz, mask=j_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    
    fdtd_2d_kernel[(1,)](
        _fict_, ex, ey, hz,
        NX, NY, TMAX,
        BLOCK_SIZE
    )