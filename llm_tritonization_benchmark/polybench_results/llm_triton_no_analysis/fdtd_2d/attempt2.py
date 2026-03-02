import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    # Define offset arrays once at the start
    j_offsets = tl.arange(0, 128)
    i_offsets = tl.arange(0, 64)
    
    # Process time steps sequentially
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        fict_val = tl.load(_fict_ptr + t)
        for j_block in range(0, NY, 128):
            current_j = j_block + j_offsets
            j_mask = current_j < NY
            ey_boundary_offsets = 0 * NY + current_j  # row 0
            tl.store(ey_ptr + ey_boundary_offsets, fict_val, mask=j_mask)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        for i in range(1, NX):
            for j_block in range(0, NY, 128):
                current_j = j_block + j_offsets
                j_mask = current_j < NY
                
                ey_idx = i * NY + current_j
                hz_idx = i * NY + current_j
                hz_prev_idx = (i - 1) * NY + current_j
                
                ey_val = tl.load(ey_ptr + ey_idx, mask=j_mask)
                hz_val = tl.load(hz_ptr + hz_idx, mask=j_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=j_mask)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_idx, new_ey, mask=j_mask)
        
        # Step 3: Update ex[i][j] for j=1 to NY-1
        for j in range(1, NY):
            for i_block in range(0, NX, 64):
                current_i = i_block + i_offsets
                i_mask = current_i < NX
                
                ex_idx = current_i * NY + j
                hz_idx = current_i * NY + j
                hz_prev_idx = current_i * NY + (j - 1)
                
                ex_val = tl.load(ex_ptr + ex_idx, mask=i_mask)
                hz_val = tl.load(hz_ptr + hz_idx, mask=i_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=i_mask)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ex_ptr + ex_idx, new_ex, mask=i_mask)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz_idx = i * NY + j
                ex_idx = i * NY + (j + 1)
                ex_curr_idx = i * NY + j
                ey_idx = (i + 1) * NY + j
                ey_curr_idx = i * NY + j
                
                hz_val = tl.load(hz_ptr + hz_idx)
                ex_next = tl.load(ex_ptr + ex_idx)
                ex_curr = tl.load(ex_ptr + ex_curr_idx)
                ey_next = tl.load(ey_ptr + ey_idx)
                ey_curr = tl.load(ey_ptr + ey_curr_idx)
                
                new_hz = hz_val - 0.7 * ((ex_next - ex_curr) + (ey_next - ey_curr))
                tl.store(hz_ptr + hz_idx, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    grid = (1,)
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX
    )