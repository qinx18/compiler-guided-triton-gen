import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    # Process time steps sequentially
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        j_offsets = tl.arange(0, 128)
        j_mask = j_offsets < NY
        fict_val = tl.load(_fict_ptr + t)
        ey_boundary_offsets = 0 * NY + j_offsets  # row 0
        tl.store(ey_ptr + ey_boundary_offsets, fict_val, mask=j_mask)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        i_offsets = tl.arange(0, 64)
        j_offsets_inner = tl.arange(0, 128)
        
        for i_block in range(1, NX):
            for j_block in range(0, NY, 128):
                current_j = j_block + j_offsets_inner
                j_mask = current_j < NY
                
                ey_idx = i_block * NY + current_j
                hz_idx = i_block * NY + current_j
                hz_prev_idx = (i_block - 1) * NY + current_j
                
                ey_val = tl.load(ey_ptr + ey_idx, mask=j_mask)
                hz_val = tl.load(hz_ptr + hz_idx, mask=j_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=j_mask)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_idx, new_ey, mask=j_mask)
        
        # Step 3: Update ex[i][j] for j=1 to NY-1
        for i_block in range(0, NX, 64):
            for j_block in range(1, NY):
                current_i = i_block + i_offsets
                i_mask = current_i < NX
                
                ex_idx = current_i * NY + j_block
                hz_idx = current_i * NY + j_block
                hz_prev_idx = current_i * NY + (j_block - 1)
                
                ex_val = tl.load(ex_ptr + ex_idx, mask=i_mask)
                hz_val = tl.load(hz_ptr + hz_idx, mask=i_mask)
                hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=i_mask)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ex_ptr + ex_idx, new_ex, mask=i_mask)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        for i_block in range(0, NX - 1, 64):
            for j_block in range(0, NY - 1, 128):
                current_i = i_block + i_offsets
                current_j = j_block + j_offsets_inner
                
                i_mask = current_i < (NX - 1)
                j_mask = current_j < (NY - 1)
                combined_mask = i_mask[:, None] & j_mask[None, :]
                
                hz_idx = current_i[:, None] * NY + current_j[None, :]
                ex_idx = current_i[:, None] * NY + (current_j[None, :] + 1)
                ex_curr_idx = current_i[:, None] * NY + current_j[None, :]
                ey_idx = (current_i[:, None] + 1) * NY + current_j[None, :]
                ey_curr_idx = current_i[:, None] * NY + current_j[None, :]
                
                hz_val = tl.load(hz_ptr + hz_idx, mask=combined_mask)
                ex_next = tl.load(ex_ptr + ex_idx, mask=combined_mask)
                ex_curr = tl.load(ex_ptr + ex_curr_idx, mask=combined_mask)
                ey_next = tl.load(ey_ptr + ey_idx, mask=combined_mask)
                ey_curr = tl.load(ey_ptr + ey_curr_idx, mask=combined_mask)
                
                new_hz = hz_val - 0.7 * ((ex_next - ex_curr) + (ey_next - ey_curr))
                tl.store(hz_ptr + hz_idx, new_hz, mask=combined_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    grid = (1,)
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX
    )