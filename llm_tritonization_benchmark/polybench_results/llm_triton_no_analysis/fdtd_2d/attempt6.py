import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, NX, NY, TMAX):
    # Execute all time steps sequentially
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        fict_val = tl.load(_fict_ptr + t)
        j_offsets = tl.arange(0, 128)  # Use power of 2 for efficiency
        for j_start in range(0, NY, 128):
            current_j = j_start + j_offsets
            j_mask = current_j < NY
            ey_idx = 0 * NY + current_j
            valid_mask = j_mask & (current_j >= 0) & (current_j < NY)
            
            # Store fict_val to ey[0][j] where valid
            ey_vals = tl.full([128], fict_val, dtype=tl.float32)
            tl.store(ey_ptr + ey_idx, ey_vals, mask=valid_mask)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        i_offsets = tl.arange(0, 64)
        j_offsets_inner = tl.arange(0, 128)
        
        for i_start in range(1, NX, 64):
            for j_start in range(0, NY, 128):
                current_i = i_start + i_offsets
                current_j = j_start + j_offsets_inner
                
                i_mask = current_i < NX
                j_mask = current_j < NY
                
                # Create 2D grid
                i_expanded = current_i[:, None]
                j_expanded = current_j[None, :]
                
                # Compute indices
                ey_idx = i_expanded * NY + j_expanded
                hz_idx = i_expanded * NY + j_expanded
                hz_prev_idx = (i_expanded - 1) * NY + j_expanded
                
                # Create masks
                valid_i = (i_expanded >= 1) & (i_expanded < NX)
                valid_j = (j_expanded >= 0) & (j_expanded < NY)
                combined_mask = valid_i & valid_j & i_mask[:, None] & j_mask[None, :]
                
                # Load values
                ey_vals = tl.load(ey_ptr + ey_idx, mask=combined_mask)
                hz_vals = tl.load(hz_ptr + hz_idx, mask=combined_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=combined_mask)
                
                # Update ey
                new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ey_ptr + ey_idx, new_ey, mask=combined_mask)
        
        # Step 3: Update ex[i][j] for j=1 to NY-1
        for i_start in range(0, NX, 64):
            for j_start in range(1, NY, 128):
                current_i = i_start + i_offsets
                current_j = j_start + j_offsets_inner
                
                i_mask = current_i < NX
                j_mask = current_j < NY
                
                # Create 2D grid
                i_expanded = current_i[:, None]
                j_expanded = current_j[None, :]
                
                # Compute indices
                ex_idx = i_expanded * NY + j_expanded
                hz_idx = i_expanded * NY + j_expanded
                hz_prev_idx = i_expanded * NY + (j_expanded - 1)
                
                # Create masks
                valid_i = (i_expanded >= 0) & (i_expanded < NX)
                valid_j = (j_expanded >= 1) & (j_expanded < NY)
                combined_mask = valid_i & valid_j & i_mask[:, None] & j_mask[None, :]
                
                # Load values
                ex_vals = tl.load(ex_ptr + ex_idx, mask=combined_mask)
                hz_vals = tl.load(hz_ptr + hz_idx, mask=combined_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_idx, mask=combined_mask)
                
                # Update ex
                new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ex_ptr + ex_idx, new_ex, mask=combined_mask)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        for i_start in range(0, NX-1, 64):
            for j_start in range(0, NY-1, 128):
                current_i = i_start + i_offsets
                current_j = j_start + j_offsets_inner
                
                # Create 2D grid
                i_expanded = current_i[:, None]
                j_expanded = current_j[None, :]
                
                # Compute indices
                hz_idx = i_expanded * NY + j_expanded
                ex_idx = i_expanded * NY + j_expanded
                ex_next_idx = i_expanded * NY + (j_expanded + 1)
                ey_idx = i_expanded * NY + j_expanded
                ey_next_idx = (i_expanded + 1) * NY + j_expanded
                
                # Create masks
                valid_i = (i_expanded >= 0) & (i_expanded < (NX-1))
                valid_j = (j_expanded >= 0) & (j_expanded < (NY-1))
                i_bound = current_i < (NX-1)
                j_bound = current_j < (NY-1)
                combined_mask = valid_i & valid_j & i_bound[:, None] & j_bound[None, :]
                
                # Load values
                hz_vals = tl.load(hz_ptr + hz_idx, mask=combined_mask)
                ex_vals = tl.load(ex_ptr + ex_idx, mask=combined_mask)
                ex_next_vals = tl.load(ex_ptr + ex_next_idx, mask=combined_mask)
                ey_vals = tl.load(ey_ptr + ey_idx, mask=combined_mask)
                ey_next_vals = tl.load(ey_ptr + ey_next_idx, mask=combined_mask)
                
                # Update hz
                new_hz = hz_vals - 0.7 * ((ex_next_vals - ex_vals) + (ey_next_vals - ey_vals))
                tl.store(hz_ptr + hz_idx, new_hz, mask=combined_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    # Launch kernel with single thread block since we handle all parallelism inside
    fdtd_2d_kernel[(1,)](
        _fict_, ex, ey, hz, 
        NX, NY, TMAX
    )