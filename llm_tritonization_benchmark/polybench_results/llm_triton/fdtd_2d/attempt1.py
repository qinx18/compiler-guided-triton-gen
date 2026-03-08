import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
                   BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr):
    
    # Define offsets once at the start
    row_offsets = tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_COL)
    
    for t in range(TMAX):
        # Load _fict_[t]
        fict_val = tl.load(_fict_ptr + t)
        
        # Phase 1: for (j = 0; j < NY; j++) ey[0][j] = _fict_[t];
        for col_start in range(0, NY, BLOCK_COL):
            current_col_offsets = col_start + col_offsets
            col_mask = current_col_offsets < NY
            
            # ey[0][j] = _fict_[t]
            ey_indices = 0 * NY + current_col_offsets
            valid_mask = col_mask
            tl.store(ey_ptr + ey_indices, fict_val, mask=valid_mask)
        
        # Phase 2: for (i = 1; i < NX; i++) for (j = 0; j < NY; j++)
        #          ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
        for row_start in range(1, NX, BLOCK_ROW):
            for col_start in range(0, NY, BLOCK_COL):
                current_row_offsets = row_start + row_offsets
                current_col_offsets = col_start + col_offsets
                
                row_mask = current_row_offsets < NX
                col_mask = current_col_offsets < NY
                valid_mask = row_mask[:, None] & col_mask[None, :]
                
                # Calculate indices
                ey_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                hz_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                hz_prev_indices = (current_row_offsets[:, None] - 1) * NY + current_col_offsets[None, :]
                
                # Load values
                ey_vals = tl.load(ey_ptr + ey_indices, mask=valid_mask)
                hz_vals = tl.load(hz_ptr + hz_indices, mask=valid_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=valid_mask)
                
                # Compute and store
                new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ey_ptr + ey_indices, new_ey, mask=valid_mask)
        
        # Phase 3: for (i = 0; i < NX; i++) for (j = 1; j < NY; j++)
        #          ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
        for row_start in range(0, NX, BLOCK_ROW):
            for col_start in range(1, NY, BLOCK_COL):
                current_row_offsets = row_start + row_offsets
                current_col_offsets = col_start + col_offsets
                
                row_mask = current_row_offsets < NX
                col_mask = current_col_offsets < NY
                valid_mask = row_mask[:, None] & col_mask[None, :]
                
                # Calculate indices
                ex_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                hz_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                hz_prev_indices = current_row_offsets[:, None] * NY + (current_col_offsets[None, :] - 1)
                
                # Load values
                ex_vals = tl.load(ex_ptr + ex_indices, mask=valid_mask)
                hz_vals = tl.load(hz_ptr + hz_indices, mask=valid_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=valid_mask)
                
                # Compute and store
                new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ex_ptr + ex_indices, new_ex, mask=valid_mask)
        
        # Phase 4: for (i = 0; i < NX - 1; i++) for (j = 0; j < NY - 1; j++)
        #          hz[i][j] = hz[i][j] - 0.7*(ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
        for row_start in range(0, NX - 1, BLOCK_ROW):
            for col_start in range(0, NY - 1, BLOCK_COL):
                current_row_offsets = row_start + row_offsets
                current_col_offsets = col_start + col_offsets
                
                row_mask = current_row_offsets < (NX - 1)
                col_mask = current_col_offsets < (NY - 1)
                valid_mask = row_mask[:, None] & col_mask[None, :]
                
                # Calculate indices
                hz_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                ex_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                ex_next_indices = current_row_offsets[:, None] * NY + (current_col_offsets[None, :] + 1)
                ey_indices = current_row_offsets[:, None] * NY + current_col_offsets[None, :]
                ey_next_indices = (current_row_offsets[:, None] + 1) * NY + current_col_offsets[None, :]
                
                # Load values
                hz_vals = tl.load(hz_ptr + hz_indices, mask=valid_mask)
                ex_vals = tl.load(ex_ptr + ex_indices, mask=valid_mask)
                ex_next_vals = tl.load(ex_ptr + ex_next_indices, mask=valid_mask)
                ey_vals = tl.load(ey_ptr + ey_indices, mask=valid_mask)
                ey_next_vals = tl.load(ey_ptr + ey_next_indices, mask=valid_mask)
                
                # Compute and store
                new_hz = hz_vals - 0.7 * (ex_next_vals - ex_vals + ey_next_vals - ey_vals)
                tl.store(hz_ptr + hz_indices, new_hz, mask=valid_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_ROW = 16
    BLOCK_COL = 16
    
    fdtd_2d_kernel[(1,)](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX,
        BLOCK_ROW=BLOCK_ROW, BLOCK_COL=BLOCK_COL
    )