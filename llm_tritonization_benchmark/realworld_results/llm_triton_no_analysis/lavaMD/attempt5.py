import triton
import triton.language as tl
import torch

@triton.jit
def lavaMD_kernel(
    box_nei_ptr, box_nn_ptr, box_offset_ptr,
    fv_v_ptr, fv_x_ptr, fv_y_ptr, fv_z_ptr,
    qv_ptr, rv_v_ptr, rv_x_ptr, rv_y_ptr, rv_z_ptr,
    alpha_val,
    NUMBER_BOXES: tl.constexpr,
    MAX_NEIGHBORS: tl.constexpr,
    NUMBER_PAR_PER_BOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each block processes one box
    l = pid
    if l >= NUMBER_BOXES:
        return
    
    first_i = tl.load(box_offset_ptr + l)
    nn = tl.load(box_nn_ptr + l)
    
    a2 = 2.0 * alpha_val * alpha_val
    
    # Process home box (k=0) and neighbor boxes (k=1..nn)
    for k in range(1 + MAX_NEIGHBORS + 1):
        continue_loop = k < (1 + nn)
        if continue_loop:
            # Get pointer to box: home box or neighbor
            if k == 0:
                pointer = l
            else:
                neighbor_idx = l * MAX_NEIGHBORS + (k - 1)
                pointer = tl.load(box_nei_ptr + neighbor_idx)
                
            first_j = tl.load(box_offset_ptr + pointer)
            
            # All-pairs interaction: home particles × neighbor particles
            for i in range(NUMBER_PAR_PER_BOX):
                idx_i = first_i + i
                
                # Load particle i data
                rv_v_i = tl.load(rv_v_ptr + idx_i)
                rv_x_i = tl.load(rv_x_ptr + idx_i)
                rv_y_i = tl.load(rv_y_ptr + idx_i)
                rv_z_i = tl.load(rv_z_ptr + idx_i)
                
                # Initialize accumulators for this particle with explicit type
                acc_fv_v = rv_v_i * 0.0  # Use same type as loaded data
                acc_fv_x = rv_v_i * 0.0
                acc_fv_y = rv_v_i * 0.0
                acc_fv_z = rv_v_i * 0.0
                
                for j in range(NUMBER_PAR_PER_BOX):
                    idx_j = first_j + j
                    
                    # Load particle j data
                    rv_v_j = tl.load(rv_v_ptr + idx_j)
                    rv_x_j = tl.load(rv_x_ptr + idx_j)
                    rv_y_j = tl.load(rv_y_ptr + idx_j)
                    rv_z_j = tl.load(rv_z_ptr + idx_j)
                    qv_j = tl.load(qv_ptr + idx_j)
                    
                    # Distance squared via precomputed v values
                    r2 = rv_v_i + rv_v_j - (rv_x_i * rv_x_j + rv_y_i * rv_y_j + rv_z_i * rv_z_j)
                    
                    u2 = a2 * r2
                    vij = tl.exp(-u2)
                    fs = 2.0 * vij
                    
                    dx = rv_x_i - rv_x_j
                    dy = rv_y_i - rv_y_j
                    dz = rv_z_i - rv_z_j
                    
                    fxij = fs * dx
                    fyij = fs * dy
                    fzij = fs * dz
                    
                    # Accumulate forces
                    acc_fv_v += qv_j * vij
                    acc_fv_x += qv_j * fxij
                    acc_fv_y += qv_j * fyij
                    acc_fv_z += qv_j * fzij
                
                # Atomically add accumulated forces to global memory
                tl.atomic_add(fv_v_ptr + idx_i, acc_fv_v)
                tl.atomic_add(fv_x_ptr + idx_i, acc_fv_x)
                tl.atomic_add(fv_y_ptr + idx_i, acc_fv_y)
                tl.atomic_add(fv_z_ptr + idx_i, acc_fv_z)

def lavaMD_triton(box_nei, box_nn, box_offset, fv_v, fv_x, fv_y, fv_z, qv, rv_v, rv_x, rv_y, rv_z, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX):
    NUMBER_BOXES = BOXES1D * BOXES1D * BOXES1D
    
    # Initialize force arrays to zero
    fv_v.zero_()
    fv_x.zero_()
    fv_y.zero_()
    fv_z.zero_()
    
    BLOCK_SIZE = 128
    
    grid = (NUMBER_BOXES,)
    
    lavaMD_kernel[grid](
        box_nei, box_nn, box_offset,
        fv_v, fv_x, fv_y, fv_z,
        qv, rv_v, rv_x, rv_y, rv_z,
        alpha_val,
        NUMBER_BOXES, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX,
        BLOCK_SIZE
    )