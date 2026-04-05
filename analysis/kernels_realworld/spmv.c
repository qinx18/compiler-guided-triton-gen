/* miniFE: CSR Sparse Matrix-Vector Multiply (SpMV)
 *
 * From Mantevo/miniFE — an ECP proxy application for implicit
 * unstructured finite element computations. SpMV is the dominant
 * hotspot in the Conjugate Gradient solver.
 *
 * Computes y = A * x where A is stored in Compressed Sparse Row (CSR) format:
 *   - row_offsets[NROWS+1]: start/end indices into cols/vals for each row
 *   - cols[NNZ]: column indices of nonzero entries
 *   - vals[NNZ]: values of nonzero entries
 *
 * Key characteristics:
 *   - Outer loop over rows is embarrassingly parallel (each row independent)
 *   - Inner loop is a variable-length reduction (sum += vals[i]*x[cols[i]])
 *   - Indirect (gather) access pattern on x via cols array
 *   - For 3D hex8 FE, each row has ~27 nonzeros (short, uniform inner loop)
 *
 * Simplified from miniFE SparseMatrix_functions.hpp matvec_std operator.
 * Original: https://github.com/Mantevo/miniFE
 */

#define NROWS 4096
#define NNZ_PER_ROW 27
#define NNZ (NROWS * NNZ_PER_ROW)
#define NCOLS NROWS

/* CSR matrix storage */
int row_offsets[NROWS + 1];
int cols[NNZ];
float vals[NNZ];

/* Vectors */
float x[NCOLS];
float y[NROWS];

void spmv_kernel() {
    int row, i;
    float sum;

#pragma scop
    for (row = 0; row < NROWS; row++) {
        sum = 0.0f;
        for (i = row_offsets[row]; i < row_offsets[row + 1]; i++) {
            sum += vals[i] * x[cols[i]];
        }
        y[row] = sum;
    }
#pragma endscop
}
