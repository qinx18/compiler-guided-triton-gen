#define LEN_1D 32000

float a[LEN_1D], b[LEN_1D];

void s000_kernel() {
#pragma scop
    for (int i = 0; i < LEN_1D; i++) {
        a[i] = b[i] + 1;
    }
#pragma endscop
}
