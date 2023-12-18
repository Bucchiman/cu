/*
 * FileName:     vector_add
 * Author:       8ucchiman
 * CreatedDate:  2023-05-10 14:31:09
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    https://github.com/olcf/cuda-training-series.git
 * Description:  C = A + B
 */


#include <stdio.h>
#define MACRO
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 4096;
const int block_size = 256; // CUDA maximum is 1024
// vector add kernel: C = A + B
__global__ void vadd(const float* A, const float* B, float* C, int ds) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < ds) {
        C[idx] = A[idx] + B[idx];
    }
}


#ifdef MACRO
int main(int argc, char* argv[]){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    for (int i=0; i<DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    vadd<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}
#endif



/*
 * <<<(DSIZE+block_size-1)/block_size, block_size>>>
 *     ------------------------------  ----------
 *    total 4096 threads
 *    +16(=(4096+256-1)/256) blocks---------------------+
 *    -  256 threads    256 threads   ...  256 threads  -
 *    -   ///...///      ///...///          ///...///   -
 *    +-------------------------------------------------+
 */
