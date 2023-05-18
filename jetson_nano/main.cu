/*
 * FileName:     main
 * Author:       8ucchiman
 * CreatedDate:  2023-05-17 16:39:00
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define MACRO
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s: %d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void addNums(int* output, int* x, int* y, int num_iters) {
    for(int i=0; i<num_iters; i++) {
        output[i] = x[i]+y[i];
    }
}


#ifdef MACRO
int main(int argc, char* argv[]){
    
    hoge<<numBlocks, threadsperblock>>>((void*) hoge);
    return 0;
}
#endif

