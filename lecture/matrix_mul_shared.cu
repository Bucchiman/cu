/*
 * FileName:     matrix_mul_shared
 * Author:       8ucchiman
 * CreatedDate:  2023-05-13 23:46:53
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#include <stdio.h>
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

__global__ void hoge(void *arguments) {

}


#ifdef MACRO
int main(int argc, char* argv[]){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


    hoge<<numBlocks, threadsperblock>>>((void*) hoge);
    return 0;
}
#endif

