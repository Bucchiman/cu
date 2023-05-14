/*
 * FileName:     stencil_1d
 * Author:       8ucchiman
 * CreatedDate:  2023-05-13 23:09:42
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#include <stdio.h>
#include <algorithm>

using namespace std;
#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

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
    int *in, *out;       // host copies of a, b, c
    int *d_in, *d_out;   // device copies of a, b, c

    in = (int *)malloc(10);
    

    hoge<<numBlocks, threadsperblock>>>((void*) hoge);
    return 0;
}
#endif

