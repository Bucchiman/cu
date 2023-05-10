/*
 * FileName:     hello_solution
 * Author:       8ucchiman
 * CreatedDate:  2023-05-10 14:22:55
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#include <stdio.h>
#define MACRO


__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}


#ifdef MACRO
int main(int argc, char* argv[]){
    hello<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
#endif

