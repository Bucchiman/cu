/*
 * FileName:     hello
 * Author:       8ucchiman
 * CreatedDate:  2023-05-10 14:03:10
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    https://github.com/olcf/cuda-training-series.git
 * Description:  ---
 */


#include <stdio.h>
#define MACRO


__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx, threadIdx);
}


#ifdef MACRO
int main(int argc, char* argv[]){
    hello<<<2, 3>>>();
    printf("before cudaDeviceSynchronize: Bucchiman was here\n");
    cudaDeviceSynchronize();
    printf("after cudaDeviceSynchronize: Bucchiman was here\n");
    return 0;
}
#endif


/*
 * <<<2, 3>>>            ... block: 2, thread: 3
 * cudaDeviceSynchronize ... カーネル関数はホスト関数とは非同期に処理されるため、同期を図るべく、宣言
 * 結果は以下のようになり、threadIdxが0のままになっていると思われる。
 * そこで、明示的にblockIdx.x, threadIdx.xとして指定することで直る。
 * hello_solution.cuはそのサンプル
 */

