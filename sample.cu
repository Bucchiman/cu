#include<stdio.h>
#include<math.h>
#include<sys/time.h>
#define ARRAY_SIZE 512*512

__global__ void sumArraysOnGPU(float* A, float* B, float* C) {
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = iy * 512 + ix;
  C[idx] = A[idx] + B[idx];
}


int main(void) {
  /* hostメモリ宣言 */
  float* h_A;
  float* h_B;
  float* h_C;

  /* hostメモリ確保 */
  h_A = (float*)malloc(ARRAY_SIZE*sizeof(float));
  h_B = (float*)malloc(ARRAY_SIZE*sizeof(float));
  h_C = (float*)malloc(ARRAY_SIZE*sizeof(float));

  /* deviceメモリの宣言 */
  float* d_A;
  float* d_B;
  float* d_C;

  /* deviceメモリの確保 */
  cudaMalloc((float**)&d_A, ARRAY_SIZE*sizeof(float));
  cudaMalloc((float**)&d_B, ARRAY_SIZE*sizeof(float));
  cudaMalloc((float**)&d_C, ARRAY_SIZE*sizeof(float));

  /* hostメモリからdeviceメモリにデータ転送 */
  cudaMemcpy(d_A, h_A, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  /* スレッド数の宣言 */
  dim3 block(512);
  dim3 grid(512);

  /* set the time */
  struct timeval tv, pe;
  gettimeofday(&tv, NULL);

  /* カーネル関数の呼び出し*/
  sumArraysOnGPU <<<block, grid>>>(d_A, d_B, d_C);

  /* 同期処理 */
  cudaDeviceSynchronize();

  /* stop the timer */
  gettimeofday(&pe, NULL);
  float timer = pe.tv_sec - tv.tv_sec + (pe.tv_usec - tv.tv_usec)*1e-6;
  printf("%f sec\n", timer);

  /* deviceメモリからhostメモリにデータ転送 */
  cudaMemcpy(h_C, d_C, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  /* ホストメモリの解放 */
  free(h_A);
  free(h_B);
  free(h_C);

  /* デバイスメモリの解放 */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
