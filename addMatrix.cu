#include<stdio.h>
#include<math.h>
#include<sys/time.h>
#define ARRAY_SIZE_X 512
#define ARRAY_SIZE_Y 512

__global__ void sumArraysOnGPU(float* A, float* B, float* C){
  /* ここに､各スレッドの処理内容を記述*/
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = iy * ARRAY_SIZE_X + ix;
  C[idx] = A[idx] + B[idx];
}

void initializeData(float *ip, int size){
  int i;
  for(i = 0; i < size; i++){
    ip[i] = 1.0;
  }
}

void checkResult(float *ip, int size){
  int i;
  int check = 0;
  
  for(i = 0; i < size; i ++){
    if(ip[i] != 2.0){
      check = 1;
    }
  }
  
  if(check == 0){
    printf("calc success!\n");
  }else{
    printf("calc result is not correct...\n");
  }
}

int main(void){
  
    
  /* hostメモリの宣言 */
  float *h_A, *h_B, *h_C;

  /* hostメモリの確保 */
  int total_array_size = ARRAY_SIZE_X * ARRAY_SIZE_Y;
  size_t nBytes = total_array_size * sizeof(float);
  
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);

  
  
  /* h_A  と h_Bを初期化 */
  initializeData(h_A, total_array_size);
  initializeData(h_B, total_array_size);
  

  /* deviceメモリの宣言　*/
  float *d_A, *d_B, *d_C;

  /* deviceメモリの確保 */  
  cudaMalloc( (float**)&d_A, nBytes );
  cudaMalloc( (float**)&d_B, nBytes );
  cudaMalloc( (float**)&d_C, nBytes );
  
  /* hostメモリからdeviceメモリにデータ転送 */
  cudaMemcpy( d_A, h_A, nBytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_B, h_B, nBytes, cudaMemcpyHostToDevice );
  cudaMemcpy( d_C, h_C, nBytes, cudaMemcpyHostToDevice );

  /* スレッド数の宣言 */
  dim3 block( ARRAY_SIZE_X );
  dim3 grid( ARRAY_SIZE_Y );

   /* set the time */
  struct timeval tv, pe;
  gettimeofday(&tv, NULL);
  
  /* カーネル関数の呼び出し */
  sumArraysOnGPU<<< block,grid >>>(d_A, d_B, d_C);
    
  /* 同期処理 */
  cudaDeviceSynchronize();
  
  /*stop the timer*/
  gettimeofday(&pe, NULL);
  float timer = pe.tv_sec - tv.tv_sec +(pe.tv_usec-tv.tv_usec)*1e-6;
  printf("%f sec\n",timer);
  
  /* deviceメモリからhostメモリにデータ転送 */
  cudaMemcpy( h_C, d_C, nBytes, cudaMemcpyDeviceToHost );

  
  
  /* 計算結果が合っているかチェック */
  checkResult(h_C, total_array_size);
  
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
