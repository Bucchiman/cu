#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#define n 512
int main(void){
  struct timeval tv, pe;
  gettimeofday(&tv, NULL);

  float A[n][n],B[n][n],C[n][n];
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      A[i][j]=1.0;
      B[i][j]=1.0;
      C[i][j]=A[i][j]+B[i][j];
    }
  }
  gettimeofday(&pe,NULL);
  printf("%f sec\n",pe.tv_sec - tv.tv_sec+(pe.tv_usec-tv.tv_usec)*1e-6);
  return 0;
}
