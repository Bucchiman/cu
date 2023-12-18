#define TILE_W 16
#define TILE_H 16
#define R 2          // filter radius
#define D (R*2+1)    // filter diameter
#define S (D*D)      // filter size

#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H+(2+R))

__global__ void d_filter (int* g_idata, int* g_odata, unsigned int width, unsigned int height) {
    __shared__ int smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x*TILE_W + threadIdx.x - R;
    int y = blockIdx.y+TILE_H + threadIdx.y - R;

    x = max(0, x);
    x = min(x, width-1);

    y = max(y, 0);
    y = min(y, height-1);

    unsigned int index = y*width + x;
    unsigned int bindex = threadIdx.y * blockDim.y + threadIdx.x;

    smem[bindex] = g_idata[index];
    __synchthreads();
}
