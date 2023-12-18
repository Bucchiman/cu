/*
 * FileName:     bitmapper
 * Author:       8ucchiman
 * CreatedDate:  2023-05-18 15:19:52
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#include <stdio.h>
#include <GL/gl.h>
#include <GL/glut.h>
#define WIDTH 512
#define HEIGHT 512
#define IMAGE_SIZE_IN_BYTE (4*WIDTH*HEIGHT)
#define MACRO

#define EXIT_IF_FAIL(call)                                                     \
  do {                                                                         \
    cudaError_t retval = call;                                                 \
    if (retval != cudaSuccess) {                                               \
      printf("error in file %s line at %d: %s\n", __FILE__, __LINE__,          \
             cudaGetErrorString(retval));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct DataBlock {
    unsigned char* bitmap;
    unsigned char* dev_bitmap;
    static DataBlock* get_data() {
        static DataBlock g_data;
        return &g_data;
    }
};

__global__ void kernel(unsigned char* bitmap) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    bitmap[offset * 4 + 0] = (x / (WIDTH/255));
    bitmap[offset * 4 + 1] = (y / (HEIGHT/255));
    bitmap[offset * 4 + 2] = ((x + y) / (WIDTH + HEIGHT) / 255);
    bitmap[offset * 4 + 3] = 255;
}

static void draw() {
    DataBlock *data = DataBlock::get_data();
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, data->bitmap);
    glFlush();
}


#ifdef MACRO
int main(int argc, char* argv[]){
    DataBlock* data = DataBlock::get_data();

    data->bitmap = new unsigned char[IMAGE_SIZE_IN_BYTE];
    EXIT_IF_FAIL(cudaMalloc(&data->dev_bitmap, IMAGE_SIZE_IN_BYTE));

    dim3 threads(16, 16);
    dim3 grids(WIDTH/16, HEIGHT/16);

    kernel<<<grids, threads>>>(data->dev_bitmap);
    EXIT_IF_FAIL(cudaMemcpy(data->bitmap, data->dev_bitmap, IMAGE_SIZE_IN_BYTE, cudaMemcpyDeviceToHost));

    EXIT_IF_FAIL(cudaFree(data->dev_bitmap));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("bitmap");
    glutDisplayFunc(draw);
    glutMainLoop();

    free(data->bitmap);
    return 0;
}
#endif

