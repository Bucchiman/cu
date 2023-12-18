/*
 * FileName:     main
 * Author:       8ucchiman
 * CreatedDate:  2023-05-18 13:36:33
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    https://yuki67.github.io/post/cuda_animation/
 * Description:  ---
 */


#include <stdio.h>
#include <iostream>
#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cmath>
#define MACRO
#define WIDTH 1024
#define HEIGHT 1024


#define EXIT_IF_FAIL(call)                                                 \
  do {                                                                     \
    (call);                                                                \
    cudaError_t err = cudaGetLastError();                                  \
    if (err != cudaSuccess) {                                              \
      std::cout << "error in file " << __FILE__ << " line at " << __LINE__ \
                << ": " << cudaGetErrorString(err) << std::endl;           \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

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

GLuint pbo;
cudaGraphicsResource* dev_resource;

int interval = 16;

// カーネル関数: bitmapに適当に色を塗る
__global__ void kernel(uchar4 *bitmap, int tick) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float theta = tick / 60.0f * 2.0f * M_PI;
    float theta_x = x / 60.0f * 2.0f * M_PI;
    float theta_y = y / 60.0f * 2.0f * M_PI;
    float r = fabs(sin(theta + theta_x));
    float g = fabs(cos(theta + theta_y));
    float b = fabs(sin(theta + theta_x) * cos(theta + theta_y));

    bitmap[offset].x = (unsigned char) (r * 255);
    bitmap[offset].y = (unsigned char) (g * 255);
    bitmap[offset].z = (unsigned char) (b * 255);
    bitmap[offset].w = 255;
}

// 描画用のコールバック関数
void draw() {
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

// 画面更新用のコールバック関数
void update (int key) {
    static int tick = 0;
    uchar4 *dev_bitmap;
    size_t size;

    // フレームバッファをマップしてアドレスを取得
    EXIT_IF_FAIL(cudaGraphicsMapResources(1, &dev_resource, NULL));
    EXIT_IF_FAIL(cudaGraphicsResourceGetMappedPointer(
                (void **)&dev_bitmap, &size, dev_resource));

    // カーネル関数呼び出し
    dim3 threads(8, 8);
    dim3 grids(WIDTH/8, HEIGHT/8);
    kernel<<<grids, threads>>> (dev_bitmap, tick);

    EXIT_IF_FAIL(cudaDeviceSynchronize());

    EXIT_IF_FAIL(cudaGraphicsUnmapResources(1, &dev_resource, NULL));

    glutPostRedisplay();

    glutTimerFunc(interval, update, 0);
    tick++;
}


#ifdef MACRO
int main(int argc, char* argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("animation");

    // コールバック作成
    glutDisplayFunc(draw);
    glutTimerFunc(interval, update, 0);

    // バッファを作成
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(char4)*WIDTH*HEIGHT, NULL, GL_DYNAMIC_DRAW);

    //EXIT_IF_FAIL(cudaGraphicsGLRegisterBuffer(&dev_resource, pbo, cudaGraphicsMapFlagsNone));
    cudaGraphicsGLRegisterBuffer(&dev_resource, pbo, cudaGraphicsMapFlagsNone);

    glutMainLoop();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo);
    //EXIT_IF_FAIL(cudaGLUnregisterBufferObject(pbo));
    EXIT_IF_FAIL(cudaGraphicsUnregisterResource(dev_resource));
    return 0;
}
#endif

