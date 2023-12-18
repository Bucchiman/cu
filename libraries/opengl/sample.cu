/*
 * FileName:     sample
 * Author:       8ucchiman
 * CreatedDate:  2023-05-18 16:02:36
 * LastModified: 2023-02-26 13:30:39 +0900
 * Reference:    8ucchiman.jp
 * Description:  ---
 */


#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// cudaのエラー検出用マクロ
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

// 画面の解像度
#define WIDTH 1024
#define HEIGHT 1024

// pixel buffer object
GLuint pbo;

// フレームバッファの取得に使用
cudaGraphicsResource *dev_resource;

// 画面更新の間隔 (ms)
int interval = 16;

// カーネル関数: bitmapに適当に色を塗る
__global__ void kernel(uchar4 *bitmap, int tick) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // 連続的になるように...
  float theta = tick / 60.0f * 2.0f * M_PI;
  float theta_x = x / 60.0f * 2.0f * M_PI;
  float theta_y = y / 60.0f * 2.0f * M_PI;
  float r = fabs(sin(theta + theta_x));
  float g = fabs(cos(theta + theta_y));
  float b = fabs(sin(theta + theta_x) * cos(theta + theta_y));

  bitmap[offset].x = (unsigned char)(r * 255);
  bitmap[offset].y = (unsigned char)(g * 255);
  bitmap[offset].z = (unsigned char)(b * 255);
  bitmap[offset].w = 255;
}

// 描画用コールバック
void draw() {
  // ピクセルバッファオブジェクトがバインドされているので、サイズを指定するだけで良い
  glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

// 画面を更新するコールバック
void update(int key) {
  static int tick = 0;  // 今何フレーム目?
  uchar4 *dev_bitmap;
  size_t size;

  // フレームバッファをマップしてアドレスを取得
  EXIT_IF_FAIL(cudaGraphicsMapResources(1, &dev_resource, NULL));
  EXIT_IF_FAIL(cudaGraphicsResourceGetMappedPointer(
      (void **)&dev_bitmap, &size, dev_resource));

  // カーネル関数を呼ぶ
  dim3 threads(8, 8);                 // 64スレッド/1グリッド
  dim3 grids(WIDTH / 8, HEIGHT / 8);  // 各ピクセルに1スレッドが割り振られる
  kernel<<<grids, threads>>>(dev_bitmap, tick);

  // カーネル関数の終了を待つ
  EXIT_IF_FAIL(cudaDeviceSynchronize());

  // リソースの開放
  EXIT_IF_FAIL(cudaGraphicsUnmapResources(1, &dev_resource, NULL));

  // ウィンドウの再描画を要求
  glutPostRedisplay();

  // interval msec後にまた呼び出す
  glutTimerFunc(interval, update, 0);
  tick++;
}

int main(int argc, char *argv[]) {
  // OpenGLの初期化
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutCreateWindow("animation");

  // コールバックを指定
  glutDisplayFunc(draw);
  glutTimerFunc(interval, update, 0);

  // バッファを作成
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER,
               sizeof(char4) * WIDTH * HEIGHT,
               NULL,
               GL_DYNAMIC_DRAW);

  // OpenGLのバッファをCudaと共有する設定
  EXIT_IF_FAIL(cudaGraphicsGLRegisterBuffer(
      &dev_resource, pbo, cudaGraphicsMapFlagsNone));

  glutMainLoop();

  // リソースの開放(glutMainLoop()は返らないので、実際は呼ばれない)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glDeleteBuffers(1, &pbo);
  EXIT_IF_FAIL(cudaGLUnregisterBufferObject(pbo));
  EXIT_IF_FAIL(cudaGraphicsUnregisterResource(dev_resource));
}
