# FileName:     Makefile
# Author: 8ucchiman
# CreatedDate:  2023-02-27 15:54:39 +0900
# LastModified: 2023-02-27 16:25:36 +0900
# Reference: 8ucchiman.jp


PROGRAM := a.out
PROFILER_NAME := profile
CUDA_FILES := addMatrix.cu
PROFILER_FILES := ${PROFILER_NAME}.qdrep ${PROFILER_NAME}.sqlite

.PHONY: clean

${PROFILER_FILES}: ${PROGRAM}
	nsys nvprof -o ${PROFILER_NAME} $<  # qdrep, sqliteファイル作成

sprof: ${PROFILER_FILES}
	nsys-ui $<  # リモート上でUI表示

${PROGRAM}: ${CUDA_FILES}
	nvcc $^

